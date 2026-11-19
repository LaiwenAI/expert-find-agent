"""Use rival work assignments to dispatch independent full-context rechecks."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from expert_search.statistics import clustered_paired_interval
from experiments.docred import work_owner_contrast as base
from experiments.followir.core17_reranking import digest, write_json


TRAIN_SHA256 = {
    "original": "c40a137a1c57f3e2daaf5ebf511eaa1518f9305ffaa3052d46271921a91826f2",
    "extended": "6828a3eea60c41aa222326c69afa33652aa2d9d7a3a6675a55ca2d771e329c00",
}
BUDGET = .05
CANDIDATE_BAND = .20
HOLDOUT_PER_GROUP = 120
GROUPS = ("disjoint", "shared")


def holdout(original, extended, excluded):
    old = original / "train_revised.json"
    new = extended / "re2docred_train_re.json"
    if (digest(old), digest(new)) != tuple(TRAIN_SHA256.values()):
        raise ValueError("Changed independent document inputs")
    updates = {doc["title"]: doc for doc in json.loads(new.read_bytes())}

    def paired():
        for doc in json.loads(old.read_bytes()):
            if doc["title"] in excluded:
                continue
            revised = updates[doc["title"]]
            if doc["sents"] != revised["sents"] or doc["vertexSet"] != revised["vertexSet"]:
                raise ValueError("Changed aligned entity indices or text")
            yield doc, revised

    cases = list(base.cases_from_documents(paired()))
    eligible = {group: [c for c in cases if (sum(c["prior"]) > 2) == (group == "shared")]
                for group in GROUPS}
    selected = [case for group in GROUPS
                for case in sorted(eligible[group], key=lambda c: c["id"])[:HOLDOUT_PER_GROUP]]
    return selected, {group: len(items) for group, items in eligible.items()}


def score_panel(cases, read, name):
    rows = []
    for i, case in enumerate(cases):
        order = base.ARMS if i % 2 == 0 else base.ARMS[::-1]
        rows.append((case, {arm: base.observe(case, read, arm) for arm in order}))
        if (i + 1) % 60 == 0:
            print(f"{name}: {i+1}/{len(cases)} batches per arm", flush=True)
    return rows


def arrays(rows):
    compact = np.asarray([obs["shared_four"]["scores"] for _, obs in rows])
    full = np.asarray([obs["full"]["scores"] for _, obs in rows])
    known = np.asarray([case["labeled"] for case, _ in rows])
    return compact, full, known


def eligible(compact, cutoff, floor, route):
    candidate = (compact > floor) & (compact <= cutoff)
    if route == "scope_recheck":
        candidate &= compact <= compact[:, [2, 3, 0, 1]]
    return candidate


def fit(compact, full, known, baseline_cutoff, floor, route, max_calls):
    cap = int((~known).sum() * BUDGET)
    cutoffs = np.unique(np.r_[compact[compact >= baseline_cutoff], baseline_cutoff])
    full_cutoffs = np.unique(np.r_[0., full[full > 0.], np.inf])
    best = None
    for high in cutoffs:
        fast = compact > high
        uncertain = eligible(compact, high, floor, route)
        for cutoff in full_cutoffs:
            checked = uncertain if cutoff != np.inf else np.zeros_like(uncertain)
            dispatched = int(checked.any(axis=1).sum())
            if dispatched > max_calls:
                continue
            accepted = fast | (checked & (full > cutoff))
            unjudged = int((accepted & ~known).sum())
            if unjudged > cap:
                continue
            objective = (int((accepted & known).sum()), -dispatched, -unjudged, high, cutoff)
            if best is None or objective > best[0]:
                best = (objective, dict(compact=float(high), floor=floor,
                                        full=float(cutoff) if np.isfinite(cutoff) else None,
                                        calibration_full_calls=dispatched))
    return best[1]


def replay(compact, full, policy, route):
    initial = compact > policy["compact"]
    if policy["full"] is None:
        return initial, np.zeros(len(compact), dtype=bool)
    candidates = eligible(compact, policy["compact"], policy["floor"], route)
    confirmed = candidates & (full > policy["full"])
    return initial | confirmed, candidates.any(axis=1)


def evaluate(rows, policies, thresholds):
    compact, full, known = arrays(rows)
    ranking = compact - compact[:, [2, 3, 0, 1]]
    no_full = np.zeros(len(rows), dtype=bool)
    outcomes = {
        "compact_absolute": (compact > thresholds["compact"], no_full),
        "compact_work_margin": (ranking > thresholds["margin"], no_full),
        "full_absolute": (full > thresholds["full"], ~no_full),
        **{route: replay(compact, full, policy, route) for route, policy in policies.items()},
    }
    compact_ms = np.asarray([o["shared_four"]["ms"] for _, o in rows])
    full_ms = np.asarray([o["full"]["ms"] for _, o in rows])
    groups = {"combined": np.ones(len(rows), dtype=bool),
              "disjoint": known.sum(axis=1) == 2, "shared": known.sum(axis=1) > 2}
    report = {}
    for route, (accepted, dispatched) in outcomes.items():
        costs = full_ms if route == "full_absolute" else compact_ms + dispatched * full_ms
        report[route] = {}
        for group, mask in groups.items():
            refs, outs = known[mask], accepted[mask]
            complete = np.all(outs | ~refs, axis=1)
            report[route][group] = dict(
                cases=int(mask.sum()), accepted_known=int((outs & refs).sum()),
                known=int(refs.sum()), accepted_unjudged=int((outs & ~refs).sum()), unjudged=int((~refs).sum()),
                complete_known=int(complete.sum()), full_batches=int(dispatched[mask].sum()),
                p95_ms=float(np.quantile(costs[mask], .95)))
        paired = ((accepted & known).sum(axis=1) - (outcomes["compact_absolute"][0] & known).sum(axis=1)) / 4
        report[route]["paired_known_delta"] = clustered_paired_interval(paired, [case["doc"] for case, _ in rows])
    return report


def run(original, extended, checkpoint, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a fresh output and Slurm CUDA")
    development = list(base.cases(original, extended, "dev"))
    excluded = {doc["title"] for split in ("dev", "test")
                for doc, _ in base.documents(original, extended, split)}
    independent, eligible_cases = holdout(original, extended, excluded)
    tokenizer, model, revision = base.reader.load_reader(checkpoint)
    reader = base.reader.reader_scorer(tokenizer, model, system=base.SYSTEM)
    for arm in base.ARMS:
        base.observe(development[0], reader, arm)
    dev = score_panel(development, reader, "calibration")
    compact, full, known = arrays(dev)
    thresholds = dict(compact=base.calibrate(dev, "shared_four", "absolute", BUDGET),
                      margin=base.calibrate(dev, "shared_four", "other_work", BUDGET),
                      full=base.calibrate(dev, "full", "absolute", BUDGET))
    floor = base.calibrate(dev, "shared_four", "absolute", CANDIDATE_BAND)
    scope = fit(compact, full, known, thresholds["compact"], floor, "scope_recheck", len(dev))
    neutral = fit(compact, full, known, thresholds["compact"], floor, "absolute_recheck",
                  scope["calibration_full_calls"])
    policies = dict(scope_recheck=scope, absolute_recheck=neutral)
    test = score_panel(independent, reader, "held-out train documents")
    write_json(output, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
                            code_sha256=digest(Path(__file__)), base_sha256=digest(Path(base.__file__)),
                            model_revision=revision["revision"], independent_assets=TRAIN_SHA256,
                            calibration_assets={"original": base.base.DATA["dev"], "extended": base.UPDATED["dev"]},
                            heldout_eligible_by_original_labels=eligible_cases,
                            distinct_documents={"calibration": len({c["doc"] for c in development}),
                                                "test": len({c["doc"] for c in independent})},
                            thresholds=thresholds, policies=policies,
                            calibration=evaluate(dev, policies, thresholds),
                            test=evaluate(test, policies, thresholds)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("original", "extended", "checkpoint", "output"):
        parser.add_argument(arg, type=Path)
    run(**vars(parser.parse_args()))
