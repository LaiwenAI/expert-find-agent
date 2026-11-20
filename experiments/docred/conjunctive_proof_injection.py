"""Separate retrieval misses from reader misses using released two-link evidence."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from expert_search.statistics import clustered_paired_interval
from experiments.docred import conjunctive_proof_windows as proof
from experiments.docred import oracle_role_evidence as readers
from experiments.docred import person_bound_evidence as people
from experiments.docred import work_owner_contrast as owners
from experiments.followir.core17_reranking import digest, write_json


TASK = people.TASK
ROUTES = ("retrieved", "first_proof", "second_proof", "both_proofs")


def baseline(case):
    group, scored = proof.features(case)
    selected = proof.choose_windows(group, scored, "global_names")
    return proof.assign_windows(scored, selected), scored


def eligible(case):
    for indices in proof.PAIR:
        if all(1 <= len(case["annotated"][i]) <= 2 for i in indices):
            yield indices


def fit_windows(case, base, scoped, indices, route):
    gold = [tuple(case["annotated"][i]) for i in indices]
    windows = list(base)
    if route == "retrieved":
        return windows
    insert = {
        "first_proof": (0,),
        "second_proof": (1,),
        "both_proofs": (0, 1),
        "repeat_first": (0, 1),
        "repeat_second": (0, 1),
    }[route]
    repeated = {"repeat_first": 0, "repeat_second": 1}.get(route)
    for side in insert:
        i = indices[side]
        annotated = gold[side if repeated is None else repeated]
        if repeated is None:
            rest = sorted((s for s in base[i] if s not in annotated), key=lambda s: (-scoped[s, i], s))
        else:
            blocked = set(gold[0]) | set(gold[1])
            candidates = set(range(len(case["sentences"]))) - blocked
            rest = sorted(candidates, key=lambda s: (s not in base[i], -scoped[s, i], s))
        windows[i] = tuple(sorted((*annotated, *rest[:2-len(annotated)])))
    if any(len(set(w)) != 2 for w in windows):
        raise AssertionError("Changed the two-sentence prompt budget")
    return windows


def prompts(case, windows):
    return [
        (f"Does the source explicitly say that {person} had the role of {people.ROLES[case['role']]} "
         f"on the work {work}?", "\n".join(case["sentences"][j] for j in window))
        for (work, person), window in zip(case["names"], windows, strict=True)
    ]


def observe(case, read, windows):
    started = perf_counter()
    scores, clipped = read(prompts(case, windows))
    if clipped or len(scores) != 4 or not np.isfinite(scores).all():
        raise ValueError("Invalid atomic judgments")
    return np.asarray(scores), (perf_counter()-started)*1000


def fit_cutoff(panel, read):
    unjudged = []
    confirmed = []
    for case in panel:
        base, _ = baseline(case)
        logits, _ = observe(case, read, base)
        for indices, known in zip(proof.PAIR, case["known"], strict=True):
            value = min(logits[i] for i in indices)
            (confirmed if known else unjudged).append(value)
    if not unjudged:
        raise ValueError("Need unjudged calibration cases")
    cutoff = sorted(unjudged, reverse=True)[int(len(unjudged)*proof.DEV_RISK)]
    return float(cutoff), dict(unjudged=len(unjudged), accepted_unjudged=int(sum(v > cutoff for v in unjudged)),
                              known=len(confirmed), accepted_known=int(sum(v > cutoff for v in confirmed)))


def measure(panel, read, threshold, offset):
    base_counts = Counter()
    outcomes = {arm: [] for arm in ROUTES}
    groups = []
    perturbations = Counter()
    for i, case in enumerate(panel):
        base, scoped = baseline(case)
        for target in eligible(case):
            truth = tuple(set(case["annotated"][j]) for j in target)
            available = sum(gold.issubset(base[j]) for j, gold in zip(target, truth, strict=True))
            base_counts[f"{available}_links_already_shown"] += 1
            groups.append(case["doc"])
            other = next(pair for pair in proof.PAIR if pair != target)
            templates = (*ROUTES, "repeat_first", "repeat_second") if truth[0].isdisjoint(truth[1]) else ROUTES
            repeated = len(templates) > len(ROUTES)
            shift = (i + offset) % len(templates)
            observations = {}
            for route in templates[shift:] + templates[:shift]:
                windows = fit_windows(case, base, scoped, target, route)
                raw, elapsed = observe(case, read, windows)
                observations[route] = (raw[list(target)], raw[list(other)], elapsed, windows)
            for arm in ROUTES:
                values, control, elapsed, windows = observations[arm]
                shown = [gold.issubset(windows[j]) for j, gold in zip(target, truth, strict=True)]
                outcomes[arm].append(dict(
                    approved=bool(np.all(values > threshold)), accepted_links=int((values > threshold).sum()),
                    changed_untouched_neighbor=bool(np.all(control > threshold) !=
                                                    np.all(observations["retrieved"][1] > threshold)),
                    complete=all(shown), measured_ms=elapsed, original=available,
                ))
            if repeated:
                perturbations["disjoint_candidates"] += 1
                both, _, _, _ = observations["both_proofs"]
                if np.all(both > threshold):
                    perturbations["passed_both_proofs"] += 1
                    for route in ("repeat_first", "repeat_second"):
                        altered, _, _, _ = observations[route]
                        dropped = 1 if route == "repeat_first" else 0
                        if np.all(altered > threshold):
                            perturbations[f"{route}_still_passed_both"] += 1
                        if altered[dropped] <= threshold:
                            perturbations[f"{route}_withheld_link_below_threshold"] += 1
    reference = np.array([row["approved"] for row in outcomes["retrieved"]], dtype=int)
    if not len(reference):
        raise ValueError("No test candidates fit the window")
    report = {}
    for route, rows in outcomes.items():
        decisions = np.array([row["approved"] for row in rows], dtype=int)
        report[route] = dict(
            approved=int(decisions.sum()),
            approved_links=sum(row["accepted_links"] for row in rows),
            complete_annotations=sum(row["complete"] for row in rows),
            approved_with_complete_annotations=sum(row["approved"] and row["complete"] for row in rows),
            rescued=int((decisions > reference).sum()), lost=int((decisions < reference).sum()),
            changed_untouched_neighbor=sum(row["changed_untouched_neighbor"] for row in rows),
            delta_vs_retrieved=clustered_paired_interval(decisions-reference, groups),
            inference_p95_ms=float(np.quantile([row["measured_ms"] for row in rows], .95)),
            approvals_by_original_coverage={str(i): sum(row["approved"] and row["original"] == i for row in rows)
                                            for i in range(3)},
        )
    return dict(known_candidates=len(reference), documents=len(set(groups)),
                original_coverage=base_counts, cross_work_probe=perturbations, routes=report)


def run(release, large, small, output, offset):
    import torch
    from experiments.followir.core17_verifier import Reranker

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output on Slurm CUDA")
    pools, hashes = proof.load(release)
    tok, model, manifest = readers.load_reader(large)
    use_large = readers.reader_scorer(tok, model, system=owners.SYSTEM)
    reranker = Reranker(small)

    def use_small(queries):
        return reranker.score_pairs(queries, task=TASK)

    control = pools["calibration"][0]
    observations = {}
    for name, read, revision in (("qwen4b", use_large, manifest["revision"]),
                                  ("qwen06", use_small, reranker.provenance["revision"])):
        window, _ = baseline(control)
        observe(control, read, window)
        threshold, calibration = fit_cutoff(pools["calibration"], read)
        observations[name] = dict(revision=revision, cutoff=threshold, calibration=calibration,
                                  groups={split: measure(pools[split], read, threshold, offset)
                                          for split in ("test", "external")})
    write_json(output, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
                            order_offset=offset, script_sha256=digest(Path(__file__)),
                            window_builder_sha256=digest(Path(proof.__file__)), inputs=hashes, models=observations))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("release", "large", "small", "output"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("--order-offset", type=int, default=0)
    args = parser.parse_args()
    run(args.release, args.large, args.small, args.output, args.order_offset)
