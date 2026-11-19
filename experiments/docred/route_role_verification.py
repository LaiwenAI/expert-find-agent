"""Execute risk-calibrated verifier cascades on fixed two-sentence windows."""

import argparse
import json
import math
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.statistics import clustered_paired_interval
from experiments.docred import oracle_role_evidence as oracle
from experiments.docred import person_bound_evidence as base


POLICIES = {
    "lexical": ("lexical", 0),
    "reranker": ("reranker", 0),
    "reader": ("reader", 0),
    "lexical_50": ("lexical", .50),
    "lexical_75": ("lexical", .75),
    "hash_75": ("lexical", .75),
    "reranker_50": ("reranker", .50),
    "reranker_75": ("reranker", .75),
}


def lexical_scores(case, indices):
    aliases = [tuple(base.tokens(name)) for name in case["people"]]
    cues = set(base.tokens(base.ROLE_CUES[case["role"]]))
    best = [0., 0.]
    for index in indices:
        words = tuple(base.tokens(case["sentences"][index]))
        role_positions = [j for j, word in enumerate(words) if word in cues]
        for i, alias in enumerate(aliases):
            n = len(alias)
            for a in range(len(words) - n + 1):
                if words[a:a + n] == alias:
                    distances = (min(abs(a-j), abs(a+n-1-j)) for j in role_positions)
                    best[i] = max(best[i], 1. / (1 + min(distances, default=math.inf)))
    return tuple(best)


def reader_input(case, indices):
    text = "\n".join(case["sentences"][i] for i in indices)
    order = 1 - 2 * (int(case["id"][0], 16) % 2)
    queries = [f"Was {person} credited in the role of {base.ROLES[case['role']]} for {case['work']}?"
               for person in case["people"][::order]]
    return [(question, text) for question in queries], order


def score_model(score, case, indices):
    prompts, order = reader_input(case, indices)
    margins, cuts = score(prompts)
    if cuts or len(margins) != 2 or not np.isfinite(margins).all():
        raise ValueError("Invalid verifier output")
    return tuple(margins[::order])


def calibrate(cheap, reader, route_cap, risk_cap):
    # Fast accept/reject is permitted only when both scores are separated from
    # the same threshold by the confidence margin. Other pairs get deferred.
    cutoffs = np.unique(cheap)
    gates = [dict(low=float(cheap.min()-1), high=float(cheap.max()+1))]
    for center in cutoffs:
        radii = np.unique(np.abs(cheap - center))
        for radius in radii:
            gates.append(dict(low=float(center-radius), high=float(center+radius)))
    best = None
    for gate in gates:
        route = fallback(cheap, gate)
        if route.sum() > route_cap:
            continue
        fast = cheap > gate["high"]
        automatic = fast & ~route[:, None]
        risks_left = risk_cap - int(automatic[:, 1].sum())
        if risks_left < 0:
            continue
        rivals = np.sort(reader[route, 1])[::-1]
        cutoff = float(rivals[risks_left]) if risks_left < len(rivals) else float(reader.min()-1)
        answers = np.where(route[:, None], reader > cutoff, fast)
        objective = (int((answers[:, 0] & ~answers[:, 1]).sum()), -int(answers[:, 1].sum()),
                     int(answers[:, 0].sum()), -int(route.sum()))
        setting = {**gate, "reader_cutoff": cutoff}
        if best is None or objective > best[0]:
            best = (objective, setting)
    if best is None:
        raise ValueError("No feasible policy")
    return best[1]


def fallback(cheap, gate):
    return ((cheap > gate["low"]) & (cheap <= gate["high"])).any(axis=-1)


def hash_route_key(case):
    return int(base.stable("role-routing-random-control-v1", case["id"])[:12], 16)


def hash_control(cases, cheap, reader, lexical_gate, risk_cap):
    calls = int(fallback(cheap, lexical_gate).sum())
    keys = np.array([hash_route_key(case) for case in cases])
    hash_cutoff = int(np.sort(keys)[calls])
    route = keys < hash_cutoff
    fast_cutoff = lexical_gate["high"]
    risks_left = risk_cap - int(((cheap[:, 1] > fast_cutoff) & ~route).sum())
    if risks_left < 0:
        raise ValueError("Hash control exhausts risk budget before fallback")
    rivals = np.sort(reader[route, 1])[::-1]
    reader_cutoff = float(rivals[risks_left]) if risks_left < len(rivals) else float(reader.min()-1)
    return dict(hash_cutoff=hash_cutoff, cutoff=fast_cutoff, reader_cutoff=reader_cutoff)


def execute(case, policy, setting, models):
    stage, _ = POLICIES[policy]
    start = perf_counter()
    indices = oracle.matched_window(case, 2)
    if stage == "lexical":
        margins = lexical_scores(case, indices)
    else:
        margins = score_model(models[stage], case, indices)
    deferred = 0
    if "hash_cutoff" in setting:
        deferred = int(hash_route_key(case) < setting["hash_cutoff"])
    elif "low" in setting:
        deferred = int(fallback(np.asarray(margins), setting))
    if deferred:
        margins = score_model(models["reader"], case, indices)
        cutoff = setting["reader_cutoff"]
    else:
        cutoff = setting.get("high", setting.get("cutoff"))
    ms = (perf_counter() - start) * 1000
    return dict(accepted=tuple(value > cutoff for value in margins), ms=ms, deferred=deferred)


def measure(cases, settings, models):
    names = tuple(settings)
    observations = []
    for i, case in enumerate(cases):
        offset = i % len(names)
        observations.append({name: execute(case, name, settings[name], models)
                             for name in names[offset:] + names[:offset]})
    return observations


def report(cases, observations):
    baseline = [int(r["reader"]["accepted"][0] and not r["reader"]["accepted"][1]) for r in observations]
    docs = [case["doc"] for case in cases]
    hash_hits = [int(r["hash_75"]["accepted"][0] and not r["hash_75"]["accepted"][1]) for r in observations]
    results = {}
    for name, (stage, _) in POLICIES.items():
        rows = [observation[name] for observation in observations]
        accepted = [row["accepted"] for row in rows]
        both = [int(supported and not rival) for supported, rival in accepted]
        deferred = sum(row["deferred"] for row in rows)
        elapsed = [row["ms"] for row in rows]
        results[name] = dict(
            both_correct=sum(both), accepted_supported=sum(p for p, _ in accepted),
            accepted_rival=sum(r for _, r in accepted), routed_pairs=deferred,
            reader_calls=deferred + int(stage == "reader") * len(rows),
            reranker_calls=int(stage == "reranker") * len(rows),
            ms_mean=float(np.mean(elapsed)), ms_p95=float(np.quantile(elapsed, .95)),
            both_delta_vs_reader=clustered_paired_interval(np.subtract(both, baseline), docs),
        )
        if name == "lexical_75":
            results[name]["both_delta_vs_hash"] = clustered_paired_interval(np.subtract(both, hash_hits), docs)
    return results


def run(snapshot, reranker, reader, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    cases, titles = {}, {}
    for split, checksum in base.DATA.items():
        path = snapshot / f"{split}_revised.json"
        if base.digest(path) != checksum:
            raise ValueError(f"Changed snapshot split: {split}")
        documents = json.loads(path.read_bytes())
        titles[split] = {doc["title"] for doc in documents}
        cases[split] = list(base.flagged_pairs(documents))
    if titles["dev"] & titles["test"]:
        raise ValueError("Overlapping document titles")
    small = base.Reranker(reranker)
    tokenizer, large, provenance = oracle.load_reader(reader)

    def small_score(pairs):
        return small.score_pairs(pairs, task=base.TASK)

    models = dict(reranker=small_score, reader=oracle.reader_scorer(tokenizer, large))
    dev, test = cases["dev"], cases["test"]
    observations = {stage: [] for stage in ("lexical", "reranker", "reader")}
    for case in dev:
        indices = oracle.matched_window(case, 2)
        observations["lexical"].append(lexical_scores(case, indices))
        for stage, model in models.items():
            observations[stage].append(score_model(model, case, indices))
    values = {key: np.array(data) for key, data in observations.items()}
    risk_cap = len(dev) // 20
    settings = {}
    for name, (stage, budget) in POLICIES.items():
        if name == "hash_75":
            settings[name] = hash_control(dev, values["lexical"], values["reader"], settings["lexical_75"], risk_cap)
        elif budget:
            settings[name] = calibrate(values[stage], values["reader"], int(budget * len(dev)), risk_cap)
        else:
            settings[name] = dict(cutoff=float(np.sort(values[stage][:, 1])[::-1][risk_cap]))
    measure(dev[:2], settings, models)
    measured_dev = report(dev, measure(dev, settings, models))
    for name, (_, budget) in POLICIES.items():
        if measured_dev[name]["accepted_rival"] > risk_cap or measured_dev[name]["routed_pairs"] > int(budget * len(dev)):
            raise ValueError(f"Live policy exceeds development budget: {name}")
    base.write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), inputs=base.DATA,
        code_sha256=base.digest(Path(__file__)), oracle_code_sha256=base.digest(Path(oracle.__file__)),
        case_code_sha256=base.digest(Path(base.__file__)), checkpoints=dict(
            reranker=small.provenance["revision"], reader=provenance["revision"]),
        calibration=dict(pairs=len(dev), rival_cap=risk_cap, settings=settings, results=measured_dev),
        test=dict(pairs=len(test), documents=len({case["doc"] for case in test}),
                  results=report(test, measure(test, settings, models))),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("snapshot", "reranker", "reader", "output"):
        parser.add_argument(arg, type=Path)
    run(**vars(parser.parse_args()))
