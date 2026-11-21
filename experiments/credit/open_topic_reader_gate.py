"""Measure held-out allocation of the six-input topical reader."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from experiments.credit import open_topic_joint_agreement as paired
from experiments.credit import open_topic_plan_reader as study
from experiments.credit import open_topic_role_matched as previous
from experiments.credit import open_topic_three_source_plan as triple
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker

live, typed = previous.live, previous.typed
FOLDS = 5
CAP = .10
METHODS = ("stable", "weak_baseline", "small_gap", "ridge")


def features(ranks, facets, original):
    _, plans = triple.triple_plan(ranks, facets, original, triple.ALL_EDGES)
    edges = study.orientations(facets, ranks)

    def worst(edge):
        return max(ranks[0][edge[0]], ranks[1][edge[1]])

    baseline = [worst(edge) for edge in edges if set(edge) <= set(original)]
    alternatives = [worst(edge) for edge in edges if not set(edge) <= set(original)]
    base = min(baseline, default=previous.DEPTH + 1)
    rival = min(alternatives, default=previous.DEPTH + 1)
    gap = rival - base
    return dict(eligible=len(plans) > 1 and len(edges) > 1,
                inputs=[base, rival, abs(gap), gap, len(baseline), len(alternatives), len(plans)])


def collect(index, postings, samples, reader, reverse):
    indexed = list(enumerate(samples[0]))
    ordered = study.ARMS[::-1] if reverse else study.ARMS
    traces, times = [], []
    study.query(index, postings, indexed[0][1], samples[1], ordered, reader=reader, gate_features=features)
    for i, atoms in reversed(indexed) if reverse else indexed:
        shift = i % len(ordered)
        row, latency, _ = study.query(index, postings, atoms, samples[1], ordered[shift:] + ordered[:shift],
                                      reader=reader, gate_features=features)
        if row["gate_features"]["eligible"] != bool(row["arms"][study.SIX]["work"]["model_calls"]):
            raise ValueError("Reader eligibility must be known before any evidence is read")
        traces.append((i, row, latency))
    for _, row, latency in sorted(traces):
        times.append(latency)
    return [row for _, row, _ in sorted(traces)], times


def matrices(rows):
    x = np.array([row["gate_features"]["inputs"] for row in rows], dtype=np.float64)
    y = np.array([int(row["arms"][study.SIX]["known"]) - int(row["arms"][study.JOINT]["known"])
                  for row in rows], dtype=np.float64)
    eligible = np.array([row["gate_features"]["eligible"] for row in rows], dtype=bool)
    stable = [int.from_bytes(sha256(json.dumps(row["atoms"]).encode()).digest()[:6], "big")
              for row in rows]
    return x, y, eligible, np.array(stable, dtype=np.float64)


def fit(method, training, test):
    train_x, train_y, eligible, train_stable = training
    test_x, _, _, test_stable = test
    if method == "stable":
        return train_stable, test_stable
    if method == "weak_baseline":
        return train_x[:, 0], test_x[:, 0]
    if method == "small_gap":
        return -train_x[:, 2], -test_x[:, 2]

    x, y = train_x[eligible], train_y[eligible]
    center = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1
    standard = (x - center) / scale
    ridge = np.linalg.solve(standard.T @ standard + 8 * np.eye(x.shape[1]),
                            standard.T @ (y - y.mean()))
    scores = lambda features: (features - center) / scale @ ridge + y.mean()
    return scores(train_x), scores(test_x)


def select(rows, calibration, method, test_indices=None):
    if test_indices is None:
        test_indices = np.arange(len(rows))
    training, test = matrices(calibration), matrices([rows[i] for i in test_indices])
    train_scores, test_scores = fit(method, training, test)
    order = np.lexsort((training[3], train_scores))[::-1]
    valid = [i for i in order if training[2][i]]
    budget = int(CAP * len(calibration))
    threshold = ((train_scores[valid[budget - 1]], training[3][valid[budget - 1]])
                 if 0 < budget <= len(valid) else (np.inf, np.inf))

    return {int(i) for j, i in enumerate(test_indices)
            if test[2][j] and (test_scores[j], test[3][j]) >= threshold}


def routing(rows, training=None):
    decisions = {method: set() for method in METHODS}
    if training is not None:
        return {method: select(rows, training, method) for method in METHODS}
    fold = [int(sha256(row["topic_pair"].encode()).hexdigest(), 16) % FOLDS for row in rows]
    for held in range(FOLDS):
        test = [i for i, col in enumerate(fold) if col == held]
        train = [row for row, col in zip(rows, fold, strict=True) if col != held]
        for method in METHODS:
            decisions[method].update(select(rows, train, method, test))
    return decisions


def summary(rows, timings, decisions):
    reports = {}
    for method, selected in decisions.items():
        observations, ms = [], []
        gain = loss = pairs = 0
        for i, (row, measured) in enumerate(zip(rows, timings, strict=True)):
            choice = study.SIX if i in selected else study.JOINT
            baseline, current = row["arms"][study.JOINT], row["arms"][choice]
            gain += current["known"] and not baseline["known"]
            loss += baseline["known"] and not current["known"]
            pairs += current["work"]["pairs"]
            ms.append(measured[choice] + row["gate_features"]["ms"])
            observations.append(row | {"arms": {study.JOINT:baseline, method:current}})
        reports[method] = dict(
            calls=len(selected), scored_pairs=pairs, label_gains=gain, label_losses=loss,
            estimated_p95_ms=float(np.quantile(ms, .95)),
            comparisons=paired.compare(observations, ((study.JOINT, method),)),
        )
    _, delta, eligible, _ = matrices(rows)
    budget = int(CAP * len(rows))
    return dict(
        eligible=int(eligible.sum()), positive=int((delta > 0).sum()), negative=int((delta < 0).sum()),
        optimistic_gain_at_matched_budget=min(budget, int((delta > 0).sum())),
        natural_joint_p95_ms=float(np.quantile([t[study.JOINT] for t in timings], .95)),
        reader_six_p95_ms=float(np.quantile([t[study.SIX] for t in timings], .95)),
        policies=reports,
    )


def run(args):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require fresh output on a Slurm GPU")
    catalogue, panels, checkpoint, _ = previous.pinned_inputs(args)
    archive = json.loads(args.archive.read_bytes())["panels"]
    reader = Reranker(args.reranker)
    encoder = previous.mixed.QwenTorchEmbedding(args.embedding)
    traces = {}
    with TemporaryDirectory(dir=args.output.parent) as temp:
        index = live.LocalIndex(catalogue, encoder, Path(temp) / "reads")
        postings = typed.RolePostings(catalogue)
        for name, samples in panels.items():
            rows, times = collect(index, postings, samples, reader, args.reverse)
            original_rows = [{key: v for key, v in row.items() if key != "gate_features"} for row in rows]
            if paired.signature(original_rows) != archive[name]["signature"]:
                raise ValueError(f"Reader arms changed on {name}")
            traces[name] = (rows, times)
    output = {name: summary(rows, times, routing(rows, traces["plos"][0] if name == "elife" else None))
              for name, (rows, times) in traces.items()}
    write_json(args.output, dict(
        panels=output, archive_sha256=digest(args.archive), script_sha256=digest(Path(__file__)),
        checkpoint=checkpoint, reader=reader.provenance["revision"], gpu=torch.cuda.get_device_name(),
        reverse=args.reverse, job=os.environ["SLURM_JOB_ID"],
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plos", "fixture", "elife", "elife_manifest", "embedding", "reference", "live_reference",
                 "archive", "reranker", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--reverse", action="store_true")
    run(parser.parse_args())
