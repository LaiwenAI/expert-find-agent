"""Learn shared four-sentence windows from missing-evidence counterfactuals."""

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.statistics import clustered_paired_interval
from experiments.credit.calibrated_topic_evidence import margins
from experiments.docred import binding_marker_training as binding
from experiments.docred import learned_conjunctive_windows as previous
from experiments.docred import person_bound_evidence as names
from experiments.docred import work_owner_contrast as source
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker


ARMS = ("global", "scoped", "learned_pair_union", "random_negative", "missing_reference")
WIDTH = 4
LOCAL_PAIRS = np.array(list(combinations(range(WIDTH), 2)))


def add_annotations(rows, path):
    documents = {item["title"]: item for item in json.loads(path.read_bytes())}
    selected = []
    for row in rows:
        document = documents[row["doc"]]
        if len(row["sentences"]) < WIDTH:
            continue
        canonical = [names.canonical(entity) for entity in document["vertexSet"]]
        annotations = defaultdict(list)
        for label in document["labels"]:
            key = canonical[label["h"]], canonical[label["t"]], label["r"]
            annotations[key].append(frozenset(label.get("evidence", ())))
        refs = []
        for work, person in row["names"]:
            candidates = annotations[(work, person, row["role"])]
            refs.append(candidates[0] if len(candidates) == 1 else frozenset())
        selected.append(dict(row, refs=refs))
    return selected


class CandidateWindows:
    def __init__(self, case):
        self.n = len(case["sentences"])
        left, right, self.pair_features = previous.pair_features(case)
        self.lookup = np.full((self.n, self.n), -1, dtype=np.int32)
        self.lookup[left, right] = np.arange(len(left))
        self.choices = np.array(list(combinations(range(self.n), WIDTH)), dtype=np.int32)

    def features(self, windows):
        rows = np.asarray(windows, dtype=np.int32)
        ids = self.lookup[rows[:, LOCAL_PAIRS[:, 0]], rows[:, LOCAL_PAIRS[:, 1]]]
        return self.pair_features[ids].max(axis=1).transpose(0, 2, 1)


def training_examples(cases):
    build = {"random_negative": [[], []], "missing_reference": [[], []]}
    counts = Counter()
    for case in cases:
        model = CandidateWindows(case)
        rng = np.random.default_rng(int(case["id"][:16], 16))
        for i, ref in enumerate(case["refs"]):
            if not 1 <= len(ref) <= WIDTH:
                continue
            present = np.isin(model.choices, sorted(ref)).sum(axis=1)
            contains = model.choices[present == len(ref)]
            without = model.choices[present < len(ref)]
            if not len(contains) or not len(without):
                continue
            positive = contains[rng.permutation(len(contains))[:4]]
            random = without[rng.permutation(len(without))[:4]]
            replacements = []
            for window in positive:
                old = set(window.tolist())
                removed = min(ref, key=lambda n: rng.random())
                added = rng.choice(sorted(set(range(model.n)) - old))
                replacements.append(sorted((old - {removed}) | {int(added)}))
            count = min(len(positive), len(random))
            pos = model.features(positive[:count])[:, i]
            for arm, negative in (("random_negative", random), ("missing_reference", replacements)):
                neg = model.features(negative[:count])[:, i]
                build[arm][0].append(pos)
                build[arm][1].append(neg)
            counts["claims"] += 1
            counts["pairs_per_arm"] += count
    return build, dict(counts)


def train(positive, negative):
    """Fixed tiny CPU linear model, fitted under the same objective in both arms."""
    positive = np.concatenate(positive)
    negative = np.concatenate(negative)
    fit = np.concatenate((positive, negative))
    mean, scale = fit.mean(0), fit.std(0).clip(.05)
    contrasts = (positive - negative) / scale
    dim = contrasts.shape[-1]
    weights = np.linalg.solve(contrasts.T @ contrasts + np.eye(dim) * 25, contrasts.sum(axis=0))
    center = np.median((fit - mean) @ (weights / scale))
    return dict(weights=weights / scale, intercept=float(-mean @ (weights / scale) - center))


def select(case, arm, trained, model, pair_weights):
    if arm == "global":
        remaining = source.window(case).splitlines()
        chosen = []
        for i, sentence in enumerate(case["sentences"]):
            if sentence in remaining:
                chosen.append(i)
                remaining.remove(sentence)
        return chosen
    if arm == "scoped":
        group, scoped = previous.lexical_features(case)
        return previous.base.choose_windows(group, scoped, "claim_coverage")
    if arm == "learned_pair_union":
        pairs = previous.candidate_windows(case, pair_weights, "learned_sum")
        return sorted({int(sentence) for pair in pairs for sentence in pair})
    features = model.features(model.choices)
    policy = trained[arm]
    scores = features @ policy["weights"] + policy["intercept"]
    # Saturation prevents a single easy claim from taking all four reads.
    utility = (1 / (1 + np.exp(np.clip(-scores, -40, 40)))).sum(axis=1)
    return model.choices[int(utility.argmax())].tolist()


def measure(cases, trained, reader, offset, pair_weights):
    records = []
    for i, case in enumerate([cases[0], *cases]):
        shift = (i + offset) % len(ARMS)
        rows = {}
        start = perf_counter()
        model = CandidateWindows(case)
        preparation = (perf_counter() - start) * 1000
        for arm in ARMS[shift:] + ARMS[:shift]:
            start = perf_counter()
            picked = select(case, arm, trained, model, pair_weights)
            if len(set(picked)) != WIDTH:
                raise AssertionError("Evidence budget differs")
            context = "\n".join(case["sentences"][i] for i in picked)
            selection_ms = (perf_counter() - start) * 1000 + (preparation if arm in trained else 0)
            prompts = [(f"Did {person} have the role of {names.ROLES[case['role']]} on {work}?", context)
                       for work, person in case["names"]]
            values, clipped = margins(reader, prompts, binding.TASK)
            scores = list(map(float, values))
            json.dumps({"sentences": picked, "scores": scores})
            ms = (perf_counter() - start) * 1000
            rows[arm] = dict(
                picked=picked, scores=scores, clipped=clipped,
                ms=ms + (preparation if arm in trained else 0),
                selection_ms=selection_ms, words=len(context.split()),
                complete=[bool(ref) and ref <= set(picked) for ref in case["refs"]],
            )
        if i:
            records.append((case, rows))
    return records


def summarize(rows, arm, threshold):
    totals = Counter()
    covered = []
    for case, observed in rows:
        measurement = observed[arm]
        successful = 0
        for j, (value, complete) in enumerate(zip(measurement["scores"], measurement["complete"], strict=True)):
            known = case["labeled"][j]
            totals["known"] += int(known)
            totals["unjudged"] += int(not known)
            totals["known_refs"] += int(known and bool(case["refs"][j]))
            totals["ref_complete"] += int(known and complete)
            totals["accepted_known"] += int(value > threshold and known)
            totals["accepted_unjudged"] += int(value > threshold and not known)
            totals["accepted_complete"] += int(value > threshold and known and complete)
            successful += int(known and complete)
        covered.append(successful)
    return dict(totals), np.array(covered)


def evaluate(sets, reader, offset, release):
    fit, cal, held = sets
    examples, training_counts = training_examples(fit)
    trained = {arm: train(*pairs) for arm, pairs in examples.items()}
    pair_weights, pair_counts = previous.fit(release, {"dev": cal, "test": held})
    calibration = measure(cal, trained, reader, offset, pair_weights)
    outcomes = measure(held, trained, reader, offset, pair_weights)
    result = {}
    for arm in ARMS:
        rivals = sorted((score for case, row in calibration
                         for score, known in zip(row[arm]["scores"], case["labeled"], strict=True) if not known),
                        reverse=True)
        threshold = rivals[len(rivals) // 20]
        calibrated, _ = summarize(calibration, arm, threshold)
        passed, coverage = summarize(outcomes, arm, threshold)
        reference = summarize(outcomes, "global", threshold)[1]
        result[arm] = dict(
            threshold=threshold, calibration=calibrated, held=passed,
            complete_delta=clustered_paired_interval(coverage - reference, [r["doc"] for r, _ in outcomes]),
            p95_ms=float(np.quantile([obs[arm]["ms"] for _, obs in outcomes], .95)),
            selector_p95_ms=float(np.quantile([obs[arm]["selection_ms"] for _, obs in outcomes], .95)),
            mean_window_words=float(np.mean([obs[arm]["words"] for _, obs in outcomes])),
            clipped=sum(obs[arm]["clipped"] for _, obs in calibration + outcomes),
            evidence=[dict(id=row["id"], sentences=obs[arm]["picked"],
                           complete=obs[arm]["complete"], accepted=[v > threshold for v in obs[arm]["scores"]])
                      for row, obs in outcomes],
        )
    return result, training_counts | {"pair_baseline": pair_counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("original", "expanded", "reranker", "output"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require Slurm CUDA and a fresh output path")
    sets = [add_annotations(rows, args.original / f"{split}_revised.json")
            for rows, split in zip(binding.panel(args.original, args.expanded), ("train", "dev", "test"), strict=True)]
    reader = Reranker(args.reranker)
    results, trained = evaluate(sets, reader, args.offset, args.original)
    metadata = {name: len(rows) for name, rows in zip(("fit", "calibration", "held"), sets, strict=True)}
    write_json(args.output, dict(
        cases=metadata, width=WIDTH, training=trained, offset=args.offset, results=results,
        gpu=torch.cuda.get_device_name(), script_sha256=digest(Path(__file__)),
        reranker_revision=reader.provenance["revision"],
        input_sha256={"train": binding.TRAIN_SHA256, "original": names.DATA, "expanded": source.UPDATED},
    ))


if __name__ == "__main__":
    main()
