"""Test compact within-profile evidence checks against person-level MERIT labels."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.statistics import clustered_paired_interval
from experiments.decision import merit_lightweight_decision as base
from experiments.decision.merit_evidence_dispatch import Decision, read_cases, terms
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker


ARMS = ("dense_max", "dense_head", "cross_max", "cross_head", "complement_head",
        "reader_two", "reader_full")
TASK = ("Determine whether the author's publication shows technical methods or skills "
        "directly relevant to reviewing the target paper.")


class ProfileCheck:
    def __init__(self, representation, verifier, reader):
        self.representation = representation
        self.verifier = verifier
        self.reader = reader

    def similarity(self, case):
        rep = self.representation
        query = rep.embedding.encode([f"Instruct: {base.TASK}\nQuery: {case.query}"])[0]
        return np.stack([rep.vectors[p] for p in case.papers]) @ query

    def two_sources(self, case, values, *, complement):
        indices = sorted(range(len(values)), key=lambda i: (-values[i], i))
        first, second = indices[:2]
        query = terms(case.query)
        overlaps = [query & self.representation.terms[doc] for doc in case.papers]
        if complement:
            remainder = query - overlaps[first]
            second = max(indices[1:], key=lambda i: (len(overlaps[i] & remainder), values[i], -i))
        pairs = [(case.query, case.papers[i]) for i in (first, second)]
        scores, clipped = self.verifier.score_pairs(pairs, task=TASK)
        if not np.isfinite(scores).all():
            raise ValueError("Unexpected nonfinite cross-encoder output")
        common = max(1, len(query))
        features = [
            max(scores), min(scores), abs(scores[0] - scores[1]),
            len(overlaps[first] | overlaps[second]) / common,
            len(overlaps[second] - overlaps[first]) / common,
        ]
        return features, clipped

    def collect(self, case):
        source_scores = self.similarity(case)
        features = base.summarize_similarities(source_scores) + self.representation.lexical(case)
        cross, clipped = self.two_sources(case, source_scores, complement=False)
        return features, features + cross, clipped

    def score(self, case, arm, models):
        if arm in ("reader_two", "reader_full"):
            history = case.papers if arm == "reader_full" else case.papers[:2]
            return self.reader.judge(case.target, history)[0], 0
        similarity = self.similarity(case)
        if arm == "dense_max":
            return float(max(similarity)), 0
        if arm == "dense_head":
            features = base.summarize_similarities(similarity) + self.representation.lexical(case)
            return base.logit(features, models["dense_head"]), 0
        cross, clipped = self.two_sources(case, similarity, complement=arm == "complement_head")
        if arm == "cross_max":
            return cross[0], clipped
        features = base.summarize_similarities(similarity) + self.representation.lexical(case) + cross
        # Use the same fitted head before and after the evidence-order intervention.
        return base.logit(features, models["cross_head"]), clipped


def cutoffs(values, labels):
    positive = sorted(score for score, held in zip(values, labels, strict=True) if held)
    if not positive:
        raise ValueError("Missing positive calibration cases")
    return {"balanced": base.threshold(values, labels),
            "retain_90pct": float(positive[len(positive) // 10])}


def outcomes(labels, scores, limit):
    predicted = scores >= limit
    tp = int((predicted & labels).sum())
    fp = int((predicted & ~labels).sum())
    return dict(
        accepted=int(predicted.sum()), true_positive=tp, false_positive=fp,
        accuracy=float(np.mean(predicted == labels)), balanced_accuracy=base.balanced(predicted, labels),
        positive_recall=tp/int(labels.sum()), positive_f1=2*tp/(int(predicted.sum())+int(labels.sum())),
    )


def run(args):
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Use a fresh output and a Slurm GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    cases = read_cases(args.data)
    splits = base.split_cases(cases)
    anchor = json.loads(args.reference.read_bytes())
    if anchor["data_sha256"] != digest(args.data):
        raise ValueError("Changed assessor release")
    counts = {key: dict(pairs=len(rows), positives=sum(case.label for _,case in rows),
                        components=len({group for group,_ in rows})) for key,rows in splits.items()}
    if counts != anchor["splits"]:
        raise ValueError("Changed component split")
    before = perf_counter()
    representation = base.Representations(cases, args.embedding)
    offline_seconds = perf_counter() - before
    verifier = Reranker(args.reranker)
    reader = Decision(args.reader)
    if representation.provenance["revision"] != anchor["embeddings"] or reader.provenance["revision"] != anchor["reader"]:
        raise ValueError("Changed historical baseline checkpoints")
    check = ProfileCheck(representation, verifier, reader)
    training, labels = [], []
    for _, case in splits["fit"]:
        training.append(check.collect(case))
        labels.append(case.label)
    dense, cross, clips = zip(*training)
    models = {"dense_head": base.fit(np.asarray(dense), np.asarray(labels)),
              "cross_head": base.fit(np.asarray(cross), np.asarray(labels))}
    limits, reports = {}, {}
    calibration_labels = np.array([case.label for _,case in splits["calibration"]], dtype=bool)
    latency = {arm: [] for arm in ARMS}
    scores = {arm: [] for arm in ARMS}
    calibration_clips = Counter()
    for arm in ARMS:
        calibration = []
        for _,case in splits["calibration"]:
            score, clip = check.score(case, arm, models)
            calibration_clips[arm] += clip
            calibration.append(score)
        limits[arm] = cutoffs(calibration, calibration_labels)
        reports[arm] = dict(calibration={
            key: outcomes(calibration_labels, np.asarray(calibration), cutoff)
            for key, cutoff in limits[arm].items()
        })
    print("Fit 356 examples and froze both operating points for all arms", flush=True)
    holdout = splits["test"]
    test_labels = np.asarray([case.label for _,case in holdout], dtype=bool)
    groups = [group for group,_ in holdout]
    test_clips = Counter()
    order = ARMS[::-1] if args.order_offset else ARMS
    for i,(_,case) in enumerate(holdout):
        offset = i % len(order)
        for arm in order[offset:] + order[:offset]:
            started = perf_counter()
            score, clip = check.score(case, arm, models)
            ms = (perf_counter() - started) * 1000
            latency[arm].append(ms)
            scores[arm].append(score)
            test_clips[arm] += clip
        if (i+1)%30 == 0:
            print(f"Scored {i+1} held paper–profile pairs", flush=True)
    baseline = np.asarray(scores["dense_head"]) >= limits["dense_head"]["balanced"]
    for arm, report in reports.items():
        values = np.asarray(scores[arm])
        report["test"] = {key:outcomes(test_labels, values, cutoff) for key,cutoff in limits[arm].items()}
        comparable = values >= limits[arm]["balanced"]
        report["accuracy_delta_vs_dense_head_at_balanced_cutoffs"] = clustered_paired_interval(
            (comparable == test_labels).astype(int) - (baseline == test_labels).astype(int), groups,
        )
        report["p95_ms"] = float(np.quantile(latency[arm], .95))
    write_json(args.output, dict(
        splits=counts, results=reports, cutoffs=limits, gpu=torch.cuda.get_device_name(),
        clips=dict(fit=sum(clips),calibration=calibration_clips,test=test_clips),
        embeddings=representation.provenance["revision"], compact_verifier=verifier.provenance["revision"],
        reader=reader.provenance["revision"], offline_embedding_seconds=offline_seconds,
        data_sha256=digest(args.data), script_sha256=digest(Path(__file__)), order_offset=args.order_offset,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "embedding", "reranker", "reader", "reference", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--order-offset", type=int, default=0, choices=(0, 1))
    run(parser.parse_args())
