"""Calibrate high-recall screening before live MERIT reviewer verification."""

import argparse
from collections import defaultdict
import math
import os
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from expert_search.statistics import clustered_paired_interval
from experiments.decision.merit_evidence_dispatch import Decision, read_cases
from experiments.decision.merit_lightweight_decision import (
    Representations, balanced, fit, logit, split_cases, threshold,
)
from experiments.followir.core17_reranking import digest, write_json

ALPHA = 0.10
SCREENS = ("lexical", "dense_max", "dense_head", "agreement")
ARMS = ("full", *SCREENS)


def scores(index, case, models, arm):
    if arm == "lexical":
        return logit(index.lexical(case), models["lexical"])
    dense = index.dense(case)
    if arm == "dense_max":
        return dense[0]
    head = logit(dense + index.lexical(case), models["dense_head"])
    if arm == "dense_head":
        return head
    center, scale = models["agreement"]
    # Skip only when both the top source and the learned aggregate score are low.
    return float(np.max((np.array([dense[0], head]) - center) / scale))


def calibration_cutoff(rows):
    """Lower score quantile on the least-supported positive in each component."""
    by_component = defaultdict(list)
    for component, positive, score in rows:
        if positive:
            by_component[component].append(score)
    minima = sorted(min(items) for items in by_component.values())
    rank = math.floor((len(minima) + 1) * ALPHA)
    return dict(cutoff=minima[rank-1] if rank else None, positive_components=len(minima), rank=rank)


def passes(score, gate):
    return gate["cutoff"] is None or score >= gate["cutoff"]


def evaluate(arm, case, index, reader, models, cutoffs, reader_cutoff):
    start = perf_counter()
    keep = arm == "full" or passes(scores(index, case, models, arm), cutoffs[arm])
    if keep:
        margin, tokens = reader.judge(case.target, case.papers)
        prediction = int(margin >= reader_cutoff)
    else:
        prediction, tokens = 0, 0
    return dict(prediction=prediction, keep=int(keep), reader_tokens=tokens, ms=(perf_counter()-start)*1000)


def summarize(rows):
    label = np.array([case.label for _, case, _ in rows], dtype=bool)
    groups = [group for group, _, _ in rows]
    reference = np.array([row["full"]["prediction"] for _, _, row in rows], dtype=bool)
    positive_groups = {g for g, positive in zip(groups, label, strict=True) if positive}
    result = {}
    for arm in ARMS:
        values = [row[arm] for _, _, row in rows]
        keep = np.array([v["keep"] for v in values], dtype=bool)
        prediction = np.array([v["prediction"] for v in values], dtype=bool)
        correct = prediction == label
        tp = int((prediction & label).sum())
        collisions = {g for g, miss in zip(groups, label & ~keep, strict=True) if miss}
        result[arm] = dict(
            passed=int(keep.sum()), skipped=int((~keep).sum()),
            positives_passed=int((keep & label).sum()), positives_total=int(label.sum()),
            positive_components_with_no_skip=len(positive_groups - collisions),
            positive_components=len(positive_groups), full_reader_true_positives_skipped=int((reference & label & ~keep).sum()),
            full_reader_false_positives_skipped=int((reference & ~label & ~keep).sum()),
            accuracy=float(correct.mean()), balanced_accuracy=balanced(prediction,label),
            positive_f1=2*tp/max(1,int(prediction.sum())+int(label.sum())),
            true_positives=tp, final_recall=float(tp/label.sum()), reader_tokens=mean(v["reader_tokens"] for v in values),
            mean_ms=mean(v["ms"] for v in values), p50_p95_ms=np.quantile([v["ms"] for v in values],[.5,.95]).tolist(),
            accuracy_delta_vs_full=clustered_paired_interval(
                correct.astype(int)-(reference == label).astype(int), groups,
            ) | {"sampling_unit":"connected paper/profile component"},
        )
    return result


def run(args):
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a fresh output and Slurm CUDA")
    cases = read_cases(args.data)
    splits = split_cases(cases)
    index = Representations(cases, args.embedding)
    reader = Decision(args.reader)
    train = [case for _, case in splits["fit"]]
    labels = np.array([case.label for case in train])
    lexical = [index.lexical(case) for case in train]
    dense = [index.dense(case) for case in train]
    joint = [a+b for a,b in zip(dense,lexical,strict=True)]
    models = dict(lexical=fit(np.array(lexical),labels), dense_head=fit(np.array(joint),labels))
    both = np.array([[d[0],logit(j,models["dense_head"])] for d,j in zip(dense,joint,strict=True)])
    models["agreement"] = both.mean(axis=0), np.maximum(both.std(axis=0),1e-8)
    calibration = []
    for group,case in splits["calibration"]:
        read, _ = reader.judge(case.target,case.papers)
        row = {arm:scores(index,case,models,arm) for arm in SCREENS}
        calibration.append((group,case.label,read,row))
    cutoffs = {arm:calibration_cutoff([(g,y,row[arm]) for g,y,_,row in calibration]) for arm in SCREENS}
    reader_cutoff = threshold([v for _,_,v,_ in calibration],[y for _,y,_,_ in calibration])
    print("Calibration thresholds frozen", cutoffs, flush=True)
    test = []
    for i,(group,case) in enumerate(splits["test"]):
        offset = i%len(ARMS)
        test.append((group,case,{arm:evaluate(arm,case,index,reader,models,cutoffs,reader_cutoff)
                                 for arm in ARMS[offset:]+ARMS[:offset]}))
        if (i+1)%30==0:
            print(f"Completed test: {i+1}",flush=True)
    write_json(args.output, dict(job=os.environ["SLURM_JOB_ID"],gpu=torch.cuda.get_device_name(),
        data_sha256=digest(args.data),script_sha256=digest(Path(__file__)), alpha=ALPHA, cutoffs=cutoffs,
        reader_cutoff=reader_cutoff,embedding=index.provenance["revision"],reader=reader.provenance["revision"],
        split_counts={name:len(rows) for name,rows in splits.items()}, arms=summarize(test)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data","embedding","reader","output"):
        parser.add_argument(name,type=Path)
    run(parser.parse_args())
