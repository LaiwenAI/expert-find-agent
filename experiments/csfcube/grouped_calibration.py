"""Replay grouped, out-of-fold calibration on expert-adjudicated CSFCube pools."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from expert_search.statistics import clustered_paired_interval
from experiments.credit.source_confirmation import accepts
from experiments.followir.core17_reranking import digest, write_json


FACETS = ("background", "method", "result")
VIEWS = ("present", "withheld")
FOLDS = 5
BUDGETS = (.05, .10, .20)
CONTROLS = ("cosine", "learned", "cosine_two", "reader_two", "reader_six")
AGREEMENT = "cosine_two_agreement"


def load(data, run):
    questions = []
    for facet in FACETS:
        path = data / f"test-pid2anns-csfcube-{facet}.json"
        if digest(path) != run["source_hashes"][path.name]:
            raise ValueError(f"Changed judged pool: {path.name}")
        for qid, row in sorted(json.loads(path.read_text()).items()):
            grades = dict(zip(map(int, row["cands"]), row["relevance_adju"], strict=True))
            grades.pop(int(qid), None)
            questions.append((qid, grades))
    if len(questions) != 50:
        raise ValueError("Expected 50 query-facet pairs")
    return questions


def partition(questions):
    ids = sorted({qid for qid, _ in questions}, key=lambda qid: sha256(qid.encode()).digest())
    fold = {qid: i % FOLDS for i, qid in enumerate(ids)}
    return [[i for i, (qid, _) in enumerate(questions) if fold[qid] == k] for k in range(FOLDS)]


def observations(run, questions):
    output = {}
    dense = run["test"]["cosine"]
    for method in (*CONTROLS, AGREEMENT):
        source = "cosine_two" if method == AGREEMENT else method
        rows = []
        for reference, row, (qid, grades) in zip(dense, run["test"][source], questions, strict=True):
            targets = {doc for doc, grade in grades.items() if grade >= 2}
            if reference["id"] != row["id"] or set(row["targets"]) != targets:
                raise ValueError(f"Inconsistent observation for {qid}")
            views = {}
            for view in VIEWS:
                arm = row["views"][view]
                scored = arm["scored"]
                if arm["pool"] != reference["views"][view]["pool"]:
                    raise ValueError(f"Different dense candidates for {qid}")
                if view == "withheld" and targets.intersection(arm["pool"]):
                    raise ValueError(f"Strong candidate leaked into withheld pool for {qid}")
                grade = grades[scored[0]]
                if method == AGREEMENT and scored[0] != reference["views"][view]["scored"][0]:
                    scored = (None, 0., 0.)
                views[view] = dict(scored=scored, grade=grade, ms=arm["ms"])
            rows.append(views)
        output[method] = rows
    return output


def fit(rows, budget):
    grid = [row[view]["scored"] for row in rows for view in VIEWS
            if row[view]["scored"][0] is not None]
    scores = sorted({x[1] for x in grid})
    margins = sorted({x[2] for x in grid})
    cap = int(budget * len(rows))
    best, selected = (0, 0, 0, 0), None
    for score in scores:
        for margin in margins:
            policy = dict(score=score, margin=margin)
            absent = sum(accepts(row["withheld"]["scored"], policy) for row in rows)
            if absent > cap:
                continue
            grades = Counter(row["present"]["grade"] for row in rows
                             if accepts(row["present"]["scored"], policy))
            low = grades[0] + grades[1]
            objective = grades[2] + grades[3], -absent, -low, -grades[0]
            if low <= cap and objective > best:
                best, selected = objective, policy
    return selected


def cross_validate(rows, folds, budget):
    predictions, thresholds = {}, []
    indices = set(range(len(rows)))
    for test in folds:
        calibration = sorted(indices.difference(test))
        policy = fit([rows[i] for i in calibration], budget)
        thresholds.append(dict(calibration_size=len(calibration), test_indices=test, policy=policy))
        for i in test:
            predictions[i] = {
                view: arm["grade"] if accepts(arm["scored"], policy) else None
                for view, arm in rows[i].items()
            }
    return [predictions[i] for i in range(len(rows))], thresholds


def summarize(predictions, rows):
    report = {}
    for view in VIEWS:
        counts = Counter(row[view] for row in predictions)
        report[view] = dict(relevance_grades=[counts[i] for i in range(4)], abstained=counts[None],
                            p95_ms=float(np.quantile([row[view]["ms"] for row in rows], .95)))
    return report


def run(data, replay, output):
    result = json.loads(replay.read_text())
    questions = load(data, result)
    groups = [qid for qid, _ in questions]
    folds = partition(questions)
    arms = observations(result, questions)
    report = {}
    for budget in BUDGETS:
        evaluated = {}
        for method, rows in arms.items():
            predicted, thresholds = cross_validate(rows, folds, budget)
            evaluated[method] = dict(predictions=predicted, thresholds=thresholds,
                                     summary=summarize(predicted, rows))
        baseline = evaluated["reader_six"]["predictions"]
        for row in evaluated.values():
            differences = [
                int((left["present"] or 0) >= 2) - int((right["present"] or 0) >= 2)
                for left, right in zip(row["predictions"], baseline, strict=True)
            ]
            row["strong_delta_vs_six"] = clustered_paired_interval(differences, groups)
        report[f"{budget:.2f}"] = evaluated
    write_json(output, dict(replay_sha256=digest(replay), folds=folds, groups=len(set(groups)), report=report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "replay", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    run(args.data, args.replay, args.output)
