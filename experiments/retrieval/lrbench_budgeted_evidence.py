"""Cross-fit selective source scoring after compact RATE profile comparison."""

import argparse
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import importlib.util
import json
from pathlib import Path

import numpy as np

from expert_search.statistics import clustered_paired_interval
from experiments.retrieval.lrbench_frozen_baselines import ASSET_HASHES, paired_stats, read_split


SIDES = ("positive_score", "negative_score")
ALIGNMENT = ("anchor_id", "positive_id", "negative_id", "type",
             "positive_ref_score", "negative_ref_score")
FOLDS = 5
BUDGETS = (.10, .25, .50)
WEIGHTS = (0., .25, .5, .75, 1.)
L2 = 8.


@dataclass(frozen=True)
class Panel:
    reference: list
    compact: np.ndarray
    sources: np.ndarray
    features: np.ndarray
    reads: np.ndarray


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def fold(anchor):
    seed = sha256(f"lrbench-cold-anchor-v1:{anchor}".encode()).digest()
    return int.from_bytes(seed[:8], "big") % FOLDS


def partitions(reference):
    folds = [fold(row["anchor_id"]) for row in reference]
    for held in range(FOLDS):
        test = [i for i, f in enumerate(folds) if f == held]
        blocked = {reference[i][key] for i in test for key in ("positive_id", "negative_id")}
        train = [i for i, row in enumerate(reference) if folds[i] != held
                 and row["positive_id"] not in blocked and row["negative_id"] not in blocked]
        if not train or not test:
            raise ValueError("An identity-separated fold is empty")
        yield np.array(train), np.array(test)


def read_scores(path, reference):
    rows = json.loads(path.read_bytes())
    for actual, expected in zip(rows, reference, strict=True):
        if any(actual[key] != expected[key] for key in ALIGNMENT):
            raise ValueError(f"Misaligned prediction: {path.name}")
    scores = np.array([[row[key] for key in SIDES] for row in rows], dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError(f"Nonfinite prediction: {path.name}")
    return scores


def load_panel(release, predictions, split):
    cases, reference = read_split(release, split)
    compact = read_scores(predictions / f"rate06-keywords_{split}.json", reference)
    sources = read_scores(predictions / f"rate06-raw-top3_{split}.json", reference)
    sizes = np.array([[len(history) for _, history in pair] for pair in cases])
    if np.any(sizes == 0):
        raise ValueError("Empty history")
    reads = sizes.sum(axis=1)
    lengths = np.log1p(sizes)
    overlap = []
    for (_, first), (_, second) in cases:
        left, right = set(first), set(second)
        overlap.append(len(left & right) / len(left | right))
    features = np.column_stack((
        np.abs(compact[:, 0] - compact[:, 1]),
        compact.mean(axis=1),
        lengths.min(axis=1),
        np.abs(lengths[:, 0] - lengths[:, 1]),
        overlap,
    ))
    return Panel(reference, compact, sources, features, reads)


def margins(scores):
    return scores[:, 0] - scores[:, 1]


def fit_fusion(panel, train, routed=None):
    # Every candidate uses the same transformation. Positive/negative ordering
    # changes the sign of a margin, but cannot change the routing features.
    scale = np.array([np.median(np.abs(margins(scores)[train]))
                      for scores in (panel.compact, panel.sources)])
    if min(scale) <= 0:
        raise ValueError("A training score scale is zero")
    compact = panel.compact / scale[0]
    sources = panel.sources / scale[1]
    fit_rows = train if routed is None else routed
    weight = max(WEIGHTS, key=lambda alpha: (
        np.count_nonzero(margins((1 - alpha) * compact[fit_rows] + alpha * sources[fit_rows]) > 0),
        -alpha,
    ))
    return (1 - weight) * compact + weight * sources, weight


def fit_utility(features, gains, train):
    mean = features[train].mean(axis=0)
    scale = features[train].std(axis=0)
    standardized = (features - mean) / np.where(scale > 0, scale, 1.0)
    design = standardized[train]
    coefficients = np.linalg.solve(design.T @ design + L2 * np.eye(design.shape[1]),
                                   design.T @ gains[train])
    return standardized @ coefficients


def threshold(scores, train, test, budget):
    # The threshold comes from the training folds. The realized held-out
    # acquisition rate can differ. Strict comparison handles tied scores
    # conservatively without using pair ordering to break ties.
    cutoff = np.quantile(scores[train], 1 - budget, method="higher")
    return scores[test] > cutoff


def assign(panel):
    n = len(panel.reference)
    policies = ("margin_fuse", "margin_raw", "margin_conditional_fuse", "utility_raw")
    names = ["always_fuse", *(f"{policy}_{int(100 * budget)}"
                              for budget in BUDGETS for policy in policies)]
    acquired = {name: np.zeros(n, dtype=bool) for name in names}
    results = {name: panel.compact.copy() for name in names}
    base_success = margins(panel.compact) > 0
    raw_gain = (margins(panel.sources) > 0).astype(float) - base_success
    folds = []

    def save(name, rows, candidate):
        acquired[name][rows] = True
        results[name][rows] = candidate[rows]

    for train, test in partitions(panel.reference):
        candidate, weight = fit_fusion(panel, train)
        if weight:
            save("always_fuse", test, candidate)
        utility = fit_utility(panel.features, raw_gain, train)
        margins_only = -panel.features[:, 0]
        conditional = {}
        for budget in BUDGETS:
            suffix = int(100 * budget)
            uncertain = test[threshold(margins_only, train, test, budget)]
            gain_routed = test[threshold(utility, train, test, budget)]
            save(f"utility_raw_{suffix}", gain_routed, panel.sources)
            save(f"margin_raw_{suffix}", uncertain, panel.sources)
            if weight:
                save(f"margin_fuse_{suffix}", uncertain, candidate)
            routed_train = train[threshold(margins_only, train, train, budget)]
            selective, selective_weight = fit_fusion(panel, train, routed_train)
            if selective_weight:
                save(f"margin_conditional_fuse_{suffix}", uncertain, selective)
            conditional[suffix] = selective_weight
        folds.append(dict(train=len(train), test=len(test), evidence_weight=weight,
                          conditional_evidence_weight=conditional))
    return results, acquired, folds


def evaluate(evaluator, split, panel):
    routed, acquired, folds = assign(panel)
    scores = {"compact": panel.compact, "raw_top_three": panel.sources, **routed}
    predictions = {"rate": panel.reference}
    for name, estimates in scores.items():
        predictions[name] = [dict(row, positive_score=float(left), negative_score=float(right))
                             for row, (left, right) in zip(panel.reference, estimates, strict=True)]
    results = paired_stats(evaluator, split, predictions)
    baseline = margins(panel.compact) > 0
    groups = [row["anchor_id"] for row in panel.reference]
    raw_success = margins(panel.sources) > 0
    raw_repair_costs = np.sort(panel.reads[raw_success & ~baseline]).cumsum()
    for name, estimates in scores.items():
        changed = (margins(estimates) > 0).astype(int) - baseline
        mask = acquired.get(name, np.full(len(groups), name != "compact"))
        counts = Counter(gain=int((changed > 0).sum()), loss=int((changed < 0).sum()),
                         acquired=int(mask.sum()), source_dot_products=int(panel.reads[mask].sum()))
        results[name].update(
            **counts,
            delta_vs_compact=clustered_paired_interval(changed, groups),
            raw_oracle_repairs_at_this_read_cost=int(np.searchsorted(
                raw_repair_costs, counts["source_dot_products"], side="right",
            )),
        )
    fused_success = margins(routed["always_fuse"]) > 0
    return dict(
        comparisons=len(groups), unique_anchors=len(set(groups)), folds=folds, results=results,
        oracle_diagnostics=dict(
            raw_can_repair=int((raw_success & ~baseline).sum()),
            raw_can_break=int((~raw_success & baseline).sum()),
            either_compact_or_raw_correct=int((raw_success | baseline).sum()),
            fused_can_repair=int((fused_success & ~baseline).sum()),
            fused_can_break=int((~fused_success & baseline).sum()),
        ),
    )


def run(release, predictions, output):
    if output.exists():
        raise FileExistsError(output)
    for asset, expected in ASSET_HASHES.items():
        if digest(release / asset) != expected:
            raise ValueError(f"Changed release file: {asset}")
    spec = importlib.util.spec_from_file_location("rate_evaluator", release / "rate-evaluation_script.py")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    score_hashes = {name: digest(predictions / name)
                    for split in ("pc", "rc") for name in
                    (f"rate06-keywords_{split}.json", f"rate06-raw-top3_{split}.json")}
    results = {split: evaluate(evaluator, split, load_panel(release, predictions, split))
               for split in ("pc", "rc")}
    with output.open("x") as handle:
        json.dump(dict(asset_hashes=ASSET_HASHES, prediction_hashes=score_hashes,
                       script_sha256=digest(Path(__file__)), splits=results), handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("release", "predictions", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.release, args.predictions, args.output)
