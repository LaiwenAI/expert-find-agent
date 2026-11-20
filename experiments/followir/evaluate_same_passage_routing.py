"""Evaluate fixed-cost routing policies using frozen, matched verifier scores."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from expert_search.statistics import paired_interval


KINDS = ("stable_positive", "invalidated", "stable_negative")
NEGATIVE_CAP = 0.10
WEIGHTS = {"stable_positive": 1, "invalidated": 2, "stable_negative": 1}


def cutoff(development, score_name):
    limits = []
    for rows in development.values():
        for kind in KINDS[1:]:
            margins = sorted((row[score_name] for row in rows if row["label"] == kind), reverse=True)
            if not margins:
                raise ValueError("Missing negative calibration stratum")
            limits.append(margins[int(NEGATIVE_CAP * len(margins))])
    return max(limits)


def margin_features(rows, cheap_cutoff):
    return np.asarray([
        [row["cheap"] - cheap_cutoff, abs(row["cheap"] - cheap_cutoff),
         row["cheap"] > cheap_cutoff, row["query_coverage"], row["criterion_coverage"],
         row["length"] / 1800, (row["cheap"] - cheap_cutoff) * row["criterion_coverage"]]
        for row in rows
    ], dtype=float)


def decisions(rows, thresholds):
    cheap, expensive = thresholds
    original = np.asarray([row["cheap"] > cheap for row in rows])
    replacement = np.asarray([row["costly"] > expensive for row in rows])
    gold = np.asarray([row["label"] == "stable_positive" for row in rows])
    weights = np.asarray([WEIGHTS[row["label"]] for row in rows])
    gain = weights * ((replacement == gold).astype(int) - (original == gold).astype(int))
    return original, replacement, gain


def fit(rows, thresholds):
    features = margin_features(rows, thresholds[0])
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale == 0] = 1
    features = (features - mean) / scale
    target = decisions(rows, thresholds)[2]
    coefficients = np.linalg.solve(features.T @ features + 16 * np.eye(features.shape[1]),
                                   features.T @ (target - target.mean()))
    return mean, scale, coefficients


def summarize(rows, accept, reference, routed, topics):
    total = {kind: Counter(rows=0, accepted=0, admitted=0, removed=0) for kind in KINDS}
    utility_delta = {key: 0 for key in topics}
    for i, row in enumerate(rows):
        kind = row["label"]
        change = int(accept[i]) - int(reference[i])
        total[kind].update(rows=1, accepted=int(accept[i]), admitted=int(change > 0), removed=int(change < 0))
        utility_delta[row["topic"]] += WEIGHTS[kind] * (change if kind == "stable_positive" else -change)
    return dict(disposition=total, routed=len(routed),
                weighted_gain=sum(utility_delta.values()),
                gain_per_topic=paired_interval(list(utility_delta.values())))


def evaluate(development, held, thresholds):
    original, replacement, gain = decisions(held, thresholds)
    cheap_cutoff = thresholds[0]
    mu, scale, coefficients = fit([row for rows in development.values() for row in rows], thresholds)
    groups = defaultdict(list)
    for index, row in enumerate(held):
        groups[row["topic"]].append(index)
    hashes = [-int(sha256(f"same-passage-v1:{row['topic']}:{row['key']}".encode()).hexdigest(), 16)
              for row in held]
    distance = np.abs([row["cheap"] - cheap_cutoff for row in held])
    learned = ((margin_features(held, cheap_cutoff) - mu) / scale) @ coefficients
    priorities = dict(
        hash=hashes, boundary=-distance, reject_first=list(zip(~original, -distance)),
        accept_first=list(zip(original, -distance)), learned=learned, oracle=gain,
    )
    frozen = {
        "all_0.6b": summarize(held, original, original, [], groups),
        "all_4b": summarize(held, replacement, original, held, groups),
    }
    budgets = {}
    for divisor in (4, 2):
        routed = {}
        for method, values in priorities.items():
            order = []
            for group in groups.values():
                batch = sorted(group, key=lambda i: (values[i], hashes[i]), reverse=True)
                if method == "oracle":
                    batch = [i for i in batch if gain[i] > 0]
                order.extend(batch[:len(group) // divisor])
            chosen = original.copy()
            chosen[order] = replacement[order]
            routed[method] = summarize(held, chosen, original, order, groups)
        budgets[f"1/{divisor}"] = routed
    return dict(cutoffs=thresholds, baseline=frozen, budgets=budgets,
                ridge_coefficients=coefficients.tolist(), topics=len(groups))


def main(first, second, baseline, output):
    with first.open() as stream:
        a = json.load(stream)
    b = json.loads(second.read_bytes())
    if not a["order_offset"] == 0 or not b["order_offset"] == 1 or any(
            a[key] != b[key] for key in a if key != "order_offset"):
        raise ValueError("GPU scores did not reproduce on opposite collection order")
    if any(sum(panel["clipped"].values()) for panel in a["panels"].values()):
        raise ValueError("A model saw truncated evidence")
    panels = {name: a["panels"][name]["rows"] for name in ("core", "news", "robust")}
    development = {name: panels[name] for name in ("core", "news")}
    held = panels["robust"]
    anchored = json.loads(baseline.read_bytes())["arms"]["relevance"]["panels"]["robust"]
    threshold = tuple(cutoff(development, name) for name in ("cheap", "costly"))
    result = {
        "zero_cutoffs": evaluate(development, held, (0, 0)),
        "transferred_risk_cutoffs": evaluate(development, held, threshold),
    }
    for setup, key in (("zero_cutoffs", "zero"), ("transferred_risk_cutoffs", "calibrated")):
        observed = result[setup]["baseline"]["all_0.6b"]["disposition"]
        if any(row["accepted"] != anchored["decisions"][key][kind]["accept_after"]
               or row["rows"] != anchored["decisions"][key][kind]["total"] for kind, row in observed.items()):
            raise ValueError("The frozen low-cost model did not reproduce the public baseline")
    with output.open("x") as stream:
        json.dump(dict(result=result, gpu=a["gpu"], source_sha256=a["source_sha256"],
                       compact_revision=a["compact_revision"], strong_revision=a["strong_revision"],
                       score_script_sha256=a["code_sha256"], replicated_collection_offsets=[0, 1]),
                  stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("first", "second", "baseline", "output"):
        parser.add_argument(name, type=Path)
    main(**vars(parser.parse_args()))
