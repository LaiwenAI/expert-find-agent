"""Audit cross-publisher risk, glossary agreement, and request wording."""

import argparse
from collections import Counter
from itertools import groupby
import json
from pathlib import Path

from expert_search.statistics import clustered_paired_interval


def decisions(row, threshold):
    return [value > threshold for value in row["scores"]]


def summarize(rows, choices):
    count = Counter()
    for row, selected in zip(rows, choices, strict=True):
        labels = row["labels"]
        support = sum(yes and label for yes, label in zip(selected, labels, strict=True))
        errors = sum(yes and not label for yes, label in zip(selected, labels, strict=True))
        count.update(accepted=support, unsupported=errors, wrong_request=bool(errors),
                     clean_nonempty=bool(support and not errors), exact=selected == labels)
    return dict(count)


def matched_risk(rows, cap, floor):
    entries = [(score, i, j, label) for i, row in enumerate(rows)
               for j, (score, label) in enumerate(zip(row["scores"], row["labels"], strict=True))
               if score > floor]
    chosen = [[False] * len(row["scores"]) for row in rows]
    best = summarize(rows, chosen)
    flags = set()
    for _, items in groupby(sorted(entries, reverse=True), key=lambda entry: entry[0]):
        group = list(items)
        flags.update(i for _, i, _, label in group if not label)
        if len(flags) > cap:
            break
        for _, i, j, _ in group:
            chosen[i][j] = True
        best = summarize(rows, chosen)
    return best


def lexical_dense_agreement(rows, lexical, dense_threshold, lexical_threshold):
    pairs = list(zip(rows, lexical, strict=True))
    if any((a["article"], a["role"], a["labels"]) != (b["article"], b["role"], b["labels"]) for a, b in pairs):
        raise ValueError("Prediction sets do not align")
    final = [[a and b for a, b in zip(decisions(dense, dense_threshold), decisions(sparse, lexical_threshold), strict=True)]
             for dense, sparse in pairs]
    return summarize(rows, final)


def clean_indicators(rows, threshold):
    outcome = []
    for row in rows:
        chosen = decisions(row, threshold)
        outcome.append(int(any(chosen) and all(not yes or label for yes, label in zip(chosen, row["labels"], strict=True))))
    return outcome


def analyze(results):
    report = {}
    for journal, wording_bank in results.items():
        report[journal] = {}
        for wording, arms in wording_bank.items():
            n = len(arms["scoped"]["requests"])
            cap = min(n // 20, arms["scoped"]["held"]["wrong_request"])
            scoped = clean_indicators(arms["scoped"]["requests"], arms["scoped"]["threshold"])
            analyzed = {}
            for arm, data in arms.items():
                rows, threshold = data["requests"], data["threshold"]
                deltas = [a - b for a, b in zip(clean_indicators(rows, threshold), scoped, strict=True)]
                analyzed[arm] = dict(
                    at_matched_wrong_cap=matched_risk(rows, cap, 0. if arm in ("dense", "lexical") else float("-inf")),
                    wording_cluster_delta_vs_scoped=clustered_paired_interval(deltas, [row["role"] for row in rows]),
                )
            dense, lexical = arms["dense"], arms["lexical"]
            report[journal][wording] = dict(
                requests=n, matched_cap=cap, policies=analyzed,
                frozen_glossary_agreement=lexical_dense_agreement(
                    dense["requests"], lexical["requests"], dense["threshold"], lexical["threshold"],
                ),
            )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = json.loads(args.input.read_bytes())
    result = raw.get("runs", [raw])[0]["results"]
    with args.output.open("x") as stream:
        json.dump(analyze(result), stream, indent=2)
        stream.write("\n")
