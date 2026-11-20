"""Separate paired ordering and observed risk from frozen issuance thresholds."""

import argparse
from itertools import groupby
import json
from pathlib import Path

from expert_search.statistics import clustered_paired_interval


def ordered(case):
    labels, scores = case["known"], case["scores"]
    return int(scores[labels.index(True)] > scores[labels.index(False)])


def matched(evidence, cap):
    records = [
        (score, i, known) for i, case in enumerate(evidence)
        for score, known in zip(case["scores"], case["known"], strict=True)
    ]
    positives, rivals = set(), set()
    best = dict(declared=0, wrong=0, clean=0)
    for _, bucket in groupby(sorted(records, reverse=True), key=lambda row: row[0]):
        for _, i, known in bucket:
            (positives if known else rivals).add(i)
        if len(rivals) > cap:
            break
        best = dict(declared=len(positives), wrong=len(rivals), clean=len(positives - rivals))
    return best


def analyze(raw):
    output = {}
    for journal, views in raw["results"].items():
        output[journal] = {}
        for view, arms in views.items():
            output[journal][view] = {}
            reference = arms["shared"]["evidence"]
            cap = arms["shared"]["held"]["wrong_declared"]
            for arm, row in arms.items():
                evidence = row["evidence"]
                pairs = list(zip(evidence, reference, strict=True))
                if any((a["article"], a["role"], a["candidates"], a["known"])
                       != (b["article"], b["role"], b["candidates"], b["known"])
                       for a, b in pairs):
                    raise ValueError("Evaluation rows no longer align")
                output[journal][view][arm] = dict(
                    cap=cap, at_matched_wrong_cap=matched(evidence, cap),
                    paired_ordering_vs_shared=clustered_paired_interval(
                        [ordered(a) - ordered(b) for a, b in pairs], [a["article"] for a, _ in pairs],
                    ),
                )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.result.read_bytes())
    with args.output.open("x") as stream:
        json.dump(analyze(data.get("runs", [data])[0]), stream, indent=2)
        stream.write("\n")
