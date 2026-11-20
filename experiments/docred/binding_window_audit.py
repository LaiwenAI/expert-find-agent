"""Audit reference evidence coverage for the fixed-window binding experiment."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from experiments.docred import person_bound_evidence as names
from experiments.docred import work_owner_contrast as source
from experiments.followir.core17_reranking import digest, write_json


def audit(original, expanded, result):
    old = {doc["title"]: doc for doc, _ in source.documents(original, expanded, "test")}
    cases = {row["id"]: row for row in source.cases(original, expanded, "test")}
    coverage = {}
    for key, case in cases.items():
        doc = old[case["doc"]]
        selected = set(source.window(case).splitlines())
        canonical = [names.canonical(entity) for entity in doc["vertexSet"]]
        annotations = defaultdict(list)
        for label in doc["labels"]:
            sentences = {case["sentences"][i] for i in label.get("evidence", ())}
            link = canonical[label["h"]], canonical[label["t"]], label["r"]
            annotations[link].append(bool(sentences) and sentences <= selected)
        statuses = []
        for work, person in case["names"]:
            matches = annotations[(work, person, case["role"])]
            statuses.append(len(matches) == 1 and matches[0])
        coverage[key] = statuses
    report = {}
    for cap, routes in result["results"].items():
        report[cap] = {}
        for route, measurements in routes.items():
            totals = Counter()
            for row in measurements["outcomes"]:
                case = cases[row["id"]]
                if tuple(row["labeled"]) != case["labeled"]:
                    raise ValueError("Recorded labels differ from the released case")
                statuses = coverage[row["id"]]
                for accepted, labeled, complete in zip(row["acceptance"], row["labeled"], statuses, strict=True):
                    totals["known_with_all_reference_sentences"] += int(labeled and complete)
                    totals["accepted_with_all_reference_sentences"] += int(accepted and labeled and complete)
                    totals["accepted_known_other"] += int(accepted and labeled and not complete)
            report[cap][route] = dict(totals)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("original", "expanded", "result", "output"):
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    result = json.loads(args.result.read_bytes())
    report = audit(args.original, args.expanded, result)
    write_json(args.output, dict(results_sha256=digest(args.result), audit=report))
