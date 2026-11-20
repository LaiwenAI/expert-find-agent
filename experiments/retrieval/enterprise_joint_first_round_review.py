"""Create method-blind exact-span reviews for the development comparison."""

import argparse
from pathlib import Path
import json

from experiments.retrieval import build_exact_enterprise_review as review
from experiments.retrieval.check_joint_review import check, FIELDS
from experiments.retrieval.enterprise_joint_first_round import ARMS
from experiments.retrieval.enterprise_project_role_rescue import profile_subset
from experiments.retrieval.enterprise_routed_review import digest, rows, write


def build(dataset, requests, trace, report, output):
    launch = json.loads(report.read_bytes())
    if digest(trace) != launch["trace_sha256"] or digest(requests) != launch["queries_sha256"]:
        raise ValueError("Changed first-round development evaluation inputs")
    q = review.index(rows(requests), lambda row: row["id"])
    observations = review.index(json.loads(trace.read_bytes())["observations"], lambda row: row["id"])
    if q.keys() != observations.keys():
        raise ValueError("Missing exact query or predictions")
    manifest = json.loads((dataset / "manifest.json").read_bytes())
    experts, corpus = (profile_subset(dataset, key, manifest) for key in ("experts", "corpus"))
    methods = {}
    for arm in ARMS:
        methods[arm] = {}
        for key, run in observations.items():
            methods[arm][key] = [
                (item["expert"], review.sources(
                    corpus, item["expert"], [dict(source_id=ref[0], start=ref[1], end=ref[2])
                                              for ref in item["citations"]]))
                for item in run["arms"][arm]["issued"]
            ]
    forms, mapping, predictions, counts = review.packet_rows(q, experts, methods)
    for view, private in zip(forms, mapping, strict=True):
        key = private["query_id"]
        view["prior_qualifications"] = [q[key]["text"][a:b]
                                         for a, b in observations[key]["arms"][ARMS[0]]["qualifications"]]
    output.mkdir(mode=0o700)
    for name in ("blind-review.jsonl", "reviewer-a.jsonl", "reviewer-b.jsonl"):
        write(output / name, forms, lines=True)
    write(output / "private-mapping.jsonl", mapping, lines=True)
    for arm, records in predictions.items():
        write(output / f"{arm}-predictions.jsonl", records, lines=True)
    committed = ["blind-review.jsonl", "private-mapping.jsonl",
                 *(f"{arm}-predictions.jsonl" for arm in predictions)]
    write(output / "manifest.json", dict(candidate_views=len(forms), queries=len(q),
          results_sha256=digest(report), trace_sha256=digest(trace),
          outputs_sha256={name: digest(output / name) for name in committed}))
    checked = check(output, output / "blind-review.jsonl")
    if any(checked["counts"][field] != {"missing": len(forms)} for field in FIELDS):
        raise ValueError("New review must contain no judgments")
    reference = {rec["case_id"] for row in predictions[ARMS[0]] for rec in row["recommendations"]}
    bounds = {}
    for arm in ARMS[1:]:
        compared = {rec["case_id"] for row in predictions[arm] for rec in row["recommendations"]}
        gains, losses = len(compared - reference), len(reference - compared)
        denominator = review.TOP * len(q)
        bounds[arm] = dict(new_views=gains, removed_views=losses,
                           possible_grounded_p5_delta=[-losses / denominator, gains / denominator])
    counts.update(queries=len(q), distinct_review_cards=len(forms), judgments=0,
                  unjudged_p5_bounds_vs_prefixes=bounds)
    write(output / "summary.json", counts)
    print(json.dumps(counts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "requests", "trace", "report", "output"):
        parser.add_argument(name, type=Path)
    build(**vars(parser.parse_args()))
