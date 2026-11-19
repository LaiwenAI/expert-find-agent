"""Bound the paired short-window effect with partially adjudicated evidence views."""

import argparse
import json
from pathlib import Path

from experiments.retrieval.build_enterprise_window_review import POLICIES, response_signature, write
from experiments.retrieval.score_enterprise_evidence_gate import (
    FIELDS, LEVEL, evidence_status, hit_bounds, paired_bounds, records, sha, view_key,
)


TOP_K = 5


def load_reviews(packet, reconciled, trace):
    manifest_path = packet / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    mapping_path = packet / "private-mapping.jsonl"
    outcomes_path = reconciled / "private-outcomes.json"
    reconciliation = json.loads((reconciled / "summary.json").read_bytes())
    if (sha(mapping_path) != manifest["outputs_sha256"]["private-mapping.jsonl"]
            or sha(packet / "blind-review.jsonl") != manifest["outputs_sha256"]["blind-review.jsonl"]
            or sha(manifest_path) != reconciliation["packet_sha256"]
            or sha(outcomes_path) != reconciliation["outcomes_sha256"]
            or sha(trace) != manifest["inputs_sha256"]["trace"]):
        raise ValueError("Changed frozen trace or evidence review")
    mapping = records(mapping_path)
    outcomes = json.loads(outcomes_path.read_bytes())
    if len(mapping) != len(outcomes) or len(mapping) != manifest["candidate_views"]:
        raise ValueError("Changed review case count")
    if len({row["case_id"] for row in mapping}) != len(mapping):
        raise ValueError("Duplicate review case ID")
    if {row["case_id"] for row in mapping} != set(outcomes):
        raise ValueError("Review and outcome inventories differ")
    reviewed = {}
    for row in mapping:
        coordinates = tuple(sorted((doc, start, end) for doc, start, end, _ in view_key(row["sources"])))
        key = row["query_id"], row["expert_id"], coordinates
        if key in reviewed:
            raise ValueError("Duplicate expert and citation view")
        case = row["case_id"]
        reviewed[key] = case, LEVEL[evidence_status(outcomes[case], FIELDS)]
    return reviewed, manifest


def compare(trace, packet, reconciled, output, priority=None):
    if output.exists() or (priority is not None and priority.exists()):
        raise FileExistsError("Use fresh aggregate and priority paths")
    reviewed, manifest = load_reviews(packet, reconciled, trace)
    saved = json.loads(trace.read_bytes())
    if (saved["queries_sha256"] != manifest["inputs_sha256"]["queries"]
            or saved["corpus_sha256"] != manifest["inputs_sha256"]["corpus"]
            or saved["aggregate_sha256"] != manifest["inputs_sha256"]["aggregate"]):
        raise ValueError("Trace inputs differ from the packet")
    requests = saved["records"]
    if len(requests) != 80 or len({row["query_id"] for row in requests}) != len(requests):
        raise ValueError("Expected 80 unique development requests")

    frames = {key: [] for key in ("all", "changed", "people_or_order", "evidence_only")}
    methods = {method: {} for method in POLICIES}
    priority_groups = []
    seen_reviewed = set()
    for row in requests:
        query = row["query_id"]
        if set(row["views"]) != set(POLICIES):
            raise ValueError("Each query requires both methods")
        frame_row = dict(id=query, group=query)
        frames["all"].append(frame_row)
        reference, criterion = (response_signature(row["views"][method]) for method in POLICIES)
        if reference != criterion:
            frames["changed"].append(frame_row)
            same_people = tuple(expert for expert, _ in reference) == tuple(expert for expert, _ in criterion)
            frames["evidence_only" if same_people else "people_or_order"].append(frame_row)
        for method, items in zip(POLICIES, (reference, criterion), strict=True):
            if len(items) > TOP_K or len({expert for expert, _ in items}) != len(items):
                raise ValueError("Invalid issued slots")
            if any(not citations or len(set(citations)) != len(citations) for _, citations in items):
                raise ValueError("Invalid citation coordinates")
            keys = [(query, expert, citations) for expert, citations in items]
            seen_reviewed.update(key for key in keys if key in reviewed)
            methods[method][query] = {key: reviewed[key][1] if key in reviewed else LEVEL["unresolved"]
                                       for key in keys}
        left, right = (methods[method][query] for method in POLICIES)
        nonshared = left.keys() ^ right.keys()
        if not nonshared <= reviewed.keys():
            raise ValueError("An influential citation view is missing from the review packet")
        cases = sorted(reviewed[key][0] for key in nonshared if reviewed[key][1] == LEVEL["unresolved"])
        if cases:
            priority_groups.append((query, cases))
    if seen_reviewed != set(reviewed):
        raise ValueError("A reviewed view does not match the scored responses")

    report = {}
    for group, frame in frames.items():
        if not frame:
            continue
        reference, criterion = ({row["id"]: methods[m][row["id"]] for row in frame} for m in POLICIES)
        bounds = paired_bounds(frame, criterion, reference, TOP_K)
        report[group] = dict(
            queries=len(frame), reference=hit_bounds(reference, TOP_K), criterion=hit_bounds(criterion, TOP_K),
            unresolved_nonshared_views=sum(
                side[key] == LEVEL["unresolved"] for row in frame for side, other in
                ((reference[row["id"]], criterion[row["id"]]), (criterion[row["id"]], reference[row["id"]]))
                for key in side.keys() - other.keys()),
            paired=bounds)
    public = dict(
        review_cases=len(reviewed),
        case_status={name: sum(level == code for _, level in reviewed.values()) for name, code in LEVEL.items()},
        diagnostic_strata=report,
        scope="Finite development workload. Person, request and exact evidence coordinates share labels. "
              "All three evidence fields must be supported. Unresolved labels are bounded, not imputed. "
              "The calculation does not give a sampling confidence interval or validate production precision.",
    )
    write(output, public)
    if priority is not None:
        priority_groups.sort(key=lambda group: (-len(group[1]), group[0]))
        write(priority, dict(groups=[dict(case_ids=cases) for _, cases in priority_groups]))
    return public


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("trace", "packet", "reconciled", "output"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("--priority", type=Path)
    args = parser.parse_args()
    result = compare(**vars(args))
    print(json.dumps(result, indent=2))
