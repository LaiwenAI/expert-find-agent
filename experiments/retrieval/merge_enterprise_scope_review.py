"""Consolidate exact-anchor development reviews without exposing model labels."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

from experiments.retrieval.enterprise_qualification_scope_review import build_packet, validate
from experiments.retrieval.enterprise_routed_review import digest, rows, write
from experiments.retrieval.export_enterprise_requirement_role_review import SEED
from experiments.retrieval.score_enterprise_requirement_role_review import (
    QUALIFICATION, frozen_predictions, load_forms,
)


def local_records(path):
    return json.loads(path.read_bytes())


def anchored_rows(forms, mapping, predictions):
    if forms.keys() != mapping.keys() or forms.keys() != predictions.keys():
        raise ValueError("Form, mapping and measured case sets disagree")
    return [(forms[key], mapping[key], predictions[key]) for key in forms]


def collect(role_packet, role_trace, compact_packet, compact_trace, scope_packet, scope_sidecar):
    forms, labels, _ = frozen_predictions(role_packet, role_trace)
    owner = {key: item["request_id"] for key, item in local_records(role_packet/"private-method-map.json").items()}
    for key, form in forms.items():
        expected = sha256(f"{SEED}:{owner[key]}:{form['start']}:{form['end']}".encode()).hexdigest()[:24]
        if expected != key:
            raise ValueError("Role case ID does not bind to original offsets")
    yield "prompt_scope", anchored_rows(forms, owner, {
        key: tuple(role == QUALIFICATION for role in votes.values()) for key,votes in labels.items()})

    manifest = local_records(compact_packet/"summary.json")
    source = compact_packet/"review.jsonl"
    if digest(source) != manifest["form_sha256"] or digest(compact_trace) != manifest["source_disagreements_sha256"]:
        raise ValueError("Changed compact recheck review or source")
    forms = load_forms(source)
    coordinates = {}
    labels = {}
    for row in local_records(compact_trace):
        key = sha256(f"compact-review-v1:{row['query_id']}:{row['start']}:{row['end']}".encode()).hexdigest()[:24]
        if key not in forms or (forms[key]["start"], forms[key]["end"]) != (row["start"],row["end"]):
            raise ValueError("Compact case ID does not bind to original offsets")
        coordinates[key] = row["query_id"]
        labels[key] = tuple(value == QUALIFICATION for value in row["labels"].values())
    yield "compact_recheck", anchored_rows(forms, coordinates, labels)

    manifest = local_records(scope_packet/"manifest.json")
    sidecar = local_records(scope_sidecar)
    source, mapping = scope_packet/"review.jsonl", scope_packet/"private-method-map.jsonl"
    if (digest(source) != manifest["original_form_sha256"] or digest(mapping) != manifest["private_mapping_sha256"]
            or sidecar["packet_sha256"] != digest(source)):
        raise ValueError("Changed scope review or sidecar")
    forms = load_forms(source)
    owner = {row["case_id"]: row["request"] for row in rows(mapping)}
    decisions = {key: (row["original_selected"], row["direct_gate"]) for key,row in sidecar["decisions"].items()}
    if any(type(value) is not bool for values in decisions.values() for value in values):
        raise ValueError("Unresolved model comparison")
    yield "qualification_scope", anchored_rows(forms, owner, decisions)


def build(queries, role_packet, role_trace, compact_packet, compact_trace, scope_packet, scope_sidecar, output):
    source = {row["id"]: row["text"] for row in rows(queries)}
    anchors = defaultdict(dict)
    count = {}
    for family, items in collect(role_packet, role_trace, compact_packet, compact_trace, scope_packet, scope_sidecar):
        count[family] = len(items)
        for form, qid, labels in items:
            start, end = form["start"], form["end"]
            if (source[qid] != form["full_request"] or not 0 <= start < end <= len(source[qid])
                    or source[qid][start:end] != form["target"]):
                raise ValueError("Audit does not refer to the same original request and excerpt")
            key = qid, start, end
            if family in anchors[key]:
                raise ValueError("Duplicate anchor within one experiment")
            anchors[key][family] = labels
    items = [dict(id=qid, start=start, end=end, full_request=source[qid], target=source[qid][start:end], chosen=None)
             for qid,start,end in sorted(anchors)]
    build_packet(items, [None]*len(items), len(items), digest(queries), output)
    forms, linkage = load_forms(output/"review.jsonl"), load_forms(output/"private-method-map.jsonl")
    if forms.keys() != linkage.keys():
        raise ValueError("Review rows and aliases disagree")
    priorities = []
    for case, row in forms.items():
        qid = linkage[case]["request"]
        families = anchors[qid,row["start"],row["end"]]
        disagreements = sorted(name for name,votes in families.items() if len(set(votes))>1)
        priorities.append(dict(case_id=case, families=sorted(families), boundary_families=disagreements))
    priorities.sort(key=lambda row: (-len(row["boundary_families"]), -len(row["families"]), row["case_id"]))
    write(output/"private-priorities.jsonl", priorities, lines=True)
    validate(output, output/"reviewer-a.jsonl", output/"reviewer-b.jsonl")
    result = dict(per_experiment=count, total_input_cases=sum(count.values()), distinct_source_anchors=len(anchors),
                  requests=len({key[0] for key in anchors}),
                  anchor_reuse=dict(sorted(Counter(len(entry) for entry in anchors.values()).items())),
                  families_disagreeing_on_qualification=dict(sorted(
                      Counter(len(x["boundary_families"]) for x in priorities).items())),
                  human_judgments=0)
    write(output/"summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ("queries","role_packet","role_trace","compact_packet","compact_trace","scope_packet","scope_sidecar","output"):
        cli.add_argument(key, type=Path)
    build(**vars(cli.parse_args()))
