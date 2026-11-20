"""Audit attainable frozen-study endpoints before reviewing evidence labels."""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.enterprise_review_gate import assess, design
from experiments.retrieval.check_joint_review import FIELDS, check, load, sha
from experiments.retrieval.score_enterprise_evidence_gate import LEVEL, validated_views
from experiments.retrieval.score_sealed_first_query import interval
from experiments.retrieval.enterprise_routed_review import rows, write


def run(protocol, lock, frame, packet, output):
    spec = json.loads(protocol.read_bytes())
    commitment = json.loads(lock.read_bytes())
    manifest = json.loads((packet / "manifest.json").read_bytes())
    if (commitment["public_protocol_sha256"] != sha(protocol) or manifest["frame_sha256"] != sha(frame)
            or manifest["lock_sha256"] != sha(lock)):
        raise ValueError("Inputs do not match the frozen protocol")
    groups = rows(frame)
    n = spec["groups"]
    if (groups != commitment["entries"] or len(groups) != n
            or len({row["group"] for row in groups}) != n or spec["requests_per_group"] != 1):
        raise ValueError("Expected one frozen query per group")

    # Only count completed review fields; the bounds treat every evidence label as unknown.
    completed = {}
    for form in ("reviewer-a.jsonl", "reviewer-b.jsonl"):
        checked = check(packet, packet / form)
        completed[form] = {field: checked["cases"] - counts.get("missing", 0)
                           for field, counts in checked["counts"].items()}
    mapping = load(packet / "private-mapping.jsonl")
    unresolved = {field: {"label": None, "status": "missing_review"} for field in FIELDS}
    unknown = dict.fromkeys(mapping, unresolved)
    arms = {}
    for arm in spec["arms"]:
        path = packet / f"{arm}-predictions.jsonl"
        if sha(path) != manifest["outputs_sha256"][path.name]:
            raise ValueError("Arm predictions do not match the frozen packet")
        arms[arm], _ = validated_views(groups, rows(path), mapping, unknown, spec["required_fields"], spec["top_k"])

    primary = arms[spec["primary"]]
    comparator = arms[spec["primary_comparator"]]
    primary_only, comparator_only = [], []
    shared_count = 0
    for row in groups:
        key = row["id"]
        a, b = primary[key].keys(), comparator[key].keys()
        shared_count += len(a & b)
        primary_only.append(len(a - b))
        comparator_only.append(len(b - a))

    # The same best or worst assignment is valid in every bootstrap draw.
    # Empty and shared slots cancel; only arm-specific views can change P@5.
    k = spec["top_k"]
    best = {key: dict.fromkeys(a.keys() - comparator[key].keys(), LEVEL["supported"])
            for key, a in primary.items()}
    empty = {key: {} for key in primary}
    draws = np.random.default_rng(spec["bootstrap_seed"]).integers(
        n, size=(spec["bootstrap_draws"], n),
    )
    best_lower, _ = interval(groups, best, empty, k, draws)
    gates = [spec[key] for key in ("coverage_floor", "conditional_risk_ceiling", "confidence")]
    output_arms = {}
    for name in (spec["primary"], spec["primary_comparator"]):
        issued = sum(bool(query) for query in arms[name].values())
        best_gate = assess(n, issued, 0, 0, *gates)
        output_arms[name] = {
            "issuance_rate": issued / n,
            "issued_groups": issued,
            "maximum_possible_supported_p5": sum(map(len, arms[name].values())) / (n * k),
            "all_supported_gate": best_gate,
        }
    write(output, {
        "groups": n, "slots": n * k, "completed_fields_by_form": completed,
        "pool_views": len(mapping), "shared_exact_views": shared_count,
        "primary_only_views": sum(primary_only), "comparator_only_views": sum(comparator_only),
        "requests_with_arm_specific_views": sum(a + b > 0 for a, b in zip(primary_only, comparator_only, strict=True)),
        "observable_best_quality_gain": sum(primary_only) / (n * k),
        "observable_worst_quality_gain": -sum(comparator_only) / (n * k),
        "maximum_bootstrap_lower_bound": best_lower,
        "quality_gain_gate_possible": best_lower > spec["minimum_grounded_p5_gain"],
        "absolute_gate_plan": design(n, *gates), "primary_arms": output_arms,
        "scope": "Output-only feasibility bounds. No evidence labels inform the estimates.",
    })


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "lock", "frame", "packet", "output"):
        cli.add_argument(name, type=Path)
    run(**vars(cli.parse_args()))
