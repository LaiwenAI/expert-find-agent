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

    k = spec["top_k"]
    primary, comparator = spec["primary"], spec["primary_comparator"]
    empty = {row["id"]: {} for row in groups}
    draws = np.random.default_rng(spec["bootstrap_seed"]).integers(
        n, size=(spec["bootstrap_draws"], n),
    )
    gates = [spec[key] for key in ("coverage_floor", "conditional_risk_ceiling", "confidence")]
    output_arms = {}
    for name in arms:
        issued = sum(bool(query) for query in arms[name].values())
        best_gate = assess(n, issued, 0, 0, *gates)
        output_arms[name] = {
            "issuance_rate": issued / n,
            "issued_groups": issued,
            "maximum_possible_supported_p5": sum(map(len, arms[name].values())) / (n * k),
            "all_supported_gate": best_gate,
        }
    output_contrasts = {}
    # Keep each base retrieval route matched. The secondary arms cannot
    # be relabeled as the primary comparison after seeing the outputs.
    for candidate, baseline in (
        (primary, comparator),
        ("global_on_abstain", "deeper_on_abstain"),
    ):
        candidate_views, baseline_views = arms[candidate], arms[baseline]
        winning = {key: dict.fromkeys(views.keys() - baseline_views[key].keys(), LEVEL["supported"])
                   for key, views in candidate_views.items()}
        losing = {key: views.keys() - candidate_views[key].keys()
                  for key, views in baseline_views.items()}
        # Supporting all exclusive candidate views maximizes every paired
        # draw. Shared views use the same label and cancel exactly.
        lower, _ = interval(groups, winning, empty, k, draws)
        output_contrasts[f"{candidate}_vs_{baseline}"] = {
            "shared_exact_views": sum(len(a.keys() & baseline_views[key].keys())
                                      for key, a in candidate_views.items()),
            "candidate_only_views": sum(map(len, winning.values())),
            "baseline_only_views": sum(map(len, losing.values())),
            "requests_with_distinct_views": sum(bool(winning[key] or losing[key]) for key in winning),
            "maximum_possible_quality_gain": sum(map(len, winning.values())) / (n * k),
            "maximum_bootstrap_lower_bound": lower,
            "quality_gain_gate_possible": lower > spec["minimum_grounded_p5_gain"],
            "both_absolute_gates_possible": all(
                output_arms[name]["all_supported_gate"]["passes"] for name in (candidate, baseline)
            ),
        }
    main = output_contrasts[f"{primary}_vs_{comparator}"]
    write(output, {
        "groups": n, "slots": n * k, "completed_fields_by_form": completed,
        "pool_views": len(mapping), "shared_exact_views": main["shared_exact_views"],
        "primary_only_views": main["candidate_only_views"], "comparator_only_views": main["baseline_only_views"],
        "requests_with_arm_specific_views": main["requests_with_distinct_views"],
        "observable_best_quality_gain": main["maximum_possible_quality_gain"],
        "observable_worst_quality_gain": -main["baseline_only_views"] / (n * k),
        "maximum_bootstrap_lower_bound": main["maximum_bootstrap_lower_bound"],
        "quality_gain_gate_possible": main["quality_gain_gate_possible"],
        "absolute_gate_plan": design(n, *gates), "primary_arms": {name: output_arms[name] for name in (primary, comparator)},
        "all_frozen_arms": output_arms,
        "matched_rescue_contrasts": output_contrasts,
        "scope": "Output-only feasibility bounds. No evidence labels inform the estimates.",
    })


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "lock", "frame", "packet", "output"):
        cli.add_argument(name, type=Path)
    run(**vars(cli.parse_args()))
