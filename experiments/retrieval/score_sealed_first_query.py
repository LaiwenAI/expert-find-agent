"""Score the prespecified paired sealed enterprise study after reconciliation."""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.enterprise_review_gate import assess
from experiments.retrieval.check_joint_review import FIELDS, check, load, sha
from experiments.retrieval.enterprise_hybrid_rescue import summary
from experiments.retrieval.enterprise_routed_review import rows, write
from experiments.retrieval.score_enterprise_evidence_gate import (
    LEVEL, hit_bounds, paired_bounds, tally_views, validated_views,
)


def interval(frame, primary, reference, k, draws):
    extrema = []
    for row in frame:
        difference = paired_bounds([row], primary, reference, k)["hit_delta"]
        extrema.append((difference["lower"], difference["upper"]))
    estimates = np.asarray(extrema)[draws].mean(axis=1)
    return [float(np.quantile(estimates[:, 0], .025)),
            float(np.quantile(estimates[:, 1], .975))]


def run(protocol, lock, frame, packet, reconciliation, aggregate, trace, output):
    spec = json.loads(protocol.read_bytes())
    commitment = json.loads(lock.read_bytes())
    manifest = json.loads((packet / "manifest.json").read_bytes())
    reconciliation_hash = sha(reconciliation / "private-outcomes.json")
    reconciliation_summary = json.loads((reconciliation / "summary.json").read_bytes())
    if (commitment["public_protocol_sha256"] != sha(protocol)
            or manifest["frame_sha256"] != sha(frame)
            or manifest["lock_sha256"] != sha(lock)
            or reconciliation_summary["outcomes_sha256"] != reconciliation_hash
            or reconciliation_summary["packet_sha256"] != sha(packet / "manifest.json")
            or len(reconciliation_summary["review_sha256"]) < 2
            or len(set(reconciliation_summary["review_sha256"][:2])) != 2):
        raise ValueError("Evaluation inputs differ from the frozen protocol or independent reviews")
    if output.exists():
        raise FileExistsError(output)
    form = packet / "blind-review.jsonl"
    check(packet, form)
    groups = rows(frame)
    if (groups != commitment["entries"] or len(groups) != spec["groups"]
            or len({row["group"] for row in groups}) != spec["groups"]):
        raise ValueError("Changed frozen group frame")
    outcomes = json.loads((reconciliation / "private-outcomes.json").read_bytes())
    if not outcomes or any(any(values[field]["status"] in ("missing_review", "disagreement")
                               for field in FIELDS) for values in outcomes.values()):
        raise ValueError("Complete both blind reviews and resolve disagreements before scoring")
    metadata = json.loads(aggregate.read_bytes())
    private = json.loads(trace.read_bytes())
    if (metadata["panel_lock_sha256"] != sha(lock)
            or any(metadata.get(key) != val for key, val in private["metadata"].items())
            or any(metadata.get(key) != value for key, value in
                   summary(private["observations"], tuple(spec["arms"])).items())):
        raise ValueError("Trace and aggregate are not from the frozen run")
    observations = {row["id"]: row["arms"] for row in private["observations"]}
    if len(observations) != len(private["observations"]) or set(observations) != {row["id"] for row in groups}:
        raise ValueError("Incomplete timing frame")
    views, timing, report = {}, {}, {}
    draws = np.random.default_rng(spec["bootstrap_seed"]).integers(
        len(groups), size=(spec["bootstrap_draws"], len(groups)))
    mapping = load(packet / "private-mapping.jsonl")
    baseline = spec["primary_comparator"]
    for arm in spec["arms"]:
        path = packet / f"{arm}-predictions.jsonl"
        if sha(path) != manifest["outputs_sha256"][path.name]:
            raise ValueError("Changed arm predictions")
        view, statuses = validated_views(groups, rows(path), mapping, outcomes,
                                         spec["required_fields"], spec["top_k"])
        views[arm] = view
        counts = tally_views(groups, view, statuses)
        gate = assess(*(counts[k] for k in ("groups", "issued", "adverse", "unresolved")),
                      spec["coverage_floor"], spec["conditional_risk_ceiling"], spec["confidence"])
        timing[arm] = np.asarray([observations[row["id"]][arm]["ms"] for row in groups])
        if not np.isfinite(timing[arm]).all() or not (timing[arm] > 0).all():
            raise ValueError("Invalid request timing")
        report[arm] = dict(
            hit=hit_bounds(view, spec["top_k"]), absolute_gate=gate,
            serial_p95_ms=float(np.quantile(timing[arm], .95)),
            verification_checks=metadata["arms"][arm]["primary_pairs"] + metadata["arms"][arm]["rescue_pairs"],
            unresolved_slots=sum(level == LEVEL["unresolved"]
                                 for query in view.values() for level in query.values()),
        )
    primary = spec["primary"]
    pair = paired_bounds(groups, views[primary], views[baseline], spec["top_k"])
    limits = interval(groups, views[primary], views[baseline], spec["top_k"], draws)
    pair["bootstrap_ci95"] = limits
    pair["p95_delta_ms_ci95"] = np.quantile(
        np.quantile(timing[primary][draws], .95, axis=1)
        - np.quantile(timing[baseline][draws], .95, axis=1), [.025, .975]).tolist()
    pair["quality_gain_passes"] = limits[0] > spec["minimum_grounded_p5_gain"]
    pair["serial_p95_improvement_passes"] = (
        limits[0] >= -spec["quality_noninferiority_margin"]
        and report[primary]["absolute_gate"]["passes"]
        and report[baseline]["absolute_gate"]["passes"]
        and pair["p95_delta_ms_ci95"][1] < 0
    )
    write(output, dict(study=spec["study"], primary=primary, comparator=baseline,
                       groups=len(groups), arms=report, contrast=pair))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "lock", "frame", "packet", "reconciliation", "aggregate", "trace", "output"):
        cli.add_argument(name, type=Path)
    run(**vars(cli.parse_args()))
