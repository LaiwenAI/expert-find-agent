"""Measure headroom in the frozen enterprise rescue pool without publishing row data."""

import argparse
import json
from pathlib import Path

from experiments.retrieval.enterprise_gap_budget_replay import ARMS, BUDGETS, prepare
from experiments.retrieval.enterprise_routed_review import digest, write


def actions(events, budget):
    return tuple((event["expert"], tuple(event["span"])) for event in events[:budget])


def accepted(events, budget):
    return {a for a, event in zip(actions(events, budget), events[:budget], strict=True) if event["accepted"]}


def run(prepared, initial, exhaustive, replay, output):
    panel, old, full, live = (json.loads(path.read_bytes()) for path in (prepared, initial, exhaustive, replay))
    if panel["trace_sha256"] != digest(initial) or full["job"] != "43026":
        raise ValueError("Changed baseline")
    if full["prepared_sha256"] != digest(prepared) or live["prepared_sha256"] != digest(prepared):
        raise ValueError("Upper bound and replay require the same prepared panel")
    queries = prepare(panel, old)
    expected = {(q["query"], a["expert"], a["span"]) for q in queries for a in q["actions"]}
    scored = {}
    for row in full["private_results"]:
        for group in row["decisions"].values():
            for item in group:
                key = row["query"], row["expert"], (item["id"], item["start"], item["end"])
                if key in scored and scored[key] != item["supported"]:
                    raise ValueError("Inconsistent reference scores")
                scored[key] = item["supported"]
    if expected != scored.keys():
        raise ValueError("Historical scores do not cover the full action pool")
    passed = {key for key, supported in scored.items() if supported}
    baseline_answered = {q["query"] for q in queries if q["issued"]}
    successful_queries = {key[0] for key in passed}
    records = live["records"]
    for row in records:
        for method in ARMS:
            for event in row["arms"][method]:
                key = row["query"], event["expert"], tuple(event["span"])
                if event["accepted"] != scored[key]:
                    raise ValueError("Replayed and exhaustive checks disagree")
    comparisons = {}
    for budget in BUDGETS:
        static = [row["arms"]["static_bottleneck"] for row in records]
        adaptive = [row["arms"]["adaptive_gap"] for row in records]
        comparisons[budget] = dict(
            changed_action_order=sum(actions(x, budget) != actions(y, budget) for x, y in zip(static, adaptive)),
            changed_accepted_views=sum(accepted(x, budget) != accepted(y, budget) for x, y in zip(static, adaptive)),
        )
    write(output, dict(
        audit_code_sha256=digest(Path(__file__)), replay_code_sha256=live["code_sha256"],
        exhaustive_reference_job=full["job"], available_windows=len(expected), supported_windows=len(passed),
        requests_with_supported_window=len(successful_queries),
        first_provisional_answer_upper_bound=len(successful_queries - baseline_answered),
        comparisons=comparisons,
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ("prepared", "initial", "exhaustive", "replay", "output"):
        cli.add_argument(key, type=Path)
    run(**vars(cli.parse_args()))
