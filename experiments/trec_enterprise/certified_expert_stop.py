"""Certify that remaining sources cannot change ordered TREC expert recommendations."""

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean

import numpy as np

from expert_search.statistics import paired_interval
from experiments.followir.core17_reranking import digest, write_json
from experiments.trec_enterprise.expert_evidence_budget import FILES, TOP_K, load
from experiments.trec_enterprise.shared_source_budget import NOISE, TRIALS, make_cases


SCHEDULES = (("round_robin", 1), ("round_robin", 4), ("rank_first", 1), ("rank_first", 4))
SUPPORT_COUNTS = (1, 2)


def replay(case, truth_mode, support_count, schedule, width, certify):
    _, _, candidates, truth, _, orders, pair_draws, source_draws = case
    checked = Counter()
    accepted = {expert: [] for expert in candidates}
    remaining = {expert: len(orders[expert]) for expert in candidates}
    read = set()
    waves = 0
    rank = {expert: i for i, expert in enumerate(candidates)}
    while True:
        confirmed = [e for e in candidates if len(accepted[e]) >= support_count]
        cutoff = rank[confirmed[TOP_K - 1]] if certify and len(confirmed) >= TOP_K else len(candidates)
        eligible = [
            e for e in candidates
            if rank[e] < cutoff and e not in confirmed
            and len(accepted[e]) + remaining[e] >= support_count
        ]
        if not eligible:
            break
        # Compare the same priority and batching with and without the certificate.
        batch = sorted(eligible, key=lambda e: (checked[e], rank[e])) if schedule == "round_robin" else eligible
        waves += 1
        for e in batch[:width]:
            doc = orders[e][checked[e]]
            checked[e] += 1
            remaining[e] -= 1
            read.add(doc)
            positive = truth[e][doc] == 2
            if truth_mode == "perfect":
                passes = positive
            else:
                draw = pair_draws[e][doc] if truth_mode == "independent" else source_draws[doc]
                passes = draw < (.90 if positive else .01)
            if passes:
                accepted[e].append(doc)
    supported = [e for e in candidates if len(accepted[e]) >= support_count]
    unsupported = [e for e in candidates if len(accepted[e]) < support_count]
    ordered = (supported + unsupported)[:TOP_K]
    selected = [(e, tuple(accepted[e]) if e in supported else ()) for e in ordered]
    return selected, dict(checks=checked.total(), reads=len(read), waves=waves)


def run(directory, output):
    experts, documents = load(directory)
    observations = []
    for trial_index, case in enumerate(make_cases(experts, documents)):
        topic, expert_grades, _, evidence_grades, *_ = case
        for verifier in NOISE:
            for count in SUPPORT_COUNTS:
                oracle, _ = replay(case, verifier, count, "rank_first", 1, False)
                supported = sum(expert_grades[e] == 2 and any(evidence_grades[e][d] == 2 for d in docs)
                                for e, docs in oracle) / TOP_K
                for schedule, width in SCHEDULES:
                    for certify in (False, True):
                        selected, cost = replay(case, verifier, count, schedule, width, certify)
                        if selected != oracle:
                            raise AssertionError("Certificate changed candidate order or attached evidence")
                        observations.append(dict(topic=topic, trial=trial_index % TRIALS,
                                                 verifier=verifier, support=count, width=width, schedule=schedule,
                                                 certify=certify, supported_p5=supported, **cost))
    report = {}
    for verifier in NOISE:
        report[verifier] = {}
        for count in SUPPORT_COUNTS:
            for schedule, width in SCHEDULES:
                relevant = [
                    r for r in observations
                    if (r["verifier"], r["support"], r["schedule"], r["width"]) == (verifier, count, schedule, width)
                ]
                by_method = [[r for r in relevant if r["certify"] == cert] for cert in (False, True)]
                before, after = by_method
                deltas = {}
                for topic in experts:
                    pairs = [(a, b) for a, b in zip(before, after, strict=True) if a["topic"] == topic]
                    if len(pairs) != TRIALS or any(
                        (a["topic"], a["trial"]) != (b["topic"], b["trial"]) for a, b in pairs
                    ):
                        raise ValueError("Unequal trials across policies")
                    deltas[topic] = mean(a["checks"] - b["checks"] for a, b in pairs)
                report[verifier][f"{schedule}_support_{count}_width_{width}"] = dict(
                    baseline={cost: mean(r[cost] for r in before) for cost in ("checks", "reads", "waves")},
                    certified={cost: mean(r[cost] for r in after) for cost in ("checks", "reads", "waves")},
                    baseline_p95_checks=float(np.quantile([r["checks"] for r in before], .95)),
                    certified_p95_checks=float(np.quantile([r["checks"] for r in after], .95)),
                    paired_checks_saved=paired_interval(list(deltas.values())),
                    stopped_early=sum(a["checks"] > b["checks"] for a, b in zip(before, after, strict=True)),
                    cases=len(before), supported_p5=mean(r["supported_p5"] for r in before),
                )
    write_json(output, dict(inputs=FILES, script_sha256=digest(Path(__file__)), topics=len(experts),
                            trials=TRIALS, report=report, observations=observations))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    run(args.input, args.output)
