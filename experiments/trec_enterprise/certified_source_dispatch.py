"""Compare batched evidence schedules with exact stopping and stable citations."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

from expert_search.statistics import clustered_paired_interval
from experiments.followir.core17_reranking import digest, write_json
from experiments.trec_enterprise.certified_expert_stop import SUPPORT_COUNTS, replay
from experiments.trec_enterprise.expert_evidence_budget import FILES, TOP_K, load
from experiments.trec_enterprise.shared_source_budget import NOISE, TRIALS, make_cases


METHODS = ("rank_first", "cache_first", "shared", "pivotal", "prefix_two", "prefix_four")
PREFIX = {"prefix_two": 2, "prefix_four": 4}
WIDTH = 4


def schedule_batch(method, eligible, frontiers, opened, rank, accepted, remaining, count):
    if method == "cache_first":
        return sorted(eligible, key=lambda e: (frontiers[e] not in opened, rank[e]))[:WIDTH]
    by_source = defaultdict(list)
    for expert in eligible:
        by_source[frontiers[expert]].append(expert)

    def priority(item):
        doc, owners = item
        if method == "pivotal":
            # Either outcome may reduce outstanding work. This uses only visible counts.
            weight = sum(
                (1 + int(len(accepted[e]) + 1 == count)
                 + int(len(accepted[e]) + remaining[e] == count)) / (1 + rank[e])
                for e in owners
            )
        else:
            weight = len(owners)
        return doc not in opened, -weight, rank[owners[0]], doc

    ordered = sorted(by_source.items(), key=priority)
    return [expert for _, owners in ordered for expert in owners][:WIDTH]


def dispatch(case, noise, count, method):
    _, _, candidates, truth, _, orders, pair_draws, source_draws = case
    checked = Counter()
    accepted = {e: [] for e in candidates}
    remaining = {e: len(orders[e]) for e in candidates}
    opened = set()
    rank = {e: i for i, e in enumerate(candidates)}
    waves = 0
    new_reads_in_wave = 0
    paid_checks = 0
    while True:
        confirmed = [e for e in candidates if len(accepted[e]) >= count]
        boundary = rank[confirmed[TOP_K - 1]] if len(confirmed) >= TOP_K else len(candidates)
        eligible = [e for e in candidates if rank[e] < boundary and len(accepted[e]) < count
                    and len(accepted[e]) + remaining[e] >= count]
        if not eligible:
            remaining_experts = [e for e in candidates if len(accepted[e]) < count]
            result = [(e, tuple(accepted[e]) if len(accepted[e]) >= count else ())
                      for e in (confirmed + remaining_experts)[:TOP_K]]
            return result, dict(checks=paid_checks, reads=len(opened), waves=waves,
                                source_opening_waves=new_reads_in_wave)
        if method in PREFIX:
            jobs = [(e, doc) for e in eligible for doc in orders[e][checked[e]:checked[e] + PREFIX[method]]]
            jobs = jobs[:WIDTH]
        else:
            frontiers = {e: orders[e][checked[e]] for e in eligible}
            batch = eligible[:WIDTH] if method == "rank_first" else schedule_batch(
                method, eligible, frontiers, opened, rank, accepted, remaining, count
            )
            jobs = [(e, frontiers[e]) for e in batch]
        new_reads_in_wave += any(doc not in opened for _, doc in jobs)
        waves += 1
        paid_checks += len(jobs)
        for e, doc in jobs:
            opened.add(doc)
            # A parallel request may finish after an earlier result makes it irrelevant.
            # Charge the request and retain only the canonical relevant prefix.
            if len(accepted[e]) >= count or len(accepted[e]) + remaining[e] < count:
                continue
            if doc != orders[e][checked[e]]:
                raise AssertionError("Source order changed")
            checked[e] += 1
            remaining[e] -= 1
            positive = truth[e][doc] == 2
            if noise == "perfect":
                passes = positive
            else:
                draw = pair_draws[e][doc] if noise == "independent" else source_draws[doc]
                passes = draw < (.90 if positive else .01)
            if passes:
                accepted[e].append(doc)


def run(directory, output):
    if output.exists():
        raise FileExistsError(output)
    experts, documents = load(directory)
    measurements = defaultdict(list)
    groups = []
    for case in make_cases(experts, documents):
        groups.append(case[0])
        for noise in NOISE:
            for count in SUPPORT_COUNTS:
                expected, _ = replay(case, noise, count, "rank_first", 1, False)
                reference, old_cost = replay(case, noise, count, "rank_first", WIDTH, True)
                if expected != reference:
                    raise AssertionError("Reference stopping certificate changed")
                for method in METHODS:
                    output_rows, work = dispatch(case, noise, count, method)
                    if output_rows != expected:
                        raise AssertionError(f"Changed ordered expert IDs or source IDs: {method}")
                    if method == "rank_first" and any(work[k] != v for k, v in old_cost.items()):
                        raise AssertionError("Fixed reference cost changed")
                    measurements[(noise, count, method)].append(work)
    table = {}
    for noise in NOISE:
        table[noise] = {}
        for count in SUPPORT_COUNTS:
            base = measurements[(noise, count, "rank_first")]
            results = {}
            for method in METHODS:
                rows = measurements[(noise, count, method)]
                fields = tuple(rows[0])
                results[method] = dict(
                    cases=len(rows),
                    average={k: mean(r[k] for r in rows) for k in fields},
                    p95={k: float(np.quantile([r[k] for r in rows], .95)) for k in fields},
                    paired_reduction={k: clustered_paired_interval(
                        [a[k] - b[k] for a, b in zip(base, rows, strict=True)], groups
                    ) | {"sampling_unit": "whole TREC topic; trial-weighted mean"} for k in fields},
                )
            table[noise][count] = results
    write_json(output, dict(
        source_hashes=FILES, script_sha256=digest(Path(__file__)), topics=len(experts), trials=TRIALS,
        max_batch_width=WIDTH, equality_checks=len(groups) * len(NOISE) * len(SUPPORT_COUNTS) * len(METHODS),
        report=table,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.input, args.output)
