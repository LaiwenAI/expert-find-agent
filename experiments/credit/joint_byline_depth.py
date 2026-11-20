"""Probe deeper candidate pools with an exact owner-postings join."""

import argparse
from collections import defaultdict
from functools import partial
from itertools import product
import json
import os
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from experiments.credit import open_topic_role_bundles as base
from experiments.followir.core17_reranking import write_json


DEPTHS = 8, 16, 32, 64


def pair_key(pair, ranks):
    a, b = ranks[0][pair[0]], ranks[1][pair[1]]
    return max(a, b), a + b, pair


def posting_plan(byline, pool, ranks, fallback, depth):
    # Rows are visited in increasing topic rank. A distinct optimal
    # pair for one owner uses only their first two rows per topic.
    owners = defaultdict(lambda: ([], []))
    for side, ranking in enumerate(ranks):
        for doi in ranking:
            for owner in byline[doi]:
                rows = owners[owner][side]
                if len(rows) < 2:
                    rows.append(doi)
    first = min(
        ((a, b) for left, right in owners.values() for a, b in product(left, right) if a != b),
        key=lambda pair: pair_key(pair, ranks), default=None,
    )
    if first is None:
        return fallback
    available = sorted(pool - set(first))
    if not available:
        return list(first)

    def gain(doi):
        links = sum(
            anchor in ranks[side] and doi in ranks[1 - side] and not byline[anchor].isdisjoint(byline[doi])
            for anchor in first for side in range(2)
        )
        return links, -min(facet.get(doi, depth + 1) for facet in ranks)

    return [*first, max(available, key=gain)]


def probe(catalogue, topics, atoms, offset, depth):
    byline = catalogue.byline
    ranks = tuple(topics[topic] for topic, _ in atoms)
    pool = set().union(*ranks)
    # Use a stable tie-break for the same fallback in both implementations.
    fallback = sorted(pool, key=lambda d: (min(side.get(d, depth + 1) for side in ranks), d))[:3]
    calls = (
        ("pair_scan", partial(base.joint_sources, pool, ranks, fallback,
                              lambda a, b: not byline[a].isdisjoint(byline[b]), depth)),
        ("owner_postings", partial(posting_plan, byline, pool, ranks, fallback, depth)),
    )
    times = {}
    routes = {}
    for name, call in calls[offset:] + calls[:offset]:
        start = perf_counter_ns()
        routes[name] = call()
        times[name] = (perf_counter_ns() - start) / 1000
    if routes["pair_scan"] != routes["owner_postings"]:
        raise AssertionError("Owner-postings plan differs from exhaustive search")
    return times


def timings(catalogue, rankings, queries, depth, offset):
    for atoms in queries[:5]:
        probe(catalogue, rankings, atoms, 0, depth)
    observations = [probe(catalogue, rankings, atoms, i % 2, depth)
                    for repeat in range(10) for i, atoms in enumerate(queries, offset + repeat)]
    return {arm: {f"p{int(100 * q)}_us": float(np.quantile([m[arm] for m in observations], q))
                  for q in (.50, .95)} for arm in observations[0]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("fixture", "checkpoint", "output"):
        p.add_argument(name, type=Path)
    p.add_argument("sources", type=Path, nargs="+")
    p.add_argument("--order-offset", type=int, default=0)
    args = p.parse_args()
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require a fresh result on a Slurm compute node")
    catalogue, _ = base.pinned.load_fixture(args.fixture, args.sources)
    subjects = base.publisher_topics(args.sources)
    if subjects.keys() != catalogue.ownership.keys():
        raise ValueError("Changed topic inventory")
    gold = base.annotation_postings(catalogue, subjects)
    queries = base.query_panel(gold)
    topics = sorted({topic for pair in queries for topic, _ in pair})
    groups = ["|".join(sorted((a[0], b[0]))) for a, b in queries]
    rankings, model = base.topic_rankings(catalogue, topics, args.checkpoint, width=max(DEPTHS))
    union = {doi: {role for a in item["authors"] for role in a["roles"]}
             for doi, item in catalogue.ownership.items()}
    rows = {}
    nominator = partial(posting_plan, catalogue.byline)
    for medium, scores in rankings.items():
        rows[medium] = {}
        for depth in DEPTHS:
            truncated = {topic: dict(list(ranks.items())[:depth]) for topic, ranks in scores.items()}
            timing = timings(catalogue, truncated, queries, depth, args.order_offset)
            measurements = [base.evaluate(catalogue, subjects, atoms, gold, truncated, union,
                                          depth=depth, joint_nominator=nominator) for atoms in queries]
            rows[medium][str(depth)] = dict(**base.summarize(measurements, groups), plan_only=timing)
    write_json(args.output, dict(
        results=rows, queries=len(queries), topic_pairs=len(set(groups)), embedding_checkpoint=model,
        previous_selector_sha256=base.pinned.search.digest(Path(base.__file__)),
        script_sha256=base.pinned.search.digest(Path(__file__)),
        manifest_sha256=base.pinned.search.digest(args.fixture / "manifest.json"),
    ))


if __name__ == "__main__":
    main()
