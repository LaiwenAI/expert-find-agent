"""Certify compact expert citations with article-level role bounds."""

import argparse
from hashlib import sha256
from itertools import combinations
import json
import os
from pathlib import Path
from statistics import mean
from time import perf_counter_ns

import numpy as np

from experiments.credit.role_evidence_postings import owner_data
from experiments.followir.core17_reranking import digest, write_json


FIXTURE_SHA256 = "74168d853ad21c795269c904fb7f8081cdf5a3e91ce7385ef8683495fd2ca7d5"
COVERAGE = (0, 50, 75, 100)
ROUTES = ("read_all", "prefix", "frontier")
REPEATS = 5
TOP = 5


def role_mask(labels, bits):
    return sum(bits[role] for role in labels)


def best_bundle(observations, required):
    """Minimize the number of citations, then their source order."""
    if not observations:
        return None
    total = 0
    for source, possible in observations:
        if possible == required:
            return (source,)
        total |= possible
    if total != required:
        return None
    best = {0: ()}
    for source, possible in observations:
        if not possible:
            continue
        for known, bundle in list(best.items()):
            merged = known | possible
            candidate = (*bundle, source)
            current = best.get(merged)
            if current is None or (len(candidate), candidate) < (len(current), current):
                best[merged] = candidate
    return best[required]


def owner_bundle(declarations, required, fetch, route):
    # Source cap is a publisher-record ceiling, never evidence of personal credit.
    state = {source: observed for source, _, observed in declarations}
    cap = {source: possible & required for source, possible, _ in declarations}
    pending = {source for source, observed in state.items() if observed is None and cap[source]}
    reads = 0

    def read(source):
        nonlocal reads
        value = fetch(source) & required
        if value & ~cap[source]:
            raise ValueError("Cached article ceiling omits an observed role")
        state[source] = value
        pending.remove(source)
        reads += 1

    if route == "read_all":
        ceiling = 0
        for source, value in state.items():
            ceiling |= cap[source] if value is None else value & required
        if ceiling != required:
            return None, reads
        for source in sorted(pending):
            read(source)
        return best_bundle([(source, (value or 0) & required) for source, value in state.items()], required), reads

    while True:
        optimistic = best_bundle([(source, cap[source] if value is None else value & required)
                                  for source, value in state.items()], required)
        if optimistic is None:
            return None, reads
        if not pending:
            return optimistic, reads
        if all(source not in pending for source in optimistic):
            return optimistic, reads
        source = min(pending) if route == "prefix" else min(
            (source for source in optimistic if source in pending),
            key=lambda source: (-cap[source].bit_count(), source))
        read(source)


def rank(owners, observations, required, read, route):
    results = []
    reads = 0
    visited = 0
    for owner, _ in owners:
        declarations = observations[owner]
        visited += 1
        bundle, cost = owner_bundle(declarations, required, lambda source: read(owner, source), route)
        reads += cost
        if bundle is not None:
            results.append((owner, bundle))
            if len(results) == TOP:
                break
    return results, reads, visited


def reference(owners, labels, required):
    result = []
    for owner, documents in owners:
        found = False
        for size in range(1, len(documents) + 1):
            for bundle in combinations(documents, size):
                union = 0
                for source in bundle:
                    union |= labels[owner, source]
                if union & required == required:
                    result.append((owner, bundle))
                    found = True
                    break
            if found:
                break
        if len(result) == TOP:
            break
    return result


def cached(owners, labels, caps, coverage):
    def visible(source):
        hashed = sha256(source.encode()).digest()
        return int.from_bytes(hashed[:8], "big") % 100 < coverage

    return {
        owner: [(source, caps[source], labels[owner, source] if visible(source) else None)
                for source in documents]
        for owner, documents in owners
    }


def replay(owners, labels, caps, requests, expected):
    result = {}
    for coverage in COVERAGE:
        state = cached(owners, labels, caps, coverage)
        observed = sum(value is not None for docs in state.values() for _, _, value in docs)
        arms = {method: [] for method in ROUTES}
        for i, required in enumerate(requests):
            times = {method: [] for method in ROUTES}
            for repeat in range(REPEATS):
                offset = (i + repeat) % len(ROUTES)
                for method in ROUTES[offset:] + ROUTES[:offset]:
                    start = perf_counter_ns()
                    ranking, reads, visited = rank(owners, state, required, lambda a, b: labels[a, b], method)
                    times[method].append((perf_counter_ns() - start) / 1000)
                    if ranking != expected[required]:
                        raise AssertionError("Rank or minimum witness bundle changed")
                    if repeat == REPEATS - 1:
                        arms[method].append((reads, visited, np.median(times[method])))
        result[coverage] = dict(indexed_declarations=observed, arms={
            method: dict(total_reads=sum(row[0] for row in rows),
                         mean_reads=mean(row[0] for row in rows), mean_visited=mean(row[1] for row in rows),
                         p95_us=float(np.quantile([row[2] for row in rows], .95)))
            for method, rows in arms.items()
        })
    return result


def run(fixture, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require a new output path and a Slurm allocation")
    if digest(fixture) != FIXTURE_SHA256:
        raise ValueError("Publisher snapshot changed")
    sources = json.loads(fixture.read_bytes())["sources"]
    owners, roles = owner_data(sources)
    kinds = sorted(set().union(*roles.values()))
    bits = {role: 1 << i for i, role in enumerate(kinds)}
    labels = {key: role_mask(values, bits) for key, values in roles.items()}
    caps = {row["doi"]: role_mask(set().union(*(a["roles"] for a in row["authors"])), bits) for row in sources}
    requests = [role_mask(group, bits) for group in combinations(kinds, 3)]
    panels = {"all": owners, "linked": [(owner, docs) for owner, docs in owners if len(docs) > 1]}
    report = {}
    for panel, candidates in panels.items():
        expected = {ask: reference(candidates, labels, ask) for ask in requests}
        picks = [bundle for ranking in expected.values() for _, bundle in ranking]
        documents_by_owner = dict(candidates)
        first_witness_counts = []
        role_bits = list(bits.values())
        for ask, ranking in expected.items():
            for owner, _ in ranking:
                witnesses = {next(doc for doc in documents_by_owner[owner] if labels[owner, doc] & role)
                             for role in role_bits if ask & role}
                first_witness_counts.append(len(witnesses))
        report[panel] = dict(owners=len(candidates), requests=len(requests), returned=len(picks),
                             complementary_bundles=sum(len(b) > 1 for b in picks),
                             minimum_citations=sum(map(len, picks)), first_per_role_citations=sum(first_witness_counts),
                             positive_requests=sum(bool(value) for value in expected.values()),
                             report=replay(candidates, labels, caps, requests, expected))
        print(panel, "complete", flush=True)
    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], fixture_sha256=digest(fixture), script_sha256=digest(Path(__file__)),
        reports=report, dataset_articles=len(sources), timing_repeats=REPEATS, top=TOP,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path)
    run(**vars(parser.parse_args()))
