"""Compare fixed-budget expert evidence schedules on public publisher declarations."""

import argparse
from bisect import insort
from collections import defaultdict
from heapq import heappop, heappush
from itertools import combinations
import json
import os
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory
from time import perf_counter_ns

import numpy as np

from experiments.credit import minimal_evidence_bundles as bundles
from experiments.followir.core17_reranking import digest, write_json


BUDGETS = (2, 4, 8)
COVERAGES = (0, 50)
ARMS = ("prior_prefix", "static_shared", "adaptive_frontier", "adaptive_shared", "incremental_frontier")


def shortest_covers(pending, missing, remaining):
    applicable = [(source, cap & missing) for source, cap in pending if cap & missing]
    for need in range(1, min(remaining, len(applicable)) + 1):
        viable = []
        for subset in combinations(applicable, need):
            union = 0
            for _, cap in subset:
                union |= cap
            if union == missing:
                viable.append(tuple(source for source, _ in subset))
        if viable:
            return viable
    return []


def options(state, required, remaining):
    """Yield unresolved feasible owners and their shortest possible source covers."""
    occupied = 0
    optimistic_rank = 0
    for owner, known, pending in state:
        missing = required & ~known
        if not missing:
            occupied += 1
            if occupied == bundles.TOP:
                break
            optimistic_rank += 1
            continue
        covers = shortest_covers(pending, missing, remaining)
        if covers:
            yield owner, optimistic_rank, len(covers[0]), covers
            optimistic_rank += 1


def priorities(state, required, remaining, arm):
    possible = list(options(state, required, remaining))
    if not possible:
        return []
    if arm == "adaptive_frontier":
        _, _, _, covers = possible[0]
        return list(min(covers))
    evidence = defaultdict(float)
    first = {}
    for _, rank, cost, covers in possible:
        sources = set(source for cover in covers for source in cover)
        for source in sources:
            first.setdefault(source, rank)
            evidence[source] += 1 / ((rank + 1) * cost)
    if arm == "prior_prefix":
        return sorted(first, key=lambda source: (first[source], source))
    return sorted(evidence, key=lambda source: (-evidence[source], first[source], source))


def read_state(owners, observed, caps, required):
    state = []
    for owner, docs in owners:
        known = 0
        pending = []
        for source in docs:
            if source in observed:
                known |= observed[source][owner]
            elif caps[source] & required:
                pending.append((source, caps[source]))
        state.append((owner, known, pending))
    return state


def incremental(owners, observed, caps, required, budget, read):
    confirmed = []
    frontier = []
    versions = [0] * len(owners)
    known = [0] * len(owners)
    affected = defaultdict(list)
    for idx, (owner, mask, pending) in enumerate(read_state(owners, observed, caps, required)):
        known[idx] = mask
        if not required & ~mask:
            confirmed.append(idx)
        else:
            for source, _ in pending:
                affected[source].append(idx)

    def update(idx, remaining):
        versions[idx] += 1
        _, docs = owners[idx]
        covers = shortest_covers([(s, caps[s]) for s in docs if s not in observed],
                                 required & ~known[idx], remaining)
        if covers:
            cover = min(covers)
            heappush(frontier, (idx, versions[idx], len(cover), cover[0]))

    for idx in range(len(owners)):
        if required & ~known[idx]:
            update(idx, budget)
    for remaining in range(budget, 0, -1):
        while frontier:
            idx, version, cost, source = heappop(frontier)
            if cost <= remaining and version == versions[idx]:
                break
        else:
            return
        if len(confirmed) >= bundles.TOP and idx > confirmed[bundles.TOP - 1]:
            return
        values = read(source)
        for candidate in affected[source]:
            if required & ~known[candidate]:
                known[candidate] |= values[owners[candidate][0]]
                if not required & ~known[candidate]:
                    versions[candidate] += 1
                    insort(confirmed, candidate)
                else:
                    update(candidate, remaining - 1)


def search(owners, initial, caps, required, budget, arm, files):
    observed = dict(initial)
    opened = []

    def read(source):
        if source in observed:
            raise ValueError("Repeated source read")
        values = json.loads(files[source].read_bytes())
        if any(value & ~caps[source] for value in values.values()):
            raise ValueError("Publisher role ceiling excludes an observed declaration")
        observed[source] = values
        opened.append(source)
        return values

    if arm == "incremental_frontier":
        incremental(owners, observed, caps, required, budget, read)
    else:
        fixed = None
        for step in range(budget):
            if fixed is None:
                ordered = priorities(read_state(owners, observed, caps, required), required, budget - step, arm)
                if arm in ("prior_prefix", "static_shared"):
                    fixed = iter(ordered)
            source = next(fixed if fixed is not None else iter(ordered), None)
            if source is None:
                break
            read(source)

    selected = []
    for owner, docs in owners:
        known = [(source, observed[source][owner] & required) for source in docs if source in observed]
        witness = bundles.best_bundle(known, required)
        if witness is not None:
            selected.append((owner, witness))
            if len(selected) == bundles.TOP:
                break
    return selected, opened


def summarize(cases):
    return {
        "mean_reads": mean(row["reads"] for row in cases),
        "p95_us": float(np.quantile([row["us"] for row in cases], .95)),
        "queries_with_recommendations": sum(bool(row["selected"]) for row in cases),
        "fully_supported_recommendations": sum(len(row["selected"]) for row in cases),
        "recovered_reference_experts": sum(row["retained"] for row in cases),
        "reference_experts": sum(row["target"] for row in cases),
        "exact_expert_order": sum(row["same_owners"] for row in cases),
        "exact_expert_order_and_citations": sum(row["exact"] for row in cases),
    }


def run(fixture, scratch, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require a fresh output and Slurm allocation")
    if digest(fixture) != bundles.FIXTURE_SHA256:
        raise ValueError("Unexpected publisher fixture")
    papers = json.loads(fixture.read_text())["sources"]
    everyone, roles = bundles.owner_data(papers)
    names = sorted(set().union(*roles.values()))
    bits = {role: 1 << i for i, role in enumerate(names)}
    labels = {key: bundles.role_mask(values, bits) for key, values in roles.items()}
    caps = {row["doi"]: bundles.role_mask(set().union(*(a["roles"] for a in row["authors"])), bits)
            for row in papers}
    requests = [bundles.role_mask(group, bits) for group in combinations(names, 3)]
    evidence = {source: {} for source in caps}
    for (owner, source), label in labels.items():
        evidence[source][owner] = label
    report = {}
    with TemporaryDirectory(prefix="credit-article-budget-", dir=scratch) as path:
        files = {source: Path(path) / f"{i}.json" for i, source in enumerate(sorted(evidence))}
        for source, record in evidence.items():
            files[source].write_text(json.dumps(record, separators=(",", ":")))
        for name, owners in ("global", everyone), ("linked", [(p, d) for p, d in everyone if len(d) > 1]):
            gold = {ask: bundles.reference(owners, labels, ask) for ask in requests}
            report[name] = {}
            for coverage in COVERAGES:
                nested = bundles.cached(owners, labels, caps, coverage)
                known_sources = {source for rows in nested.values() for source, _, mask in rows if mask is not None}
                initial = {source: evidence[source] for source in known_sources}
                report[name][coverage] = {}
                for budget in BUDGETS:
                    rows = {arm: [] for arm in ARMS}
                    for i, ask in enumerate(requests):
                        reference = [owner for owner, _ in gold[ask]]
                        offset = i % len(ARMS)
                        for arm in ARMS[offset:] + ARMS[:offset]:
                            start = perf_counter_ns()
                            selected, opened = search(owners, initial, caps, ask, budget, arm, files)
                            elapsed = (perf_counter_ns() - start) / 1000
                            reads = len(opened)
                            assert reads <= budget
                            selected_owners = [owner for owner, _ in selected]
                            for owner, bundle in selected:
                                declared = 0
                                for source in bundle:
                                    declared |= labels[owner, source]
                                assert declared & ask == ask, "Unsupported recommendation"
                            rows[arm].append(dict(selected=selected, opened=opened, reads=reads, us=elapsed,
                                                  retained=len(set(reference) & set(selected_owners)),
                                                  target=len(reference), same_owners=reference == selected_owners,
                                                  exact=selected == gold[ask]))
                        slow, fast = (rows[arm][-1] for arm in ("adaptive_frontier", "incremental_frontier"))
                        if any(slow[k] != fast[k] for k in ("opened", "selected")):
                            raise AssertionError("Incremental frontier changed the read order or citations")
                    report[name][coverage][budget] = {arm: summarize(cases) for arm, cases in rows.items()}
                print(name, coverage, "complete", flush=True)
    write_json(output, dict(job=os.environ["SLURM_JOB_ID"], queries=len(requests), top=bundles.TOP,
                            fixture_sha256=digest(fixture), script_sha256=digest(Path(__file__)),
                            sources=len(papers), owners=len(everyone), report=report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("fixture", "scratch", "output"):
        parser.add_argument(key, type=Path)
    run(**vars(parser.parse_args()))
