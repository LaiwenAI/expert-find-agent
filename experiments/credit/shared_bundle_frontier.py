"""Share source reads across experts while certifying the same ranked citations."""

import argparse
from collections import Counter
from itertools import combinations
import json
import os
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter_ns

import numpy as np

from experiments.credit import minimal_evidence_bundles as baseline
from experiments.followir.core17_reranking import digest, write_json


ARMS = ("owner_frontier", "shared_sequential", "shared_top_five")
LEVELS = (0, 50, 100)
REPEATS = 5


def rank_shared(owners, index, required, load, *, parallel):
    """Keep the first five possible owners and shrink their optimistic proofs."""
    loaded = {}
    visited = 0
    active = []
    iterator = iter(owners)

    def optimistic(owner, declarations):
        observed = [
            (source, (loaded[source][owner] if source in loaded else
                      cap if cached is None else cached) & required)
            for source, cap, cached in declarations
        ]
        return baseline.best_bundle(observed, required)

    def pending(bundle, declarations):
        unread = {source for source, _, cached in declarations if cached is None and source not in loaded}
        return [source for source in bundle if source in unread]

    while True:
        while len(active) < baseline.TOP:
            candidate = next(iterator, None)
            if candidate is None:
                break
            owner, _ = candidate
            visited += 1
            declaration = index[owner]
            proof = optimistic(owner, declaration)
            if proof is not None:
                active.append((owner, declaration, proof))

        counts = Counter()
        caps = {}
        for _, declaration, proof in active:
            sources = pending(proof, declaration)
            if not sources:
                continue
            counts.update(sources)
            caps.update((source, (cap & required).bit_count()) for source, cap, _ in declaration if source in sources)
            if not parallel:
                break
        if not counts:
            return [(owner, proof) for owner, _, proof in active], visited

        source = min(counts, key=lambda doc: (-counts[doc], -caps[doc], doc))
        loaded[source] = load(source)
        for owner, declaration, proof in active:
            for doc, cap, _ in declaration:
                if doc == source and loaded[source][owner] & ~cap:
                    raise ValueError("Source declaration exceeds its indexed ceiling")
        updated = []
        for owner, declaration, proof in active:
            # Revealing a different source only weakens alternatives.
            if source in proof:
                proof = optimistic(owner, declaration)
            if proof is not None:
                updated.append((owner, declaration, proof))
        active = updated


def measure(files, owners, index, required, arm):
    opened = {}

    def read(source):
        if source not in opened:
            opened[source] = json.loads(files[source].read_bytes())
        return opened[source]

    started = perf_counter_ns()
    if arm == "owner_frontier":
        found, _, plan_count = baseline.rank(
            owners, index, required, lambda owner, source: read(source)[owner], "frontier")
    else:
        found, plan_count = rank_shared(owners, index, required, read, parallel=arm == "shared_top_five")
    return found, frozenset(opened), plan_count, (perf_counter_ns() - started) / 1000


def necessary_files(owners, index, labels, required, expected, opened):
    """Test each opened file against extremes when all other evidence is known."""
    def reveal_except(unread, upper):
        ranking = []
        for owner, _ in owners:
            items = []
            for source, cap, cached in index[owner]:
                if source == unread and cached is None:
                    value = cap if upper else 0
                else:
                    value = labels[owner, source]
                items.append((source, value & required))
            proof = baseline.best_bundle(items, required)
            if proof is not None:
                ranking.append((owner, proof))
                if len(ranking) == baseline.TOP:
                    break
        return ranking

    identity_order = tuple(owner for owner, _ in expected)
    sizes = tuple(len(proof) for _, proof in expected)
    required_by = {"identity_order": set(), "rank_and_size": set(), "rank_and_exact_citations": set()}
    for source in opened:
        for upper in (False, True):
            rank = reveal_except(source, upper)
            if rank == expected:
                continue
            required_by["rank_and_exact_citations"].add(source)
            if tuple(owner for owner, _ in rank) != identity_order:
                required_by["identity_order"].add(source)
                required_by["rank_and_size"].add(source)
            elif tuple(len(proof) for _, proof in rank) != sizes:
                required_by["rank_and_size"].add(source)
    return required_by


def compare(files, owners, labels, caps, asks, anchors):
    result = {}
    for percent in LEVELS:
        index = baseline.cached(owners, labels, caps, percent)
        reports = {arm: [] for arm in ARMS}
        mandatory = Counter()
        for i, query in enumerate(asks):
            measurements = {arm: [] for arm in ARMS}
            for repeat in range(REPEATS):
                rotate = (i + repeat) % len(ARMS)
                for arm in ARMS[rotate:] + ARMS[:rotate]:
                    found, opened, plans, elapsed = measure(files, owners, index, query, arm)
                    if found != anchors[query]:
                        raise AssertionError("The ordered identities or canonical citations changed")
                    measurements[arm].append((opened, plans, elapsed))
            for arm in ARMS:
                observations = measurements[arm]
                if len({row[:2] for row in observations}) != 1:
                    raise AssertionError("Planning counts varied across timing repetitions")
                opened, visited, _ = observations[0]
                reports[arm].append((len(opened), visited, median(row[2] for row in observations)))
            reference_files = measurements["owner_frontier"][0][0]
            needed = necessary_files(owners, index, labels, query, anchors[query], reference_files)
            if any(not needed["rank_and_exact_citations"].issubset(measurements[arm][0][0])
                   for arm in ARMS):
                raise AssertionError("Scheduler skipped a necessary file")
            mandatory.update({kind: len(docs) for kind, docs in needed.items()})

        result[percent] = {"necessary_files_lower_bound": dict(mandatory)}
        for arm, rows in reports.items():
            old = reports["owner_frontier"]
            result[percent][arm] = dict(
                files_read=sum(row[0] for row in rows), candidates_visited=sum(row[1] for row in rows),
                p95_us=float(np.quantile([row[2] for row in rows], .95)),
                queries_with_fewer_files=sum(a[0] < b[0] for a,b in zip(rows, old, strict=True)),
                queries_with_more_files=sum(a[0] > b[0] for a,b in zip(rows, old, strict=True)),
            )
        print(f"{len(owners)} authors, {percent}% cache schedule: verified {len(asks)} requests", flush=True)
    return result


def run(fixture, scratch, output):
    if not os.environ.get("SLURM_JOB_ID") or output.exists():
        raise ValueError("Run on a Slurm compute node and use a new output")
    if digest(fixture) != baseline.FIXTURE_SHA256:
        raise ValueError("Publisher fixture changed")
    sources = json.loads(fixture.read_bytes())["sources"]
    owners, roles = baseline.owner_data(sources)
    bits = {role: 1 << i for i, role in enumerate(sorted(set().union(*roles.values())))}
    labels = {key: baseline.role_mask(v, bits) for key, v in roles.items()}
    caps = {row["doi"]: baseline.role_mask(set().union(*(a["roles"] for a in row["authors"])), bits)
            for row in sources}
    requests = [sum(group) for group in combinations(bits.values(), 3)]
    records = {doc: {} for doc in caps}
    for (owner, doc), observation in labels.items():
        records[doc][owner] = observation
    panels = dict(all=owners, linked=[row for row in owners if len(row[1]) > 1])
    result = {}
    with TemporaryDirectory(prefix="shared-bundle-frontier-", dir=scratch) as directory:
        root = Path(directory)
        files = {doc: root / f"{i}.json" for i, doc in enumerate(records)}
        for doc, record in records.items():
            files[doc].write_text(json.dumps(record, separators=(",", ":")))
        for name, candidates in panels.items():
            expected = {ask: baseline.reference(candidates, labels, ask) for ask in requests}
            result[name] = dict(
                owners=len(candidates), report=compare(files, candidates, labels, caps, requests, expected),
            )
        byte_count = sum(file.stat().st_size for file in files.values())
    write_json(output, dict(
        panels=result, articles=len(records), materialized_bytes=byte_count,
        requests=len(requests), timing_repeats=REPEATS, reference_top=baseline.TOP,
        fixture_sha256=digest(fixture), code_sha256=digest(Path(__file__)),
        baseline_sha256=digest(Path(baseline.__file__)),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("fixture", "scratch", "output"):
        parser.add_argument(field, type=Path)
    run(**vars(parser.parse_args()))
