"""Place verified declarations before serving queries under a fixed byte budget."""

import argparse
from collections import Counter
from hashlib import sha256
from itertools import combinations
import json
import os
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory

import numpy as np

from experiments.credit import minimal_evidence_bundles as bundles
from experiments.credit.shared_bundle_frontier import measure
from experiments.followir.core17_reranking import digest, write_json


FOLDS = (0, 4, 8, 12)
LEVELS = (10, 25, 50)
HASH_SEEDS = (0, 1, 2, 3, 4, 5)


def state(owners, labels, caps, selected):
    return {
        owner: [(source, caps[source], labels[owner, source] if source in selected else None)
                for source in documents]
        for owner, documents in owners
    }


def cold_sources(owners, index, labels, request):
    viewed = set()

    def read(owner, source):
        viewed.add(source)
        return labels[owner, source]

    evidence, _, _ = bundles.rank(owners, index, request, read, "frontier")
    return viewed, evidence


def optimistic_sources(owners, caps, request):
    shown = 0
    sources = set()
    for _, documents in owners:
        choice = bundles.best_bundle([(doc, caps[doc] & request) for doc in documents], request)
        if choice is not None:
            sources.update(choice)
            shown += 1
            if shown == bundles.TOP:
                break
    return sources


def pack(order, budgets, sizes):
    selections = {}
    for budget in budgets:
        selected = set()
        remaining = budget
        for doc in order:
            if sizes[doc] <= remaining:
                selected.add(doc)
                remaining -= sizes[doc]
        selections[budget] = selected
    return selections


def placements(owners, caps, costs, training, traces):
    top_author = {}
    for position, (_, sources) in enumerate(owners):
        for source in sources:
            top_author.setdefault(source, position)
    expected = Counter()
    observed = Counter()
    for ask in training:
        expected.update(optimistic_sources(owners, caps, ask))
        observed.update(traces[ask])
    policies = {
        "author_prior": sorted(costs, key=lambda doc: (top_author[doc], costs[doc], doc)),
        "optimistic_proof_per_byte": sorted(costs, key=lambda doc: (-expected[doc] / costs[doc], costs[doc], doc)),
        "training_read_frequency_per_byte": sorted(costs, key=lambda doc: (-observed[doc] / costs[doc], costs[doc], doc)),
    }
    for seed in HASH_SEEDS:
        policies[f"hash_{seed}"] = sorted(
            costs, key=lambda doc: sha256(f"declaration-budget-v1:{seed}:{doc}".encode()).digest(),
        )
    return policies


def execute(files, owners, labels, caps, costs, asks, order_by_query, expected, policies, budgets):
    grids = {name: {budget: (selected, state(owners, labels, caps, selected))
                    for budget, selected in pack(order, budgets, costs).items()}
             for name, order in policies.items()}
    result = {}
    for budget in budgets:
        readings = {name: [] for name in policies}
        names = tuple(policies)
        for i, query in enumerate(asks):
            times = {name: [] for name in policies}
            opened = {}
            for repeat in range(3):
                offset = (i + repeat) % len(names)
                for name in names[offset:] + names[:offset]:
                    _, index = grids[name][budget]
                    found, used, _, elapsed = measure(files, order_by_query[query], index, query, "owner_frontier")
                    if found != expected[query] or (name in opened and opened[name] != used):
                        raise AssertionError("Different ordered evidence or reads")
                    opened[name] = used
                    times[name].append(elapsed)
            for name in policies:
                readings[name].append((len(opened[name]), median(times[name])))
        result[budget] = {}
        for name, observed in readings.items():
            indexed, _ = grids[name][budget]
            result[budget][name] = dict(
                source_files=len(indexed), stored_bytes=sum(costs[doc] for doc in indexed),
                disk_reads=sum(row[0] for row in observed),
                planning_and_decoding_p95_us=float(np.quantile([row[1] for row in observed], .95)),
            )
    return result


def cold_workload(order_by_query, index, labels):
    traces, references = {}, {}
    for ask, order in order_by_query.items():
        traces[ask], response = cold_sources(order, index, labels, ask)
        references[ask] = bundles.reference(order, labels, ask)
        if response != references[ask]:
            raise AssertionError("The unindexed comparison changed the proof")
    return traces, references


def run(fixture, scratch, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require a new output and a Slurm allocation")
    if digest(fixture) != bundles.FIXTURE_SHA256:
        raise ValueError("Unexpected publisher projection")
    sources = json.loads(fixture.read_bytes())["sources"]
    owners, roles = bundles.owner_data(sources)
    bits = {role: 1 << i for i, role in enumerate(sorted(set().union(*roles.values())))}
    labels = {key: bundles.role_mask(value, bits) for key, value in roles.items()}
    caps = {row["doi"]: bundles.role_mask(set().union(*(a["roles"] for a in row["authors"])), bits)
            for row in sources}
    requests = [sum(group) for group in combinations(bits.values(), 3)]
    records = {doc: {} for doc in caps}
    for (owner, doc), value in labels.items():
        records[doc][owner] = value
    with TemporaryDirectory(dir=scratch, prefix="declaration-budget-") as directory:
        root = Path(directory)
        files = {doc: root / f"{i}.json" for i, doc in enumerate(records)}
        for doc, record in records.items():
            files[doc].write_text(json.dumps(record, separators=(",", ":")))
        sizes = {doc: path.stat().st_size for doc, path in files.items()}
        total = sum(sizes.values())
        budgets = [total * level // 100 for level in LEVELS]
        opaque = state(owners, labels, caps, set())
        fixed_order = dict.fromkeys(requests, owners)
        criterion_order = {
            ask: sorted(owners, key=lambda row: (
                -sum(bool(caps[doc] & ask) for doc in row[1]),
                -max((caps[doc] & ask).bit_count() for doc in row[1]),
            ))
            for ask in requests
        }
        traces, expected = cold_workload(fixed_order, opaque, labels)
        shifted_traces, shifted_references = cold_workload(criterion_order, opaque, labels)
        results = {}
        for fold in FOLDS:
            role_bit = 1 << fold
            training = [ask for ask in requests if not ask & role_bit]
            held = [ask for ask in requests if ask & role_bit]
            policies = placements(owners, caps, sizes, training, traces)
            shifted_policies = dict(policies)
            shifted_policies["refit_read_frequency_per_byte"] = placements(
                owners, caps, sizes, training, shifted_traces,
            )["training_read_frequency_per_byte"]
            results[fold] = dict(
                calibration_queries=len(training), held_queries=len(held),
                shifted_ranking_changed=sum(shifted_references[ask] != expected[ask] for ask in held),
                zero_cache_files_opened=sum(len(traces[ask]) for ask in held),
                shifted_zero_cache_files_opened=sum(len(shifted_traces[ask]) for ask in held),
                fixed_prior=execute(files, owners, labels, caps, sizes, held, fixed_order,
                                    expected, policies, budgets),
                criterion_prior=execute(files, owners, labels, caps, sizes, held, criterion_order,
                                        shifted_references, shifted_policies, budgets),
            )
            print(f"Held role {fold}: tested {len(held)} combinations", flush=True)
    write_json(output, dict(
        fixture_sha256=digest(fixture), script_sha256=digest(Path(__file__)),
        source_measure_sha256=digest(Path(measure.__code__.co_filename)),
        roles=len(bits), sources=len(records), owner_count=len(owners), bytes=total,
        budgets=dict(zip(LEVELS, budgets, strict=True)), folds=results,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixture", "scratch", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
