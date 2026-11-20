"""Materialize verified role columns under the same byte cap as whole sources."""

import argparse
from collections import Counter
from heapq import heappop, heappush
from itertools import combinations
import json
import os
from pathlib import Path
from statistics import median
from struct import Struct
from tempfile import TemporaryDirectory
from time import perf_counter_ns

import numpy as np

from experiments.credit import budgeted_declaration_index as allocation
from experiments.credit import minimal_evidence_bundles as bundles
from experiments.credit.declaration_index_publisher_transfer import inventory
from experiments.credit.plos_role_inventory import ROLES
from experiments.followir.core17_reranking import digest, write_json


HEADER = Struct("<HH")
ARMS = ("whole_author", "whole_cold", "column_frequency", "column_smoothed", "column_bundles")
REPEATS = 3


def selected_bits(mask):
    while mask:
        role = mask & -mask
        yield role
        mask ^= role


def encode(ids, authors, labels, masks):
    result = bytearray()
    for doc, mask in sorted(masks.items()):
        if not mask:
            continue
        result.extend(HEADER.pack(ids[doc], mask))
        people = authors[doc]
        width = (len(people) + 7) // 8
        for role in selected_bits(mask):
            owners = sum(1 << i for i, person in enumerate(people) if labels[person, doc] & role)
            result.extend(owners.to_bytes(width, "little"))
    return bytes(result)


def decode(payload, docs, authors, caps):
    result = {}
    position = 0
    while position < len(payload):
        identifier, mask = HEADER.unpack_from(payload, position)
        position += HEADER.size
        doc = docs[identifier]
        if not mask or mask & ~caps[doc] or doc in result:
            raise ValueError("Invalid materialized role mask")
        people = authors[doc]
        width = (len(people) + 7) // 8
        masks = dict.fromkeys(people, 0)
        for role in selected_bits(mask):
            if position + width > len(payload):
                raise ValueError("Incomplete role column")
            column = int.from_bytes(payload[position:position + width], "little")
            position += width
            if column >> len(people):
                raise ValueError("Out-of-range indexed author")
            for i, person in enumerate(people):
                if column & (1 << i):
                    masks[person] |= role
        result[doc] = mask, masks
    return result


def materialize_columns(rank, budget, widths):
    stored = {}
    for doc, role in rank:
        cost = widths[doc] + (0 if doc in stored else HEADER.size)
        if cost <= budget:
            stored[doc] = stored.get(doc, 0) | role
            budget -= cost
    return stored


def materialize_bundles(observed, budget, widths):
    """Pay for columns together when they complete a source-level query mask."""
    stored = {}
    queue = []

    def enqueue(doc, remaining):
        seen = stored.get(doc, 0)
        entries = observed[doc]
        outstanding = {ask & ~seen for ask in entries if ask & ~seen}
        best = None
        for addition in outstanding:
            cost = widths[doc] * addition.bit_count() + (0 if seen else HEADER.size)
            if cost > remaining:
                continue
            gain = sum(count for ask, count in entries.items()
                       if ask & ~seen and not (ask & ~(seen | addition)))
            key = (-gain / cost, cost, doc, addition)
            if best is None or key < best:
                best = key
        if best is not None:
            heappush(queue, best)

    for doc in observed:
        enqueue(doc, budget)
    while queue:
        _, cost, doc, addition = heappop(queue)
        if cost > budget:
            enqueue(doc, budget)
            continue
        stored[doc] = stored.get(doc, 0) | addition
        budget -= cost
        enqueue(doc, budget)
    return stored


def partial_rank(owners, caps, index, required, load):
    result = []
    loaded = {}
    for owner, docs in owners:
        lower = {}
        upper = {}
        for doc in docs:
            if doc in loaded:
                lower[doc] = upper[doc] = loaded[doc][owner] & required
            elif doc in index:
                known, values = index[doc]
                lower[doc] = values[owner] & required
                upper[doc] = lower[doc] | (caps[doc] & required & ~known)
            else:
                lower[doc] = 0
                upper[doc] = caps[doc] & required

        while (possible := bundles.best_bundle(upper.items(), required)) is not None:
            evidence = 0
            for doc in possible:
                evidence |= lower[doc]
            if evidence == required:
                result.append((owner, possible))
                break
            doc = min((doc for doc in possible if upper[doc] != lower[doc]),
                      key=lambda key: (-upper[key].bit_count(), key))
            resolved = load(doc)
            loaded[doc] = resolved
            now = resolved[owner] & required
            if now & ~upper[doc] or lower[doc] & ~now:
                raise ValueError("Fetched declaration conflicts with cached bounds")
            upper[doc] = lower[doc] = now
        if len(result) == bundles.TOP:
            break
    return result, len(loaded)


def designs(owners, caps, widths, costs, train, traces, budget):
    earliest = {}
    for position, (_, docs) in enumerate(owners):
        for doc in docs:
            earliest.setdefault(doc, position)
    total = Counter()
    by_role = Counter()
    masks = {}
    for ask in train:
        opens = traces[ask]
        total.update(opens)
        for doc in opens:
            overlap = caps[doc] & ask
            by_role.update((doc, role) for role in selected_bits(overlap))
            if overlap:
                masks.setdefault(doc, Counter())[overlap] += 1
    whole_author = sorted(costs, key=lambda doc: (earliest[doc], costs[doc], doc))
    whole_cold = sorted(costs, key=lambda doc: (-total[doc] / costs[doc], costs[doc], doc))
    candidates = [(doc, role) for doc in costs for role in selected_bits(caps[doc])]
    rank = sorted(candidates, key=lambda pair: (
        -by_role[pair] / widths[pair[0]], costs[pair[0]], *pair,
    ))
    smoothed = sorted(candidates, key=lambda pair: (
        -(3 * by_role[pair] + total[pair[0]]) / widths[pair[0]], costs[pair[0]], *pair,
    ))
    planned = {
        "whole_author": allocation.pack(whole_author, [budget], costs)[budget],
        "whole_cold": allocation.pack(whole_cold, [budget], costs)[budget],
    }
    return dict(
        **{name: {doc: caps[doc] for doc in chosen} for name, chosen in planned.items()},
        column_frequency=materialize_columns(rank, budget, widths),
        column_smoothed=materialize_columns(smoothed, budget, widths),
        column_bundles=materialize_bundles(masks, budget, widths),
    )


def measure(owners, files, caps, plans, held, reference, baseline_reads):
    reads = {name: [] for name in ARMS}
    for i, ask in enumerate(held):
        times = {name: [] for name in ARMS}
        counts = {}
        for repeat in range(REPEATS):
            offset = (i + repeat) % len(ARMS)
            for name in ARMS[offset:] + ARMS[:offset]:
                loaded_bytes = 0

                def load(doc):
                    nonlocal loaded_bytes
                    payload = files[doc].read_bytes()
                    loaded_bytes += len(payload)
                    return json.loads(payload)

                began = perf_counter_ns()
                result, opened = partial_rank(owners, caps, plans[name], ask, load)
                times[name].append((perf_counter_ns() - began) / 1000)
                volume = opened, loaded_bytes
                if result != reference[ask] or (name in counts and counts[name] != volume):
                    raise AssertionError("Exact ranked citation output or repeated read volume changed")
                counts[name] = volume
        for name in ARMS:
            reads[name].append((*counts[name], median(times[name]), baseline_reads[ask]))
    return {name: dict(file_reads=sum(row[0] for row in values),
                       decoded_bytes=sum(row[1] for row in values),
                       p95_decoded_bytes=float(np.quantile([row[1] for row in values], .95)),
                       p95_us=float(np.quantile([row[2] for row in values], .95)),
                       zero_cache_file_reads=sum(row[3] for row in values))
            for name, values in reads.items()}


def run(corpus, manifest, scratch, output, publisher):
    if output.exists() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Use a new output on a Slurm compute node")
    expected = (bundles.FIXTURE_SHA256 if publisher == "plos"
                else json.loads(manifest.read_bytes())["projection_sha256"])
    if digest(corpus) != expected:
        raise ValueError("Unexpected publisher projection")
    roles = {name: 1 << i for i, name in enumerate(sorted(ROLES))}
    requests = [sum(group) for group in combinations(roles.values(), 3)]
    result = {}
    with TemporaryDirectory(dir=scratch, prefix="role-column-") as directory:
        owners, labels, caps, files, _, _ = inventory(
            corpus, Path(directory), roles, complete_only=publisher == "elife",
        )
        doc_order = sorted(files)
        ids = {doc: i for i, doc in enumerate(doc_order)}
        authors = {doc: [] for doc in files}
        for owner, docs in owners:
            for doc in docs:
                authors[doc].append(owner)
        widths = {doc: (len(people) + 7) // 8 for doc, people in authors.items()}
        costs = {doc: HEADER.size + widths[doc] * cap.bit_count()
                 for doc, cap in caps.items() if cap}
        total = sum(costs.values())
        if len(encode(ids, authors, labels, caps)) != total:
            raise AssertionError("Measured packed budget differs from the complete index")
        budgets = {level: total * level // 100 for level in allocation.LEVELS}
        base = allocation.state(owners, labels, caps, set())
        traces, reference = allocation.cold_workload(dict.fromkeys(requests, owners), base, labels)
        cold = {ask: len(sources) for ask, sources in traces.items()}

        for fold in allocation.FOLDS:
            bit = 1 << fold
            train = [ask for ask in requests if not ask & bit]
            held = [ask for ask in requests if ask & bit]
            result[fold] = {}
            for level, budget in budgets.items():
                selected = designs(owners, caps, widths, costs, train, traces, budget)
                sizes, compiled = {}, {}
                for name, masks in selected.items():
                    payload = encode(ids, authors, labels, masks)
                    sizes[name] = len(payload)
                    if sizes[name] > budget:
                        raise AssertionError("Encoded index exceeded the byte limit")
                    compiled[name] = decode(payload, doc_order, authors, caps)
                report = measure(owners, files, caps, compiled, held, reference, cold)
                for name, row in report.items():
                    row.update(payload_bytes=sizes[name], articles=len(selected[name]),
                               columns=sum(m.bit_count() for m in selected[name].values()),
                               held_role_columns=sum(bool(m & bit) for m in selected[name].values()))
                result[fold][level] = report
            print(f"{publisher}: held role {fold} verified", flush=True)
    write_json(output, dict(
        folds=result, publisher=publisher, sources=len(files), owners=len(owners),
        held_queries_per_fold=len(held), complete_index_bytes=total, byte_budgets=budgets,
        fixture_sha256=digest(corpus), script_sha256=digest(Path(__file__)),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "manifest", "scratch", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--publisher", choices=("plos", "elife"), required=True)
    run(**vars(parser.parse_args()))
