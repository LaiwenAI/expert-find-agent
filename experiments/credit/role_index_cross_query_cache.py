"""Measure cross-query reuse of immutable verified declarations at equal byte caps."""

import argparse
from collections import OrderedDict
from hashlib import sha256
from itertools import combinations
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from experiments.credit import role_column_index as columns
from experiments.credit.role_index_read_waves import frontier_rank
from experiments.followir.core17_reranking import digest, write_json

# Each tuple is (offline allocation, share of the byte budget reserved for it).
ARMS = {
    "cold_static": ("whole_cold", 2),
    "column_static": ("column_smoothed", 2),
    "lru_only": ("whole_cold", 0),
    "cold_plus_lru": ("whole_cold", 1),
    "column_plus_lru": ("column_smoothed", 1),
}


def replay(owners, caps, reference, static, raw, costs, capacity, queries):
    cache = OrderedDict()
    used = peak = 0
    reads = []
    payload = rounds = 0
    for ask in queries:
        fetched = {}
        index = {**static, **cache}

        def load_many(docs):
            declarations = {doc: json.loads(raw[doc]) for doc in docs}
            if declarations.keys() & fetched.keys():
                raise AssertionError("Planner reloaded a declaration")
            fetched.update(declarations)
            return declarations

        answer, count, waves = frontier_rank(owners, caps, index, ask, load_many, 4)
        if answer != reference[ask] or count != len(fetched):
            raise AssertionError("Changed exact ranked evidence or loaded a source twice")
        if set(fetched) & cache.keys():
            raise AssertionError("Planner fetched a fully cached declaration")
        # Refresh cached sources actually cited, and admit new source reads.
        for _, citations in answer:
            for doc in citations:
                if doc in cache:
                    cache.move_to_end(doc)
        for doc in fetched:
            cost = costs[doc]
            if cost > capacity:
                continue
            while used + cost > capacity:
                dropped, _ = cache.popitem(last=False)
                used -= costs[dropped]
            cache[doc] = caps[doc], fetched[doc]
            used += cost
            peak = max(peak, used)
        reads.append(count)
        rounds += waves
        payload += sum(len(raw[doc]) for doc in fetched)
    return dict(requests=len(queries), source_reads=sum(reads), source_rounds=rounds,
                read_payload_bytes=payload, peak_cache_bytes=peak,
                no_source_read=sum(n == 0 for n in reads), p95_source_reads=float(np.quantile(reads, .95)),
                first_request_reads=reads[0])


def run(args):
    if not os.environ.get("SLURM_JOB_ID") or args.output.exists():
        raise ValueError("Use a fresh output on a Slurm compute node")
    anchor = json.loads(args.prior.read_bytes())["publishers"][args.publisher]
    if digest(args.corpus) != anchor["fixture_sha256"] or digest(Path(columns.__file__)) != anchor["script_sha256"]:
        raise ValueError("Changed corpus or exact index implementation")
    roles = {role: 1 << i for i, role in enumerate(sorted(columns.ROLES))}
    asks = [sum(group) for group in combinations(roles.values(), 3)]
    budget = anchor["byte_budgets"]["25"]
    records = []
    with TemporaryDirectory(dir=args.scratch, prefix="role-cache-") as directory:
        owners, labels, caps, files, _, _ = columns.inventory(
            args.corpus, Path(directory), roles, complete_only=args.publisher == "elife",
        )
        raw = {doc: file.read_bytes() for doc, file in files.items()}
        docs = sorted(files)
        ids = {doc: i for i, doc in enumerate(docs)}
        authors = {doc: [] for doc in docs}
        for owner, publications in owners:
            for doc in publications:
                authors[doc].append(owner)
        widths = {doc: (len(people) + 7) // 8 for doc, people in authors.items()}
        costs = {doc: columns.HEADER.size + widths[doc] * cap.bit_count() for doc, cap in caps.items() if cap}
        empty = columns.allocation.state(owners, labels, caps, set())
        traces, expected = columns.allocation.cold_workload(dict.fromkeys(asks, owners), empty, labels)
        for fold in columns.allocation.FOLDS:
            train = [ask for ask in asks if not ask & (1 << fold)]
            held = [ask for ask in asks if ask & (1 << fold)]
            plans = {reserved: columns.designs(owners, caps, widths, costs, train, traces, reserved)
                     for reserved in (budget // 2, budget)}
            for name, (method, fraction) in ARMS.items():
                reserved = budget * fraction // 2
                selected = plans[reserved][method] if reserved else {}
                packed = columns.encode(ids, authors, labels, selected)
                if len(packed) > reserved:
                    raise AssertionError("Offline index exceeded its share")
                static = columns.decode(packed, docs, authors, caps)
                for seed in (20261121, 20261122):
                    order = sorted(held, key=lambda ask: sha256(f"{seed}:{fold}:{ask}".encode()).digest())
                    outcome = replay(owners, caps, expected, static, raw, costs, budget - reserved, order)
                    if len(packed) + outcome["peak_cache_bytes"] > budget:
                        raise AssertionError("Combined placement exceeded the byte budget")
                    records.append(dict(arm=name, fold=fold, seed=seed, static_bytes=len(packed), **outcome))
    write_json(args.output, dict(publisher=args.publisher, byte_budget=budget, fixture_sha256=digest(args.corpus),
                                 prior_sha256=digest(args.prior), script_sha256=digest(Path(__file__)), results=records))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "prior", "scratch", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--publisher", choices=("plos", "elife"), required=True)
    run(parser.parse_args())
