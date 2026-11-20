"""Measure bounded parallel source reads for exact author-role proofs."""

import argparse
from collections import Counter
from itertools import chain, combinations
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from experiments.credit import role_column_index as columns
from experiments.followir.core17_reranking import digest, write_json


def frontier_rank(owners, caps, index, required, load_many, width):
    """Hold the earliest possible experts until their canonical proofs settle."""
    loaded = {}
    rounds = 0
    remaining = iter(owners)
    active = []

    def bounds(owner, doc):
        if doc in loaded:
            exact = loaded[doc][owner] & required
            return exact, exact
        known, values = index.get(doc, (0, {}))
        lower = values.get(owner, 0) & required
        return lower, lower | (caps[doc] & required & ~known)

    while True:
        candidates = active
        active = []
        unresolved = Counter()
        upper_roles = {}
        for owner, docs in chain(((owner, docs) for owner, docs, _ in candidates), remaining):
            limits = {doc: bounds(owner, doc) for doc in docs}
            proof = columns.bundles.best_bundle(
                [(doc, upper) for doc, (_, upper) in limits.items()], required,
            )
            if proof is None:
                continue
            active.append((owner, docs, proof))
            proved = 0
            for doc in proof:
                proved |= limits[doc][0]
            if proved != required:
                for doc in proof:
                    lower, upper = limits[doc]
                    if lower != upper:
                        unresolved[doc] += 1
                        upper_roles[doc] = upper_roles.get(doc, 0) | (upper & ~lower)
            if len(active) == columns.bundles.TOP:
                break

        if not unresolved:
            return [(owner, proof) for owner, _, proof in active], len(loaded), rounds
        selected = sorted(unresolved, key=lambda doc: (-unresolved[doc], -upper_roles[doc].bit_count(), doc))[:width]
        declarations = load_many(selected)
        if set(declarations) != set(selected):
            raise ValueError("A source response is missing or unexpected")
        for doc, declaration in declarations.items():
            known, cached = index.get(doc, (0, {}))
            if any(value & ~caps[doc] or (value & known) != cached.get(owner, 0)
                   for owner, value in declaration.items()):
                raise ValueError("Source response conflicts with verified index")
        loaded.update(declarations)
        rounds += 1


def summarize(samples):
    return {
        "requests": len(samples),
        "source_reads": sum(row[0] for row in samples),
        "source_payload_bytes": sum(row[2] for row in samples),
        "source_rounds": sum(row[1] for row in samples),
        "p95_source_rounds": float(np.quantile([row[1] for row in samples], .95)),
        "requests_with_different_source_sets": sum(row[3] for row in samples),
    }


def evaluate(owners, caps, plan, ask, raw, reference, samples):
    opened = []

    def load(doc):
        opened.append(doc)
        return json.loads(raw[doc])

    result, count = columns.partial_rank(owners, caps, plan, ask, load)
    if result != reference or count != len(opened):
        raise AssertionError("Existing serial planner did not reproduce expected results")
    serial = frozenset(opened)
    samples["owner_serial"].append((count, count, sum(len(raw[doc]) for doc in opened), False))

    for width in (1, 2, 4):
        opened.clear()

        def read(docs):
            return {doc: load(doc) for doc in docs}

        result, count, rounds = frontier_rank(owners, caps, plan, ask, read, width)
        if result != reference or count != len(opened):
            raise AssertionError("Read wave changed exact citations or repeated a source")
        samples[f"frontier_{width}"].append((
            count, rounds, sum(len(raw[doc]) for doc in opened), frozenset(opened) != serial,
        ))


def run(args):
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Use a fresh output on a Slurm compute node")
    expected = (columns.bundles.FIXTURE_SHA256 if args.publisher == "plos"
                else json.loads(args.manifest.read_bytes())["projection_sha256"])
    anchor = json.loads(args.prior.read_bytes())["publishers"][args.publisher]
    if digest(args.corpus) != expected or anchor["fixture_sha256"] != expected:
        raise ValueError("Unexpected publisher projection")
    if digest(Path(columns.__file__)) != anchor["script_sha256"]:
        raise ValueError("Unexpected index planner version")
    roles = {name: 1 << i for i, name in enumerate(sorted(columns.ROLES))}
    asks = [sum(group) for group in combinations(roles.values(), 3)]
    allocations = ("whole_cold", "column_smoothed")
    results = {}
    with TemporaryDirectory(dir=args.scratch, prefix="role-read-waves-") as directory:
        owners, labels, caps, files, _, _ = columns.inventory(
            args.corpus, Path(directory), roles, complete_only=args.publisher == "elife",
        )
        raw = {doc: path.read_bytes() for doc, path in files.items()}
        docs = sorted(files)
        ids = {doc: i for i, doc in enumerate(docs)}
        authors = {doc: [] for doc in docs}
        for owner, publications in owners:
            for doc in publications:
                authors[doc].append(owner)
        widths = {doc: (len(people) + 7) // 8 for doc, people in authors.items()}
        costs = {doc: columns.HEADER.size + widths[doc] * cap.bit_count() for doc, cap in caps.items() if cap}
        budget = anchor["byte_budgets"]["25"]
        base = columns.allocation.state(owners, labels, caps, set())
        traces, expected_ranks = columns.allocation.cold_workload(dict.fromkeys(asks, owners), base, labels)
        samples = {arm: {kind: [] for kind in ("owner_serial", "frontier_1", "frontier_2", "frontier_4")}
                   for arm in allocations}
        for fold in columns.allocation.FOLDS:
            bit = 1 << fold
            train = [ask for ask in asks if not ask & bit]
            held = [ask for ask in asks if ask & bit]
            plans = columns.designs(owners, caps, widths, costs, train, traces, budget)
            for arm in allocations:
                payload = columns.encode(ids, authors, labels, plans[arm])
                control = anchor["folds"][str(fold)]["25"][arm]
                if len(payload) != control["payload_bytes"]:
                    raise AssertionError("Encoded capacity has changed")
                plan = columns.decode(payload, docs, authors, caps)
                serial = samples[arm]["owner_serial"]
                start = len(serial)
                for ask in held:
                    evaluate(owners, caps, plan, ask, raw, expected_ranks[ask], samples[arm])
                if sum(row[0] for row in serial[start:]) != control["file_reads"]:
                    raise AssertionError("Recorded serial read count has changed")
            print(f"{args.publisher} fold {fold} verified.", flush=True)
        for arm, values in samples.items():
            results[arm] = {kind: summarize(rows) for kind, rows in values.items()}
    write_json(args.output, {
        "publisher": args.publisher, "fixture_sha256": expected, "prior_sha256": digest(args.prior),
        "byte_cap": budget, "results": results, "script_sha256": digest(Path(__file__)),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "manifest", "prior", "scratch", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--publisher", choices=("plos", "elife"), required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
