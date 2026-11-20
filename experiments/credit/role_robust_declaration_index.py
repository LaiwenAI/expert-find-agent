"""Reserve author-order coverage and calibrate on role-stratified read regret."""

import argparse
from collections import Counter
from itertools import combinations
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from experiments.credit import budgeted_declaration_index as allocation
from experiments.credit import minimal_evidence_bundles as bundles
from experiments.credit.declaration_index_publisher_transfer import inventory
from experiments.credit.plos_role_inventory import ROLES
from experiments.followir.core17_reranking import digest, write_json


RESERVES = (0, 25, 50, 75, 100)


def mixed_orders(author_order, cold_order, costs, budget):
    options = {}
    for percent in RESERVES:
        prefix = allocation.pack(author_order, [budget * percent // 100], costs)[budget * percent // 100]
        order = [doc for doc in author_order if doc in prefix]
        order.extend(doc for doc in cold_order if doc not in prefix)
        options[f"reserve_{percent}"] = order if percent < 100 else author_order
    return options


def calibrate(owners, labels, limits, costs, train, roles, options, byte_limit):
    counts = {}
    groups = {role: [i for i, query in enumerate(train) if query & role] for role in roles}
    for name, order in options.items():
        selected = allocation.pack(order, [byte_limit], costs)[byte_limit]
        cached = allocation.state(owners, labels, limits, selected)
        counts[name] = np.asarray([
            len(allocation.cold_sources(owners, cached, labels, query)[0]) for query in train
        ])
    baseline = counts["reserve_100"]
    estimates = {}
    for name, values in counts.items():
        regret = values - baseline
        estimates[name] = dict(
            total=int(values.sum()), worst_role_mean_excess=max(float(regret[idx].mean()) for idx in groups.values()),
            p95_query_excess=float(np.quantile(regret, .95)),
        )
    order = list(options)
    chosen = {
        "mean": min(order, key=lambda n: estimates[n]["total"]),
        "role_guard": min(order, key=lambda n: (
            estimates[n]["worst_role_mean_excess"], estimates[n]["total"],
        )),
        "query_tail_guard": min(order, key=lambda n: (
            estimates[n]["p95_query_excess"], estimates[n]["total"],
        )),
    }
    return chosen, estimates


def run(corpus, manifest, scratch, output, publisher):
    if output.exists() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Use a fresh output on a Slurm compute node")
    expected = (bundles.FIXTURE_SHA256 if publisher == "plos"
                else json.loads(manifest.read_bytes())["projection_sha256"])
    if digest(corpus) != expected:
        raise ValueError("Unexpected publisher projection")
    bits = tuple(1 << i for i in range(len(ROLES)))
    roles = dict(zip(sorted(ROLES), bits, strict=True))
    requests = [sum(group) for group in combinations(bits, 3)]
    results = {}
    with TemporaryDirectory(dir=scratch, prefix="robust-index-") as directory:
        owners, labels, caps, files, costs, _ = inventory(
            corpus, Path(directory), roles, complete_only=publisher == "elife",
        )
        budgets = {level: sum(costs.values()) * level // 100 for level in allocation.LEVELS}
        order = dict.fromkeys(requests, owners)
        traces, reference = allocation.cold_workload(
            order, allocation.state(owners, labels, caps, set()), labels,
        )
        author_prior = {}
        for position, (_, sources) in enumerate(owners):
            for doc in sources:
                author_prior.setdefault(doc, position)
        authors = sorted(costs, key=lambda doc: (author_prior[doc], costs[doc], doc))

        for fold, excluded in enumerate(bits):
            train = [ask for ask in requests if not ask & excluded]
            held = [ask for ask in requests if ask & excluded]
            cold_counts = Counter()
            for query in train:
                cold_counts.update(traces[query])
            cold = sorted(costs, key=lambda doc: (-cold_counts[doc] / costs[doc], costs[doc], doc))
            fold_results = {}
            for level, budget in budgets.items():
                variants = mixed_orders(authors, cold, costs, budget)
                picked, development = calibrate(owners, labels, caps, costs, train,
                                                 [bit for bit in bits if bit != excluded], variants, budget)
                # Every mixed arm must return the same exact authoritative result.
                measured = allocation.execute(
                    files, owners, labels, caps, costs, held, order, reference, variants, [budget],
                )[budget]
                fold_results[level] = dict(chosen=picked, development=development, held=measured)
            results[fold] = fold_results
            print(f"{publisher} held role {fold}: {len(held)} exact queries verified", flush=True)
    write_json(output, dict(
        publisher=publisher, roles=list(roles), sources=len(files), authors=len(owners),
        byte_budgets=budgets, held_queries_per_fold=len(held), folds=results,
        corpus_sha256=digest(corpus), script_sha256=digest(Path(__file__)),
        allocation_sha256=digest(Path(allocation.__file__)),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "manifest", "scratch", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--publisher", required=True, choices=("plos", "elife"))
    run(**vars(parser.parse_args()))
