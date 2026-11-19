"""Measure real local declaration loading for compact expert citation bundles."""

import argparse
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


LEVELS = (0, 50, 75, 100)
ARMS = (*bundles.ROUTES, "resident")
REPEATS = 5


def measure(files_by_source, owners, state, labels, request, route):
    reads = {}
    if route == "resident":
        read = lambda owner, source: labels[owner, source]
    else:
        def read(owner, source):
            if source not in reads:
                reads[source] = json.loads(files_by_source[source].read_bytes())
            return reads[source][owner]

    started = perf_counter_ns()
    ranking, lookups, _ = bundles.rank(owners, state, request, read, "read_all" if route == "resident" else route)
    elapsed_us = (perf_counter_ns() - started) / 1000
    return ranking, lookups, len(reads), elapsed_us


def run(fixture, scratch, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require a new output path and Slurm")
    if digest(fixture) != bundles.FIXTURE_SHA256:
        raise ValueError("Publisher snapshot changed")
    sources = json.loads(fixture.read_bytes())["sources"]
    candidates, roles = bundles.owner_data(sources)
    owners = [(owner, docs) for owner, docs in candidates if len(docs) > 1]
    role_names = sorted(set().union(*roles.values()))
    bits = {role: 1 << i for i, role in enumerate(role_names)}
    labels = {key: bundles.role_mask(values, bits) for key, values in roles.items()}
    retained = {doi for _, docs in owners for doi in docs}
    article_records = {source: {} for source in retained}
    for (owner, source), labels_mask in labels.items():
        if source in article_records:
            article_records[source][owner] = labels_mask
    caps = {source: bundles.role_mask(set().union(*(a["roles"] for a in rec["authors"])), bits)
            for rec in sources if (source := rec["doi"]) in retained}
    requests = [bundles.role_mask(group, bits) for group in combinations(role_names, 3)]
    gold = {ask: bundles.reference(owners, labels, ask) for ask in requests}
    report = {}
    with TemporaryDirectory(prefix="credit-bundles-", dir=scratch) as tmp:
        root = Path(tmp)
        files = {source: root / f"{i}.json" for i, source in enumerate(sorted(article_records))}
        for source, record in article_records.items():
            files[source].write_text(json.dumps(record, separators=(",", ":")))
        size_bytes = sum(path.stat().st_size for path in root.iterdir())
        for coverage in LEVELS:
            cached = bundles.cached(owners, labels, caps, coverage)
            stats = {arm: [] for arm in ARMS}
            for i, request in enumerate(requests):
                timing = {arm: [] for arm in ARMS}
                for repeat in range(REPEATS):
                    offset = (i + repeat) % len(ARMS)
                    for arm in ARMS[offset:] + ARMS[:offset]:
                        ranking, lookups, loaded, micros = measure(
                            files, owners, cached, labels, request, arm)
                        if ranking != gold[request]:
                            raise AssertionError("Changed minimum publisher evidence or rank")
                        timing[arm].append(micros)
                        if repeat == REPEATS - 1:
                            stats[arm].append((lookups, loaded, np.median(timing[arm])))
            report[coverage] = {
                arm: dict(declaration_checks=sum(row[0] for row in rows),
                          files_loaded=sum(row[1] for row in rows),
                          mean_files_loaded=mean(row[1] for row in rows),
                          p95_us=float(np.quantile([row[2] for row in rows], .95)))
                for arm, rows in stats.items()
            }
            print("Coverage", coverage, "complete", flush=True)
    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], source_script_sha256=digest(Path(bundles.__file__)),
        script_sha256=digest(Path(__file__)), fixture_sha256=digest(fixture), reports=report,
        retained_owners=len(owners), source_files=len(retained), source_bytes=size_bytes,
        queries=len(requests), repeats=REPEATS,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixture", "scratch", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
