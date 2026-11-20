"""Stress-test role lookup with a replicated catalog tail and fixed head order."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from experiments.credit import lazy_role_index as lookup
from experiments.credit import stable_topic_compaction as batches
from experiments.credit import topic_role_postings as pilot
from experiments.credit.legacy_contribution_text import stable


COPIES = (1, 64, 512)
REPEATS = 7


def benchmark(catalog, db, copies, pairs):
    size = len(catalog.articles)
    offsets = list(range(0, size * copies, size))
    person = {pair: {article + offset for offset in offsets for article in articles}
              for pair, articles in catalog.person.items()}
    lookup.compile_postings(db, person)
    row_count = db.execute("SELECT count(*) FROM postings").fetchone()[0]
    scenarios = {}
    for pct in pilot.CACHE:
        complete = {article + offset for offset in offsets for article in catalog.visible[pct]}
        observations = {arm: dict(counts=Counter(), times=[]) for arm in lookup.ARMS}
        for repeat in range(REPEATS):
            for number, pair in enumerate(pairs):
                head = sorted(range(size), key=lambda i: stable("synthetic-lookup-head", pair, i))
                ranked = head + list(range(size, size * copies))
                shift = (number + repeat) % len(lookup.ARMS)
                expected = None
                for arm in lookup.ARMS[shift:] + lookup.ARMS[:shift]:
                    started = perf_counter()
                    selected, counts = lookup.select(db, ranked, pair, complete, person, arm)
                    elapsed = (perf_counter() - started) * 1e6
                    if expected is None:
                        expected = selected
                    elif selected != expected:
                        raise ValueError("Lookup routes selected different sources")
                    observations[arm]["counts"].update(counts)
                    observations[arm]["times"].append(elapsed)
        scenarios[pct] = {arm: dict(
            counts=seen["counts"], selection_p50_us=float(np.quantile(seen["times"], .5)),
            selection_p95_us=float(np.quantile(seen["times"], .95)),
        ) for arm, seen in observations.items()}
    return dict(rows=row_count, requests_per_arm_per_mask=len(pairs) * REPEATS, masks=scenarios)


def run(projection, manifest, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Require fresh output and a Slurm compute allocation")
    if pilot.digest(projection) != json.loads(manifest.read_bytes())["projection_sha256"]:
        raise ValueError("Expected the pinned projection")
    catalog = pilot.Catalog(json.loads(projection.read_bytes())["sources"],
                            SimpleNamespace(encode=lambda texts: None), None)
    pairs = sorted({pair for _, _, pair in batches.queries(catalog)})
    results = {}
    for copies in COPIES:
        with TemporaryDirectory() as scratch:
            path = Path(scratch) / "postings.sqlite"
            db = sqlite3.connect(path)
            try:
                stats = benchmark(catalog, db, copies, pairs)
            finally:
                db.close()
            results[len(catalog.articles) * copies] = dict(index_bytes=path.stat().st_size, **stats)
    pilot.write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], script_sha256=pilot.digest(Path(__file__)),
        lookup_sha256=pilot.digest(Path(lookup.__file__)), manifest_sha256=pilot.digest(manifest),
        pairs=len(pairs), sqlite=sqlite3.sqlite_version, results=results,
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("projection", "manifest", "output"):
        cli.add_argument(name, type=Path)
    run(**vars(cli.parse_args()))
