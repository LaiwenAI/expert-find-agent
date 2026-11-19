"""Measure the exact hindsight source-read requirement for ranked citations."""

import argparse
from collections import defaultdict
import inspect
import json
import os
from pathlib import Path
from random import Random
from time import perf_counter_ns

from expert_search.constraint_verification import _rank, resolve_constraints
from experiments.credit.constraint_certificate import candidates, SCOPES, VIEWS
from experiments.followir.core17_reranking import digest, write_json


METHODS = ("short_circuit", "bundled_first", "shortest_first", "local_certificate", "refutation_first")


def minimum_certificate(owners, full_observations, top):
    """Count the least reads needed to certify the realized ranking and citations.

    Reads are charged per expert and source. For a returned expert, every
    canonical positive witness and the sources preceding it must be read.
    For a skipped expert ahead of the cutoff, exhausting any fully negative
    criterion suffices. These requirements are independent across experts.
    """
    selected = set()
    confirmed = 0
    for expert, requirements in owners:
        prefixes = set()
        failures = []
        for criterion, witnesses in enumerate(requirements):
            for i, source in enumerate(witnesses):
                if full_observations[expert, source].get(criterion) is True:
                    prefixes.update(witnesses[:i + 1])
                    break
            else:
                failures.append(set(witnesses))
        if failures:
            required = min(failures, key=len)
        else:
            required = prefixes
            confirmed += 1
        selected.update((expert, source) for source in required)
        if confirmed == top:
            break
    observed = {key: full_observations[key] for key in selected}
    reference = _rank(owners, full_observations, top, False)
    if not (_rank(owners, observed, top, False) == _rank(owners, observed, top, True) == reference):
        raise AssertionError("Constructed reads did not certify the reference")
    return len(selected)


def stress():
    reports = {}
    for width in ("balanced", "one_narrow"):
        for probability in (.03, .35):
            for size in (100, 1000):
                minimum_reads = 0
                totals = {method: dict(reads=0, nanoseconds=0, hits_minimum=0) for method in METHODS}
                for trial in range(20):
                    rng = Random(20261119 + trial)
                    widths = (4, 4, 4, 4) if width == "balanced" else (4, 4, 4, 1)
                    owners = [(expert, tuple(tuple(rng.sample(range(6), count))
                                            for count in widths)) for expert in range(size)]
                    observations = {(expert, source): {i: rng.random() < probability
                                                        for i, sources in enumerate(requirements) if source in sources}
                                    for expert, requirements in owners
                                    for source in sorted({src for sources in requirements for src in sources})}
                    minimum = minimum_certificate(owners, observations, 3)
                    reference = _rank(owners, observations, 3, False)
                    minimum_reads += minimum
                    offset = trial % len(METHODS)
                    for method in METHODS[offset:] + METHODS[:offset]:
                        start = perf_counter_ns()
                        result = resolve_constraints(owners, lambda e, s: observations[e, s], top=3, strategy=method)
                        elapsed = perf_counter_ns() - start
                        if result["ranking"] != reference or result["reads"] < minimum:
                            raise AssertionError("Controller disagrees with the exact reference")
                        row = totals[method]
                        row["reads"] += result["reads"]
                        row["nanoseconds"] += elapsed
                        row["hits_minimum"] += result["reads"] == minimum
                reports[f"{width}/{probability}/{size}"] = dict(
                    cases=20, minimum_reads=minimum_reads, arms=totals,
                )
    return reports


def run(fixture, output, stress_tests=False):
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((fixture / "manifest.json").read_bytes())["files"]
    paths = [fixture / f"{name}.json" for name in ("sources", "queries")]
    if any(manifest[path.name] != digest(path) for path in paths):
        raise ValueError("Publisher inputs differ from their manifest")
    sources, queries = [json.loads(path.read_bytes()) for path in paths]
    selected = [q for q in queries if q["split"] == "test" and q["track"] == "flip" and q["view"] in VIEWS]
    report = defaultdict(lambda: dict(
        cases=0, minimum_reads=0, fixed_hits_minimum=0, local_hits_minimum=0,
        refutation_hits_minimum=0, refutation_reads=0, shortest_hits_minimum=0, shortest_reads=0,
    ))
    for query in selected:
        for scope in SCOPES:
            owners, read = candidates(sources, query, query["view"], scope)
            keys = set((expert, source) for expert, requirements in owners
                       for witnesses in requirements for source in witnesses)
            observations = {key: read(*key) for key in sorted(keys)}
            for top in (1, 3):
                minimum = minimum_certificate(owners, observations, top)
                fixed = resolve_constraints(owners, read, top=top, strategy="bundled_first")
                local = resolve_constraints(owners, read, top=top, strategy="local_certificate")
                refutation = resolve_constraints(owners, read, top=top, strategy="refutation_first")
                shortest = resolve_constraints(owners, read, top=top, strategy="shortest_first")
                reference = _rank(owners, observations, top, False)
                if any(arm["ranking"] != reference or arm["reads"] < minimum
                       for arm in (fixed, local, refutation, shortest)):
                    raise AssertionError("A controller changed the reference output")
                record = report[f"{scope}/{top}/{query['view']}"]
                record["cases"] += 1
                record["minimum_reads"] += minimum
                record["fixed_hits_minimum"] += fixed["reads"] == minimum
                record["local_hits_minimum"] += local["reads"] == minimum
                record["refutation_hits_minimum"] += refutation["reads"] == minimum
                record["refutation_reads"] += refutation["reads"]
                record["shortest_hits_minimum"] += shortest["reads"] == minimum
                record["shortest_reads"] += shortest["reads"]
    result = dict(
        fixture_sha256=digest(fixture / "manifest.json"), script_sha256=digest(Path(__file__)),
        scheduler_sha256=digest(Path(inspect.getfile(resolve_constraints))),
        job=os.environ.get("SLURM_JOB_ID"),
        method="Exact hindsight minimum for certifying ranked canonical citations under per-expert source reads",
        report=report,
    )
    if stress_tests:
        result["stress"] = stress()
    write_json(output, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--stress", dest="stress_tests", action="store_true")
    run(**vars(parser.parse_args()))
