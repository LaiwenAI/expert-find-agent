"""Compare whole-pool and candidate-local stopping with exact citation checks."""

import argparse
from collections import defaultdict
from hashlib import sha256
import inspect
import json
from pathlib import Path
import platform
from random import Random
from statistics import mean
from time import perf_counter_ns

import numpy as np

from expert_search.constraint_verification import resolve_constraints
from experiments.credit.constraint_certificate import candidates, SCOPES, VIEWS
from experiments.followir.core17_reranking import digest, write_json


METHODS = ("exhaustive", "bundled_first", "certificate", "local_certificate")
REPEATS = 7
SCALES = (10, 100, 1000)
DENSITIES = (.03, .35)
TRIALS = 20
LARGE_TRIALS = 5


def timed(owners, read, top, strategy):
    start = perf_counter_ns()
    result = resolve_constraints(owners, read, top=top, strategy=strategy)
    return result, (perf_counter_ns() - start) / 1000


def trace(owners, read, top, strategy):
    actions = []

    def observe(expert, source):
        actions.append((expert, source))
        return read(expert, source)

    return resolve_constraints(owners, observe, top=top, strategy=strategy), actions


def compare(owners, read, top, offset, repeats=REPEATS):
    timings = {name: [] for name in METHODS}
    results = {name: resolve_constraints(owners, read, top=top, strategy=name) for name in METHODS[:2]}
    results["certificate"], original_actions = trace(owners, read, top, "certificate")
    results["local_certificate"], local_actions = trace(owners, read, top, "local_certificate")
    reference = results["exhaustive"]["ranking"]
    if any(result["ranking"] != reference for result in results.values()):
        raise AssertionError("Expert or canonical witness changed")
    if results["certificate"] != results["local_certificate"] or original_actions != local_actions:
        raise AssertionError("Candidate-local control changed the global action sequence")
    for repeat in range(repeats):
        rotation = (offset + repeat) % len(METHODS)
        for name in METHODS[rotation:] + METHODS[:rotation]:
            result, microseconds = timed(owners, read, top, name)
            if result != results[name]:
                raise AssertionError("Timed result differed from checked result")
            timings[name].append(microseconds)
    return dict(
        owners=len(owners), returned=len(reference),
        arms={name: dict(reads=results[name]["reads"], checks=results[name]["checks"],
                         median_us=float(np.median(times))) for name, times in timings.items()},
    )


def summaries(rows):
    return dict(cases=len(rows), nonempty=sum(row["returned"] > 0 for row in rows),
                mean_owners=mean(row["owners"] for row in rows),
                methods={name: dict(
                    mean_reads=mean(row["arms"][name]["reads"] for row in rows),
                    mean_checks=mean(row["arms"][name]["checks"] for row in rows),
                    p50_us=float(np.median([row["arms"][name]["median_us"] for row in rows])),
                    p95_us=float(np.quantile([row["arms"][name]["median_us"] for row in rows], .95)),
                ) for name in METHODS})


def credits(fixture):
    manifest = json.loads((fixture / "manifest.json").read_bytes())["files"]
    paths = [fixture / f"{name}.json" for name in ("sources", "queries")]
    if any(manifest[path.name] != digest(path) for path in paths):
        raise ValueError("Publisher inputs differ from their manifest")
    sources, queries = [json.loads(path.read_bytes()) for path in paths]
    selected = [q for q in queries if q["split"] == "test" and q["track"] == "flip" and q["view"] in VIEWS]
    outputs = defaultdict(list)
    for i, query in enumerate(selected):
        for scope in SCOPES:
            owners, read = candidates(sources, query, query["view"], scope)
            for top in (1, 3):
                result = compare(owners, read, top, i)
                outputs[scope, top, query["view"]].append(result)
                outputs[scope, top, "all"].append(result)
    return {f"{scope}/{top}/{view}": summaries(rows) for (scope, top, view), rows in outputs.items()}


def synthetic(size, probability, i):
    rng = Random(20261119 + i)
    owners = [(expert, tuple(tuple(rng.sample(range(6), 4)) for _ in range(4)))
              for expert in range(size)]
    scores = {(expert, source): {criterion: rng.random() < probability
                                 for criterion, order in enumerate(requirements) if source in order}
              for expert, requirements in owners for source in range(6)}
    return owners, lambda expert, source: scores[expert, source]


def run(fixture, output):
    if output.exists():
        raise FileExistsError(output)
    natural = credits(fixture)
    synthetic_report = {
        f"{size}/{probability}": summaries([
            compare(*synthetic(size, probability, i), 3, i, repeats=1 if size == max(SCALES) else REPEATS)
            for i in range(LARGE_TRIALS if size == max(SCALES) else TRIALS)
        ]) for size in SCALES for probability in DENSITIES
    }
    write_json(output, dict(
        protocol=dict(repeated_timings_per_query=REPEATS, trials_per_synthetic_condition=TRIALS,
                      largest_pool_repeats=1, largest_pool_trials=LARGE_TRIALS,
                      stress_support_probability=DENSITIES,
                      stats="Across queries, quantiles of per-query median warm CPU time."),
        machine=dict(platform=platform.platform(), python=platform.python_version()),
        inputs=dict(fixture=digest(fixture / "manifest.json"), script=digest(Path(__file__)),
                    scheduler=sha256(Path(inspect.getfile(resolve_constraints)).read_bytes()).hexdigest()),
        report=dict(publisher_declarations=natural, synthetic_stress=synthetic_report),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path)
    run(**vars(parser.parse_args()))
