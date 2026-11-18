"""Enumerate minimum source-read certificates for the small PLOS hard-constraint fixture."""

import argparse
from collections import defaultdict
import json
from itertools import combinations
from pathlib import Path

from expert_search.constraint_verification import _rank, resolve_constraints
from experiments.credit.constraint_certificate import candidates, SCOPES, VIEWS
from experiments.followir.core17_reranking import digest, write_json


def minimum_certificate(owners, full_observations, top):
    reference = _rank(owners, full_observations, top, False)
    for count in range(len(full_observations) + 1):
        for subset in combinations(full_observations, count):
            observed = {key: full_observations[key] for key in subset}
            if _rank(owners, observed, top, False) == _rank(owners, observed, top, True) == reference:
                return count
    raise AssertionError("The full observations must certify the reference")


def run(fixture, output):
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((fixture / "manifest.json").read_bytes())["files"]
    paths = [fixture / f"{name}.json" for name in ("sources", "queries")]
    if any(manifest[path.name] != digest(path) for path in paths):
        raise ValueError("Publisher inputs differ from their manifest")
    sources, queries = [json.loads(path.read_bytes()) for path in paths]
    selected = [q for q in queries if q["split"] == "test" and q["track"] == "flip" and q["view"] in VIEWS]
    report = defaultdict(lambda: dict(cases=0, minimum_reads=0, fixed_hits_minimum=0, local_hits_minimum=0))
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
                reference = _rank(owners, observations, top, False)
                if fixed["ranking"] != reference or local["ranking"] != reference:
                    raise AssertionError("A controller changed the reference output")
                record = report[f"{scope}/{top}/{query['view']}"]
                record["cases"] += 1
                record["minimum_reads"] += minimum
                record["fixed_hits_minimum"] += fixed["reads"] == minimum
                record["local_hits_minimum"] += local["reads"] == minimum
    write_json(output, dict(
        fixture_sha256=digest(fixture / "manifest.json"), script_sha256=digest(Path(__file__)),
        method="Minimum over all subsets of source reads certifying exact ranked canonical citations; oracle knows the answers",
        report=report,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path)
    run(**vars(parser.parse_args()))
