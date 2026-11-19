"""Replay conservative role postings over pinned PLOS publisher declarations."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import inspect
from itertools import combinations
import json
from pathlib import Path
from statistics import mean
from time import perf_counter_ns

import numpy as np

from expert_search.constraint_verification import resolve_constraints
from expert_search.role_evidence_index import RoleEvidenceIndex
from experiments.credit.cross_source_identity import load
from experiments.followir.core17_reranking import digest, write_json


ARMS = ("bundled_reads", "forward_cache", "role_postings", "auto")
COVERAGE = (0, 25, 50, 75, 100)
MISSINGNESS = ("author_declaration", "whole_article")
TOP = 5
REPEATS = 7


def prepare(catalog, folders, output):
    if output.exists():
        raise FileExistsError(output)
    expected = json.loads(catalog.read_bytes())["records"]
    loaded = load(folders)
    sources = []
    for doi, record in sorted(loaded.items()):
        if doi not in expected:
            continue
        if expected[doi]["source_sha256"] != record["source_sha256"]:
            raise ValueError(f"Publisher digest mismatch for {doi}")
        record["authors"] = [
            dict(ordinal=author["ordinal"], person=author["person"], roles=author["roles"])
            for author in record["authors"]
        ]
        sources.append(record)
    if not sources:
        raise ValueError("No pinned publisher data")
    # Only author ordinals, dataset-local identity tokens and licensed role labels are released.
    with output.open("x") as stream:
        json.dump(dict(catalog_sha256=digest(catalog), sources=sources), stream, ensure_ascii=True, indent=2)
        stream.write("\n")


def owner_data(sources):
    owners = defaultdict(list)
    byline = {}
    values = {}
    for article in sorted(sources, key=lambda row: row["doi"]):
        doi = article["doi"]
        for row in article["authors"]:
            owner = row["person"] or f"{doi}::{row['ordinal']}"
            if (owner, doi) in values:
                raise ValueError("Duplicate author identity in one publication")
            roles = frozenset(row["roles"])
            owners[owner].append(doi)
            values[owner, doi] = roles
            byline[owner] = min(byline.get(owner, row["ordinal"]), row["ordinal"])
    order = sorted(owners, key=lambda owner: (byline[owner], owner))
    return [(owner, tuple(owners[owner])) for owner in order], values


def compile_index(owners, values, percent, missingness):
    def resolve(owner, doc):
        roles = values[owner, doc]
        if roles:
            group = doc if missingness == "whole_article" else f"{owner}|{doc}"
            stable = sha256(group.encode()).digest()
            if int.from_bytes(stable[:8], "big") % 100 < percent:
                return roles
        return None

    start = perf_counter_ns()
    index = RoleEvidenceIndex(
        (owner, [(doc, resolve(owner, doc)) for doc in documents]) for owner, documents in owners
    )
    return index, (perf_counter_ns() - start) / 1000


def bundled(owners, values, roles, strategy="bundled_first"):
    scope = [(owner, (documents,) * len(roles)) for owner, documents in owners]

    def read(owner, doc):
        evidence = values[owner, doc]
        return {i: role in evidence for i, role in enumerate(roles)}

    return resolve_constraints(scope, read, top=TOP, strategy=strategy)


def compare(owners, values, references, missingness):
    def read(owner, doc):
        return values[owner, doc]

    report = {}
    for percent in COVERAGE:
        index, compile_us = compile_index(owners, values, percent, missingness)
        actual_coverage = sum(labels is not None for docs in index.documents for _, labels in docs)
        arms = {
            "bundled_reads": lambda request: bundled(owners, values, request),
            "forward_cache": lambda request: index.rank(request, read, top=TOP, postings=False),
            "role_postings": lambda request: index.rank(request, read, top=TOP, postings=True),
            "auto": lambda request: index.rank(request, read, top=TOP),
        }
        rows = defaultdict(list)
        for i, (request, expected) in enumerate(references.items()):
            times = defaultdict(list)
            counts = {}
            for repeat in range(REPEATS):
                rotation = (i + repeat) % len(ARMS)
                for name in ARMS[rotation:] + ARMS[:rotation]:
                    start = perf_counter_ns()
                    found = arms[name](request)
                    times[name].append((perf_counter_ns() - start) / 1000)
                    if found["ranking"] != expected:
                        raise AssertionError("Rank or canonical witness changed")
                    counts[name] = found
            for name in ARMS:
                rows[name].append(dict(reads=counts[name]["reads"],
                                       visited=counts[name].get("visited_candidates"),
                                       median_us=float(np.median(times[name]))))
        report[percent] = dict(
            build_us=compile_us, indexed_declarations=actual_coverage, methods={
                name: dict(p50_us=float(np.median([row["median_us"] for row in observed])),
                           p95_us=float(np.quantile([row["median_us"] for row in observed], .95)),
                           mean_open_reads=mean(row["reads"] for row in observed),
                           mean_visited_candidates=mean(row["visited"] for row in observed)
                           if observed[0]["visited"] is not None else None)
                for name, observed in rows.items()
            },
        )
    return report


def run(fixture, output):
    if output.exists():
        raise FileExistsError(output)
    sources = json.loads(fixture.read_bytes())["sources"]
    owners, values = owner_data(sources)
    roles = sorted(set().union(*values.values()))
    references = {request: bundled(owners, values, request, "exhaustive")["ranking"]
                  for request in combinations(roles, 3)}
    report = {missingness: compare(owners, values, references, missingness) for missingness in MISSINGNESS}
    write_json(output, dict(
        fixture_sha256=digest(fixture), script_sha256=digest(Path(__file__)),
        index_sha256=digest(Path(inspect.getfile(RoleEvidenceIndex))),
        queries=len(references), owner_count=len(owners), articles=len(sources),
        declarations=len(values), source_coverage_counts=Counter(map(len, (docs for _, docs in owners))),
        positive_queries=sum(bool(ranking) for ranking in references.values()),
        full_top5=sum(len(ranking) == TOP for ranking in references.values()),
        repeat_timings=REPEATS, report=report,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    build = actions.add_parser("prepare")
    build.add_argument("catalog", type=Path)
    build.add_argument("output", type=Path)
    build.add_argument("folders", nargs="+", type=Path)
    replay = actions.add_parser("run")
    replay.add_argument("fixture", type=Path)
    replay.add_argument("output", type=Path)
    args = vars(parser.parse_args())
    {"prepare": prepare, "run": run}[args.pop("action")](**args)
