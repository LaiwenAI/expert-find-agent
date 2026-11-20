"""Audit a bound evidence lookup when sources are withheld or misassociated."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.credit import legacy_contribution_text as legacy
from experiments.followir.core17_reranking import digest, write_json


ROLES = {phrase: key for key, phrases in legacy.PHRASES.items() for phrase in phrases}
ROUTES = ("pooled", "report_scoped", "provenance_bound")
UNRESOLVED = "unresolved"


def notes_index(text):
    postings = {}
    offset = 0
    for sentence in legacy.SENTENCE.split(text):
        start = text.index(sentence, offset)
        end = start + len(sentence)
        offset = end
        head, separator, tail = sentence.partition(":")
        role = legacy.LABELS.get(head.casefold().strip())
        initials = tail.strip().rstrip(".").split()
        if separator and role and initials and all(legacy.INITIAL.fullmatch(name) for name in initials):
            if role in postings:
                return {}
            postings[role] = frozenset(initials), start, end
    return postings


def content_id(text):
    return sha256(text.encode()).hexdigest()


def fixture_views(row):
    labels, notes = row["study_labels"], row["notes"]
    sources = dict(zip(labels, notes, strict=True))
    expected = {label: content_id(text) for label, text in sources.items()}
    good, bad = labels[row["positive_index"]], labels[1 - row["positive_index"]]
    tasks = row["tasks"]
    return (
        ("supported", good, tasks, sources, expected, "supported"),
        ("unsupported", bad, tasks, sources, expected, "unsupported"),
        ("missing_report", good, tasks, {bad: sources[bad]}, expected, UNRESOLVED),
        ("substituted_report", bad, tasks, {**sources, bad: sources[good]}, expected, UNRESOLVED),
        ("unknown_role", good, [tasks[0], "audited the instrument calibration logs"], sources, expected, UNRESOLVED),
    )


def evidence(route, report, tasks, sources, expected, indexes, initials):
    roles = tuple(ROLES.get(task) for task in tasks)
    if None in roles or (route != "pooled" and report not in sources):
        return UNRESOLVED, ()
    if route == "provenance_bound" and content_id(sources[report]) != expected[report]:
        return UNRESOLVED, ()
    candidates = sources if route == "pooled" else (report,)
    citations = []
    for role in roles:
        could_verify = False
        for label in candidates:
            text = sources[label]
            if role not in indexes[text]:
                continue
            could_verify = True
            owners, start, end = indexes[text][role]
            if initials in owners:
                citations.append((label, start, end))
                break
        else:
            return "unsupported" if could_verify else UNRESOLVED, ()
    return "supported", tuple(citations)


def compare(fixture):
    rows = fixture["rows"]
    start = perf_counter()
    indexes = {text: notes_index(text) for row in rows for text in row["notes"]}
    index_ms = (perf_counter() - start) * 1000
    metrics = {route: defaultdict(Counter) for route in ROUTES}
    times = {route: [] for route in ROUTES}
    for row in rows:
        for kind, report, tasks, sources, expected, gold in fixture_views(row):
            for route in ROUTES:
                start = perf_counter()
                decision, citations = evidence(route, report, tasks, sources, expected, indexes, row["initials"])
                times[route].append((perf_counter() - start) * 1000)
                metrics[route][kind].update(
                    cases=1, correct=int(decision == gold),
                    false_support=int(decision == "supported" and gold != "supported"),
                    unresolved=int(decision == UNRESOLVED),
                    wrong_source_witnesses=sum(label != report or content_id(sources[label]) != expected[label]
                                               for label, _, _ in citations),
                )
                for label, begin, end in citations:
                    if not (0 <= begin < end <= len(sources[label]) and row["initials"] in sources[label][begin:end]):
                        raise ValueError("Citation is missing its original owner and offsets")
    return dict(indexed_sources=len(indexes), role_entries=sum(map(len, indexes.values())),
                index_ms=index_ms, cases=5 * len(rows),
                results={route: dict(by_view=metrics[route], p95_ms=float(np.quantile(times[route], .95)))
                         for route in ROUTES})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("fixture", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    records = json.loads(args.fixture.read_bytes())
    write_json(args.output, dict(fixture_sha256=digest(args.fixture), script_sha256=digest(Path(__file__)),
                                 host=platform.node(), python=platform.python_version(),
                                 **compare(records)))
