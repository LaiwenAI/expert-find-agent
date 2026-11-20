"""Replay first-query role nomination with matched verifier reads and live timing."""

import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from experiments.credit import bounded_topic_nomination as nomination
from experiments.credit import role_first_query_factorial as factorial
from experiments.credit import topic_role_postings as pilot
from experiments.followir.core17_reranking import digest, write_json


ARMS = ("topic_adaptive", "person_adaptive", "topic_fixed", "person_fixed", "person_global_fixed")
FIRST, POOL, EXTRA = 4, 8, 2


def replay(catalog, topic, pair, visible, cutoff, arm, *, include_witnesses=False):
    started = perf_counter_ns()
    ranked = catalog.rank(topic)
    original = ranked[:POOL]
    compatible = [i for i in ranked if i not in visible or i in catalog.person[pair]]
    first = (compatible if arm.startswith("person") else ranked)[:FIRST]
    initial_scores = catalog.verify(topic, first)
    initial = nomination.evaluate(catalog, topic, pair, first, initial_scores, cutoff)

    seen = set(first)
    local = [i for i in compatible if i in original and i not in seen]
    adaptive = arm.endswith("adaptive")
    early_stop = adaptive and initial["issued"] == pilot.TOP
    if early_stop:
        extra = []
    elif arm.endswith("_global_fixed"):
        extra = [i for i in compatible if i not in seen][:EXTRA]
    else:
        extra = local[:EXTRA]
    initial_extra = len(extra)
    if not adaptive and len(extra) < EXTRA:
        spare = [i for i in original if i not in seen and i not in extra]
        extra.extend(spare[:EXTRA - len(extra)])
    if len(set(first + extra)) != len(first) + len(extra):
        raise ValueError("Repeated source")
    if not adaptive and len(extra) != EXTRA:
        raise ValueError("Fixed route missed the second-stage budget")

    extra_scores = catalog.verify(topic, extra) if extra else []
    final = nomination.evaluate(
        catalog, topic, pair, first + extra, list(initial_scores) + list(extra_scores), cutoff,
    )
    elapsed_ms = (perf_counter_ns() - started) / 1e6
    result = dict(
        grounded=final["grounded"], off_label=final["cited_off_label"], issued=final["issued"],
        first_grounded=initial["grounded"], reads=len(first + extra),
        skipped_when_full=EXTRA if early_stop else 0,
        missing_compatible=EXTRA - initial_extra if not early_stop else 0,
        padded=len(extra) - initial_extra,
        outside_pool=len(set(extra) - set(original)),
        elapsed_ms=elapsed_ms,
    )
    if include_witnesses:
        result.update(first_witnesses=initial["witnesses"], witnesses=final["witnesses"])
    return result


def summarize(all_rows):
    measures = {}
    for arm, trials in all_rows.items():
        counts = Counter()
        for row in trials:
            counts.update({key: value for key, value in row.items() if key != "elapsed_ms"})
        measures[arm] = dict(counts, serial_p95_ms=float(np.quantile([r["elapsed_ms"] for r in trials], .95)))
    return measures


def run(projection, embedding, reranker, reference, offset):
    frozen = json.loads(reference.read_bytes())
    if digest(projection) != frozen["projection_sha256"]:
        raise ValueError("Unexpected projection")
    for directory, key in ((embedding, "embedding_sha256"),
                           (reranker / factorial.Verifier.name, "reranker_sha256")):
        if {name: digest(directory / name) for name in frozen[key]} != frozen[key]:
            raise ValueError(f"Unexpected checkpoint files: {directory}")

    articles = json.loads(projection.read_bytes())["sources"]
    catalog = pilot.Catalog(articles, factorial.Encoder(embedding), factorial.Verifier(reranker))
    complete = {i for i, article in enumerate(articles) if article["declaration_complete"]}
    visible = catalog.visible[100] & complete
    panels = factorial.panel_queries(catalog)
    cutoff = frozen["cutoff"]
    results = {}
    for field, (lookup, queries) in panels.items():
        catalog.topics = lookup
        test = [(topic, pair) for topic, _, pair in queries if int.from_bytes(
            pilot.stable("factorial-calibration-v1", topic)[:8], "big") % 3 != 0]
        if len(test) != frozen["results"][field]["queries"]:
            raise ValueError("Evaluation split differs from the frozen replay")
        for arm in ARMS:
            replay(catalog, *test[0], visible, cutoff, arm)
        counts = {arm: [] for arm in ARMS}
        for i, (topic, pair) in enumerate(test):
            shift = (offset + i) % len(ARMS)
            for arm in ARMS[shift:] + ARMS[:shift]:
                counts[arm].append(replay(catalog, topic, pair, visible, cutoff, arm))
        results[field] = dict(queries=len(test), arms=summarize(counts))
    return dict(reference_sha256=digest(reference), order_offset=offset, cutoff=cutoff, results=results)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("projection", "embedding", "reranker", "reference", "output"):
        cli.add_argument(name, type=Path)
    cli.add_argument("--order-offset", type=int, default=0)
    args = cli.parse_args()
    report = run(args.projection, args.embedding, args.reranker, args.reference, args.order_offset)
    write_json(args.output, dict(script_sha256=digest(Path(__file__)), **report))
