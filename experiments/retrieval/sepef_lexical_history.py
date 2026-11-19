"""Test fast community-local lexical aggregation against the full activity prior."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from expert_search.statistics import clustered_paired_interval
from experiments.followir.core17_reranking import digest, write_json
from experiments.retrieval.sepef_prior_fusion import DEPTH, fuse, quality


WEIGHTS = (0., .05, .1, .25, .5, 1.)
MODES = ("max_answer", "sum_top100")
ARMS = ("prior", *MODES, "validation_fusion")
METRICS = ("hit5", "mrr10")


def rank(docs, scores, source_owners, owners, ids, mode):
    ordered = docs[np.argsort(-scores[docs], kind="stable")]
    if mode == "sum_top100":
        ordered = ordered[:100]
    totals, evidence = {}, {}
    for doc in ordered:
        owner = source_owners[doc]
        if owner not in evidence:
            evidence[owner] = doc
            totals[owner] = scores[doc]
        elif mode == "sum_top100":
            totals[owner] += scores[doc]
    top = sorted(totals, key=lambda owner: (-totals[owner], owner))[:DEPTH]
    return [(owners[owner], ids[evidence[owner]]) for owner in top]


def averages(rows):
    return {field: float(np.mean([row[field] for row in rows])) for field in METRICS}


def evaluate(panel, prior, scratch, output):
    from experiments.retrieval import sepef_first_query as baseline

    if output.exists() or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Require a fresh output and a Slurm compute node")
    owners, ids, texts, positions, manifest = baseline.payload(panel)
    data = json.loads(prior.read_bytes())
    if (manifest["source_sha256"]["SE-PEF/SE-PEF/expert_test_data.json"] != data["history_sha256"]
            or any(digest(panel / f"{name}.json") != expected for name, expected in data["query_sha256"].items())):
        raise ValueError("Prior and lexical panels differ")
    groups = defaultdict(list)
    for i, identifier in enumerate(ids):
        groups[identifier.split("_", 1)[0]].append(i)
    community = {site: np.array(indices) for site, indices in groups.items()}
    prior_by_site = data["ranking"]
    with TemporaryDirectory(dir=scratch, prefix="sepef-lexical-history-") as tmp:
        start = perf_counter()
        index = baseline.lexical_index(texts, Path(tmp))
        index_seconds = perf_counter() - start

        def matches(row, mode):
            scores, active = index.document_arrays(baseline.clean(row["text"]))
            documents = community[row["community"]]
            return rank(documents[active[documents]], scores, positions, owners, ids, mode)

        def leaders(row):
            return prior_by_site[row["community"]]

        val = json.loads((panel / "val.json").read_bytes())
        trials = {(mode, weight): [] for mode in MODES for weight in WEIGHTS}
        for row in val:
            for mode in MODES:
                lexical = matches(row, mode)
                for weight in WEIGHTS:
                    trials[mode, weight].append(quality(row, fuse(leaders(row), lexical, weight)))
        validation = {arm: averages(values) for arm, values in trials.items()}
        mode, weight = max(validation, key=lambda arm: (validation[arm]["mrr10"], validation[arm]["hit5"], -arm[1]))
        selected = dict(mode=mode, lexical_weight=weight, quality=validation[mode, weight])
        test = json.loads((panel / "test.json").read_bytes())
        measurements = {arm: [] for arm in ARMS}
        for i, row in enumerate(test):
            offset = i % len(ARMS)
            for arm in ARMS[offset:] + ARMS[:offset]:
                start = perf_counter()
                if arm == "prior" or (arm == "validation_fusion" and weight == 0):
                    results = leaders(row)[:baseline.TOP]
                elif arm == "validation_fusion":
                    results = fuse(leaders(row), matches(row, mode), weight)
                else:
                    results = matches(row, arm)[:baseline.TOP]
                json.dumps(results, separators=(",", ":"))
                elapsed = (perf_counter() - start) * 1000
                measurements[arm].append(dict(**quality(row, results), ms=elapsed))
    controls = {mode: dict(**averages(rows), p95_ms=float(np.quantile([row["ms"] for row in rows], .95)))
                for mode, rows in measurements.items()}
    clusters = [row["community"] for row in test]
    paired = {arm: {field: clustered_paired_interval(
        [float(a[field]) - float(b[field]) for a, b in zip(values, measurements["prior"], strict=True)], clusters,
    ) | {"sampling_unit": "whole community; query-weighted mean"} for field in METRICS}
        for arm, values in measurements.items() if arm != "prior"}
    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], script_sha256=digest(Path(__file__)),
        manifest_sha256=digest(panel / "manifest.json"), prior_sha256=digest(prior),
        validation_queries=len(val), test_queries=len(test), selected=selected,
        validation_grid={mode: {str(weight): validation[mode, weight] for weight in WEIGHTS} for mode in MODES},
        test=controls, paired_minus_prior=paired, index_seconds=index_seconds,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("panel", "prior", "scratch", "output"):
        parser.add_argument(arg, type=Path)
    evaluate(**vars(parser.parse_args()))
