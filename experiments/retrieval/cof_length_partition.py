"""Partition frontier waves using measured CUDA costs at padded shapes."""

import argparse
from functools import partial
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import numpy as np

from expert_search.deferred_verification import resolve_shared_frontier
from expert_search.statistics import paired_interval
from experiments.retrieval.cof_binary_head import BinaryWaveReader
from experiments.retrieval.cof_frontier_wave import SUPPORT, WaveScheduler, run, verify
from experiments.retrieval.cof_live_certified_batching import CAP, GUARD_MARGIN, TOP


ARMS = ("frontier4", "frontier8", "bucket8", "profiled8")
LENGTH_STEP = 64
WIDTH = 8


def plan(sorted_jobs, lengths, cost):
    """Partition a length-sorted wave into contiguous batches of minimum cost."""
    best = [(0.0, 0, ())]
    for end in range(1, len(sorted_jobs) + 1):
        estimates = [
            (value + cost(end - start, lengths[end - 1]), count + 1, (*cuts, end))
            for start, (value, count, cuts) in enumerate(best)
        ]
        best.append(min(estimates))
    cuts = (0, *best[-1][2])
    return [sorted_jobs[left:right] for left, right in zip(cuts, cuts[1:])]


def profile(reader, weights):
    wave = BinaryWaveReader(reader, "Calibration", {}, (), "two_rows_batched_tokenization", weights)
    # Probe shape cost with a non-padding token and no task or dataset labels.
    token = reader.labels[0]
    shapes = [(n, length) for n in range(1, WIDTH + 1) for length in range(LENGTH_STEP, CAP + 1, LENGTH_STEP)]
    timing = {shape: [] for shape in shapes}
    wave._forward([{"input_ids": [token] * CAP}] * WIDTH, dynamic=True)
    for repeat in range(3):
        order = shapes if repeat % 2 else list(reversed(shapes))
        for n, length in order:
            rows = [{"input_ids": [token] * length}] * n
            begin = perf_counter()
            wave._forward(rows, dynamic=True)
            timing[n, length].append((perf_counter() - begin) * 1000)
    costs = {shape: median(values) for shape, values in timing.items()}
    return costs


def score_partition(reader, jobs, mode, costs):
    ordered = sorted(jobs, key=lambda job: reader.length(job[1]))
    lengths = [reader.length(doc) for _, doc in ordered]
    if mode == "bucket8":
        batches = [ordered[i:i + 4] for i in range(0, len(ordered), 4)]
    else:
        def cost(rows, length):
            rounded = ((length + LENGTH_STEP - 1) // LENGTH_STEP) * LENGTH_STEP
            return costs[rows, rounded]
        batches = plan(ordered, lengths, cost)
    margins = {job: value for batch in batches
               for job, value in zip(batch, reader.score(batch), strict=True)}
    return [margins[job] for job in jobs]


def verify_partition(reader, weights, candidates, texts, query, arm, relevant, *, costs):
    if arm in ("frontier4", "frontier8"):
        return verify(reader, weights, candidates, texts, query, arm, relevant)
    begin = perf_counter()
    wave = BinaryWaveReader(reader, query, texts, {doc for _, docs in candidates for doc in docs},
                           "two_rows_batched_tokenization", weights)
    base = WaveScheduler(wave, candidates, WIDTH, False)
    out = resolve_shared_frontier(
        candidates, partial(score_partition, wave, mode=arm, costs=costs), base.recheck, wave.length,
        support=SUPPORT, top=TOP, width=WIDTH, band=GUARD_MARGIN, cost_aware=False,
    )
    return dict(
        selected=out["selected"], ms=(perf_counter() - begin) * 1000,
        forwards=wave.score_calls + wave.recheck_calls, docs=wave.scored_documents,
        padded_tokens=wave.padded_tokens, rechecks=wave.rechecked_pairs,
        cited_positives=sum(expert in relevant and bool(sources) for expert, sources in out["selected"]),
    )


def summarize(rows):
    result = {}
    for arm in ARMS:
        current = [row[arm] for row in rows]
        baseline = [row["frontier4"] for row in rows]
        result[arm] = dict(
            exact_outputs=sum(a["selected"] == b["selected"] for a, b in zip(current, baseline, strict=True)),
            cited_positives=sum(row["cited_positives"] for row in current),
            p95_ms=float(np.quantile([row["ms"] for row in current], .95)),
            paired_latency_delta=paired_interval([a["ms"] - b["ms"] for a, b in zip(current, baseline, strict=True)]),
            **{key: mean(row[key] for row in current)
               for key in ("forwards", "docs", "padded_tokens", "rechecks")},
        )
    return result


def run_partition(dataset, embedding, reranker, output):
    costs = {}

    def setup(reader, weights):
        started = perf_counter()
        costs.update(profile(reader, weights))
        return {"shape_profile_seconds": perf_counter() - started,
                "shape_costs_ms": {f"{n}x{length}": value for (n, length), value in costs.items()}}

    def measure(reader, weights, candidates, texts, query, arm, relevant):
        return verify_partition(reader, weights, candidates, texts, query, arm, relevant, costs=costs)

    run(dataset, embedding, reranker, output, methods=ARMS, measure=measure, aggregate=summarize, setup=setup)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("KDD", "SciRepEval"))
    for key in ("embedding", "reranker", "output"):
        parser.add_argument(key, type=Path)
    run_partition(**vars(parser.parse_args()))
