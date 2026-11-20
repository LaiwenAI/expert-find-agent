"""Observe queue buildup across paired loopback arrival rates."""

import argparse
import os
from pathlib import Path

import numpy as np

from experiments.retrieval import enterprise_packed_role_load as service
from experiments.retrieval.enterprise_role_first_query import require_cuda
from experiments.retrieval.enterprise_routed_review import digest, write


RATES = (2.2, 2.7, 3.2)
ARMS = ("padded_rescue", "guarded_rescue")
CYCLES = 2
SEED = 20261120


def workload_schedule(size, rate):
    rng = np.random.default_rng(SEED)
    order = [i for _ in range(CYCLES) for i in rng.permutation(size).tolist()]
    gaps = rng.exponential(size=len(order) - 1)
    times = np.r_[0., np.cumsum(gaps)]
    return order, times * ((len(order) - 1) / rate / times[-1])


def queue_summary(times, timings):
    completions = np.sort([row["done"] for row in timings])
    if len(completions) != len(times) or not np.isfinite(completions).all():
        raise ValueError("Malformed completion sequence")
    milestones = [len(times) * (i+1) // 4 - 1 for i in range(4)]
    # The current scheduled arrival counts as present at each boundary.
    unfinished = [int(ind + 1 - np.searchsorted(completions, times[ind], side="right"))
                  for ind in milestones]
    done_before_last = int(np.searchsorted(completions, times[-1], side="right"))
    if min(unfinished) < 0:
        raise ValueError("Completion preceded scheduled arrival")
    queue_blocks = [
        float(np.quantile([timings[i]["queue_ms"] for i in ids], .95))
        for ids in np.array_split(np.arange(len(times)),4)
    ]
    end = float(completions[-1])
    return dict(
        unfinished_at_arrival_quartile=unfinished,
        queue_p95_by_arrival_quartile_ms=queue_blocks,
        completions_before_last_arrival=done_before_last,
        completed_per_s_during_arrivals=done_before_last / times[-1],
        last_completion_seconds=end, drain_seconds=max(0, end - times[-1]),
    )


def run(queries, corpus, roles, vectors, meta, models, serial, output):
    require_cuda(output)
    texts, fixture, fingerprint = service.init_fixture(queries, corpus, roles, vectors, meta, models, serial)
    report = {}
    for i, rate in enumerate(RATES):
        order, times = workload_schedule(len(texts), rate)
        results, responses = {}, {}
        arms = ARMS if i % 2 == 0 else ARMS[::-1]
        for arm in arms:
            responses[arm], aggregate = service.replay(arm, fixture, texts * CYCLES, order, times, timeline=True)
            timeline = aggregate.pop("timings")
            results[arm] = dict(aggregate, queue_dynamics=queue_summary(times, timeline))
            print(f"Completed {arm} at {rate:.1f} offered requests per second", flush=True)
        differences = sum(a != b for a,b in zip(responses[ARMS[0]],responses[ARMS[1]],strict=True))
        report[rate] = dict(execution_order=arms, response_byte_changes=differences, arms=results)
    write(output, dict(
        job=os.environ["SLURM_JOB_ID"], code_sha256=digest(Path(__file__)),
        service_sha256=digest(Path(service.__file__)), seed=SEED, cycles=CYCLES, report=report, **fingerprint,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("queries", "corpus", "roles", "vectors", "meta", "models", "serial", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
