"""Overlap CPU evidence preparation with one canonical GPU lane under offered load."""

import argparse
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from itertools import count
import json
import os
from pathlib import Path
from queue import PriorityQueue
from time import perf_counter, sleep

import numpy as np

from experiments.credit import elife_mesh as mesh
from experiments.credit import elife_mesh_scope as scope
from experiments.credit import role_field_ranking as field
from experiments.credit import role_gap_selective_recheck as scoring
from experiments.credit import role_gap_title_refill as gap
from experiments.credit import stable_topic_compaction as batches
from experiments.credit import topic_role_postings as pilot


RATES = (18, 36)
SEEDS = (44071, 44083)
REPEATS = 2


def prepare(catalog, titles, reader, request, vector):
    topic, _, pair = request
    ranked = catalog.rank_for_vector(topic, vector)
    excluded = catalog.visible[100] - catalog.person[pair]
    sources = gap.refill(ranked, lambda: np.argsort(-(titles @ vector), kind="stable").tolist(), excluded)
    encoded = reader.encode(f"Research about {topic}", [catalog.texts[i] for i in sources])
    return sources, encoded


def verify(catalog, reader, request, bundle):
    topic, _, pair = request
    sources, encoded = bundle
    values, _ = reader.score(encoded, range(len(encoded)))
    return scoring.serialize(catalog, topic, pair, sources, dict(enumerate(values)))[1]


def encode(catalog, request):
    return catalog.encoder.encode([f"Research about {request[0]}"], query=True)[0]


def drive(catalog, titles, reader, cases, schedule, pipelined):
    queue = PriorityQueue()
    sequence = count()
    futures = [Future() for _ in cases]
    arrivals = []
    gpu_seconds = Counter()
    prep_seconds = [0.] * len(cases)

    def enqueue(priority, index, payload=None):
        queue.put((priority, next(sequence), index, payload))

    def prep(index, vector):
        start = perf_counter()
        bundle = prepare(catalog, titles, reader, cases[index], vector)
        prep_seconds[index] = perf_counter() - start
        return bundle

    def commit_prepared(index, done):
        try:
            enqueue(0, index, done.result())
        except Exception as exc:
            futures[index].set_exception(exc)

    def gpu_lane(cpu):
        while True:
            kind, _, index, payload = queue.get()
            if index is None:
                return
            try:
                if kind:
                    start = perf_counter()
                    vector = encode(catalog, cases[index])
                    gpu_seconds["encoding"] += perf_counter() - start
                    if pipelined:
                        cpu.submit(prep, index, vector).add_done_callback(partial(commit_prepared, index))
                        continue
                    payload = prep(index, vector)
                start = perf_counter()
                output = verify(catalog, reader, cases[index], payload)
                ended = perf_counter()
                gpu_seconds["verification"] += ended - start
                futures[index].set_result((output, ended))
            except Exception as exc:
                futures[index].set_exception(exc)

    with ThreadPoolExecutor(max_workers=1) as cpu, ThreadPoolExecutor(max_workers=1) as gpu:
        lane = gpu.submit(gpu_lane, cpu)
        started = perf_counter()
        try:
            for i, offset in enumerate(schedule):
                arrival = started + offset
                sleep(max(0., arrival - perf_counter()))
                arrivals.append(perf_counter())
                enqueue(1, i)
            finished = [future.result(timeout=30) for future in futures]
        finally:
            enqueue(2, None)
            lane.result(timeout=30)

    end = np.array([stamp - started for _, stamp in finished])
    latencies = (end - schedule) * 1000
    return [body for body, _ in finished], dict(
        p50_p95_p99_ms=np.quantile(latencies, [.50, .95, .99]).tolist(),
        p95_arrival_lag_ms=float(np.quantile(np.asarray(arrivals) - started - schedule, .95) * 1000),
        drain_ms=float((end.max() - schedule[-1]) * 1000),
        offered_requests_per_second=float((len(end) - 1) / (schedule[-1] - schedule[0])),
        completed_requests_per_second=float(len(end) / end.max()),
        gpu_lane_seconds=dict(gpu_seconds), cpu_preparation_seconds=sum(prep_seconds),
    )


def run(elife, snapshot, frozen, embedding, reranker, output):
    import torch
    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.followir.core17_verifier import Reranker

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    records = json.loads(snapshot.read_bytes())
    if records["projection_sha256"] != pilot.digest(elife):
        raise ValueError("Expected frozen eLife catalog")
    prior = json.loads(frozen.read_bytes())["run"]
    info = pilot.verify_checkpoint(embedding)
    torch.backends.cuda.matmul.allow_tf32 = False
    encoder = QwenTorchEmbedding(embedding)
    encoder.model.config.use_cache = False
    ranker = Reranker(reranker)
    if (info["revision"], ranker.provenance["revision"]) != (prior["embedding"], prior["reranker"]):
        raise ValueError("Expected frozen checkpoints")
    catalog, titles, _ = field.data_model(elife, encoder, ranker)
    complete = {i for i, row in enumerate(catalog.articles) if row["declaration_complete"]}
    catalog.visible[100] &= complete
    mesh.assign_topics(catalog, records["records"], mesh.ROUTINE_DESCRIPTORS, major_only=False)
    topics = sorted(t for t, sources in catalog.topics.items()
                    if scope.MINIMUM_SUPPORT <= len(sources) <= scope.MAXIMUM_SUPPORT)
    queries = list(pilot.query_panel(catalog, topics))
    reader = batches.BatchReader(ranker)
    if len(queries) != 110:
        raise ValueError("Unexpected replay query count")
    for mode in (False, True):
        drive(catalog, titles, reader, queries[:4], np.arange(4) * .002, mode)

    results = {}
    for rate in RATES:
        results[rate] = {}
        for index, seed in enumerate(SEEDS):
            rng = np.random.default_rng(seed)
            cases = [queries[i] for _ in range(REPEATS) for i in rng.permutation(len(queries))]
            interarrival = rng.exponential(1.0 / rate, size=len(cases) - 1)
            schedule = np.concatenate(([0.], np.cumsum(interarrival)))
            structures = ("serial", "pipeline") if index == 0 else ("pipeline", "serial")
            rows = {
                route: drive(catalog, titles, reader, cases, schedule, route == "pipeline")
                for route in structures
            }
            reference, measured = rows["serial"][0], rows["pipeline"][0]
            canonical = prior["elife"]["metrics"]["100"]["gap_title_four"]["compared"]["topic_four"]
            witnesses = [json.loads(body) for body in reference]
            if (sum(len(row) for row in witnesses) != REPEATS * canonical["issued"]
                    or sum(source in catalog.topics[case[0]]
                           for case, row in zip(cases, witnesses, strict=True)
                           for _, source in row) != REPEATS * canonical["grounded"]):
                raise ValueError("Serial route does not reproduce the frozen witness counts")
            results[rate][seed] = dict(
                requests=len(cases), exact_responses=sum(a == b for a, b in zip(reference, measured, strict=True)),
                routes={route: report for route, (_, report) in rows.items()},
            )
    pilot.write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), script_sha256=pilot.digest(Path(__file__)),
        reference_sha256=pilot.digest(frozen), projection_sha256=pilot.digest(elife),
        mesh_sha256=pilot.digest(snapshot), queries=len(queries), results=results,
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("elife", "snapshot", "frozen", "embedding", "reranker", "output"):
        cli.add_argument(name, type=Path)
    run(**vars(cli.parse_args()))
