"""Measure shared-weight packed verification over continuous and saturated load."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from transformers import AttentionInterface

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.retrieval.cof_global_profile_pooling import ProfileIndex, read_data, select
from experiments.retrieval.cof_ragged_concurrent import PackedBroker, WORKERS, replay, serve
from experiments.retrieval.cof_ragged_frontier import PACKED_ATTENTION, packed_attention
from experiments.retrieval.cof_ragged_shared_weights import measure_storage, packed_view


RPS = 12
WINDOW = 12
WINDOWS = 3
SATURATED_SECONDS = 18
ARMS = ("padded4", "packed8")


def p95_ms(values):
    return 1000 * float(np.quantile(values, .95))


def saturated(sequence, arm, query_fn, comparison):
    def uncalled(_):
        raise AssertionError("Saturation test does not enable cross-request merging")

    with PackedBroker(uncalled) as broker, ThreadPoolExecutor(WORKERS) as executor:
        start = perf_counter()
        deadline = start + SATURATED_SECONDS

        def worker(offset):
            checked = 0
            completed = []
            i = offset
            while (arrival := perf_counter()) < deadline:
                q = sequence[i % len(sequence)]
                outcome, elapsed, _, finished = query_fn(arm, arrival, q, broker=broker)
                checked += outcome == comparison[q]
                completed.append((elapsed, finished - start))
                i += WORKERS
            return checked, completed

        samples = list(executor.map(worker, range(WORKERS)))
    events = [event for _, values in samples for event in values]
    completed = [event for event in events if event[1] <= SATURATED_SECONDS]
    return dict(
        exact=sum(count for count, _ in samples), submitted=len(events),
        completed_within_window=len(completed),
        completed_rps=len(completed) / SATURATED_SECONDS,
        p95_ms=p95_ms([time for time, _ in events]),
    )


def report_open(samples, sequence, canonical):
    width = WINDOW * RPS
    arrivals = len(sequence)
    assert arrivals == width * WINDOWS
    return dict(
        requests=arrivals,
        exact=sum(out == canonical[q] for q, out in zip(sequence, samples["outputs"], strict=True)),
        p95_ms=p95_ms(samples["latency"]),
        worker_wait_p95_ms=p95_ms(samples["worker_wait"]),
        arrival_lag_max_ms=1000 * max(samples["arrival_lag"]),
        drained_rps=arrivals / samples["duration"],
        window_p95_ms=[p95_ms(samples["latency"][i:i + width]) for i in range(0, arrivals, width)],
        window_completed=[
            sum(i * WINDOW <= elapsed + pos / RPS < (i + 1) * WINDOW
                for pos, elapsed in enumerate(samples["latency"]))
            for i in range(WINDOWS)
        ],
    )


def run(dataset, embedding, reranker, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    provenance = verify_checkpoint(embedding)
    root, papers, _, reference, ids, _, _ = read_data(dataset, provenance)
    texts = {pid: (paper.get("title", "") + "\n" + paper.get("abstract", "")).strip()
             for pid, paper in papers.items()}
    index = ProfileIndex(reference)
    encoder = QwenTorchEmbedding(embedding, batch_size=1)
    encoder.model.config.use_cache = False
    reader = Reranker(reranker)
    base = reader.model.model
    if reader.model.lm_head.bias is not None or set(base.config.layer_types) != {"full_attention"}:
        raise ValueError("Expected the pinned bias-free, full-context reranker")
    AttentionInterface.register(PACKED_ATTENTION, packed_attention)
    with torch.inference_mode():
        weights = reader.model.lm_head.weight[reader.labels].contiguous()
        packed = packed_view(base, share_weights=True)
    storage = measure_storage(base, packed)
    if storage["extra_parameter_bytes"] or storage["shared_parameter_bytes"] <= 0:
        raise AssertionError("Expected shared packed parameter storage")

    def query_fn(arm, arrived, q, *, broker):
        return serve(arm, arrived, texts[q], texts, encoder, index, reader, weights, packed, broker)

    panel = sorted(select(dataset, ids, texts), key=lambda q: sha256(f"sustained-v1:{q}".encode()).digest())
    sequence = [panel[pos % len(panel)] for pos in range(RPS * WINDOW * WINDOWS)]
    for arm in ARMS:
        replay(30, panel[:6], arm, query_fn, packed, weights)
    trials = {}
    execution_order = ARMS if dataset == "KDD" else tuple(reversed(ARMS))
    for arm in execution_order:
        trials[arm] = replay(RPS, sequence, arm, query_fn, packed, weights)
        print(f"{dataset}: completed {arm} open loop", flush=True)
    canonical = {}
    for q, out in zip(sequence, trials["padded4"]["outputs"], strict=True):
        if canonical.setdefault(q, out) != out:
            raise AssertionError("Reference changed across repeated queries")
    open_loop = {arm: report_open(trials[arm], sequence, canonical) for arm in ARMS}
    saturation = {}
    for arm in reversed(execution_order):
        saturation[arm] = saturated(panel, arm, query_fn, canonical)
        print(f"{dataset}: completed {arm} saturation", flush=True)
    write_json(output, dict(
        dataset=dataset, job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), workers=WORKERS,
        shared_weights=storage, panel_queries=len(panel), arrival_rps=RPS, window_seconds=WINDOW,
        windows=WINDOWS, saturated_seconds=SATURATED_SECONDS,
        panel_sha256=sha256("\n".join(panel).encode()).hexdigest(), code_sha256=digest(Path(__file__)),
        vector_protocol_sha256=digest(root / "qwen-embedding-torch-v1/protocol.json"),
        embedding_revision=provenance["revision"], reranker_revision=reader.provenance["revision"],
        open_loop=open_loop, saturation=saturation,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("KDD", "SciRepEval"))
    for name in ("embedding", "reranker", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
