"""Compare fixed and packed evidence verification under matched open-loop arrivals."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import partial
from hashlib import sha256
import os
from pathlib import Path
import random
from time import perf_counter, sleep

import numpy as np
import torch
from transformers import AttentionInterface

from expert_search.deferred_verification import resolve_shared_frontier
from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.retrieval.cof_binary_head import BinaryWaveReader
from experiments.retrieval.cof_frontier_wave import SUPPORT, WaveScheduler
from experiments.retrieval.cof_global_profile_pooling import ProfileIndex, read_data, select
from experiments.retrieval.cof_live_certified_batching import GUARD_MARGIN, TOP
from experiments.retrieval.cof_ragged_frontier import PACKED_ATTENTION, packed_attention, packed_forward
from experiments.retrieval.cof_verification_microbatch import BatchBroker


RATES = (4, 8, 12)
ARMS = ("padded4", "packed8", "packed8_merge")
WORKERS = 4
REQUESTS = 40
REPEATS = 2


class PackedBroker(BatchBroker):
    @staticmethod
    def bucket(rows):
        # Packed attention does not require length-compatible batches.
        return 0


class ServiceReader(BinaryWaveReader):
    def __init__(self, reader, query, texts, candidates, arm, weights, packed, broker):
        self.mode = arm
        self.packed = packed
        super().__init__(reader, query, texts, {doc for _, docs in candidates for doc in docs},
                         "two_rows_batched_tokenization", weights)
        self.broker = broker

    def _forward(self, batch, dynamic):
        if not dynamic:
            return super()._forward(batch, dynamic)
        if self.mode == "packed8_merge":
            return self.broker.score(batch)[0]
        if self.mode == "packed8" and len(batch) > 1:
            return self.broker.serial(partial(packed_forward, self.packed, self.weight, batch))[0]
        return self.broker.serial(partial(super()._forward, batch, dynamic))


def serve(arm, arrived, query, texts, encoder, index, reader, weights, packed, broker):
    started = perf_counter()
    candidates = broker.serial(partial(index.search_query, query, encoder, "mean_max"))[:TOP]
    wave = ServiceReader(reader, query, texts, candidates, arm, weights, packed, broker)
    width = 4 if arm == "padded4" else 8
    policy = WaveScheduler(wave, candidates, width, False)
    result = resolve_shared_frontier(candidates, policy.score, policy.recheck, wave.length,
                                     support=SUPPORT, top=TOP, width=width, band=GUARD_MARGIN, cost_aware=False)
    end = perf_counter()
    return result["selected"], (end - arrived), (started - arrived), end


def replay(rate, sequence, arm, query_fn, packed, weights):
    def merged_forward(rows):
        values, _ = packed_forward(packed, weights, rows)
        return values, max(len(row["input_ids"]) for row in rows)

    with PackedBroker(merged_forward) as broker, ThreadPoolExecutor(WORKERS) as clients:
        start = perf_counter()
        submitted = []
        arrival_lag = []
        for i, q in enumerate(sequence):
            expected = start + i / rate
            delay = expected - perf_counter()
            if delay > 0:
                sleep(delay)
            arrival_lag.append(perf_counter() - expected)
            submitted.append(clients.submit(query_fn, arm, expected, q, broker=broker))
        values = [future.result() for future in submitted]
    duration = max(v[3] for v in values) - start
    work = dict(broker.work)
    # The broker's rectangular padded-token counter is not meaningful for packed execution.
    work.pop("dynamic_padded_tokens", None)
    return dict(outputs=[v[0] for v in values], latency=[v[1] for v in values],
                worker_wait=[v[2] for v in values], duration=duration, arrival_lag=arrival_lag, work=work)


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
    if reader.model.lm_head.bias is not None or set(reader.model.model.config.layer_types) != {"full_attention"}:
        raise ValueError("Expected the pinned bias-free, full-context reranker")
    AttentionInterface.register(PACKED_ATTENTION, packed_attention)
    with torch.inference_mode():
        weights = reader.model.lm_head.weight[reader.labels].contiguous()
        packed = deepcopy(reader.model.model).eval()
    packed.config._attn_implementation = PACKED_ATTENTION
    if reader.model.model.config._attn_implementation == PACKED_ATTENTION:
        raise AssertionError("Canonical and packed configurations must be independent")
    model_bytes = sum(param.numel() * param.element_size() for param in packed.parameters())
    chosen = sorted(select(dataset, ids, texts), key=lambda q: sha256(f"packed-load-v1:{q}".encode()).digest())[:REQUESTS]
    assert len(chosen) == REQUESTS

    def query_fn(arm, arrived, q, *, broker):
        return serve(arm, arrived, texts[q], texts, encoder, index, reader, weights, packed, broker)

    for arm in ARMS:
        replay(40, chosen[:6], arm, query_fn, packed, weights)
    results = {}
    canonical = {}
    for rate in RATES:
        trials = {arm: [] for arm in ARMS}
        sequences = []
        for repeat in range(REPEATS):
            sequence = list(chosen)
            random.Random(482 + repeat).shuffle(sequence)
            sequences.extend(sequence)
            offset = (RATES.index(rate) + repeat) % len(ARMS)
            for arm in ARMS[offset:] + ARMS[:offset]:
                trials[arm].append(replay(rate, sequence, arm, query_fn, packed, weights))
                print(f"{dataset}: {rate} req/s, {repeat + 1}/{REPEATS}, {arm}", flush=True)
        baseline = [out for trial in trials["padded4"] for out in trial["outputs"]]
        for query, response in zip(sequences, baseline, strict=True):
            if canonical.setdefault(query, response) != response:
                raise AssertionError("Padded baseline changes under load")
        results[rate] = {}
        for arm, samples in trials.items():
            outputs = [out for s in samples for out in s["outputs"]]
            results[rate][arm] = dict(
                exact=sum(a == b for a, b in zip(outputs, baseline, strict=True)),
                requests=len(outputs),
                p95_ms=1000 * float(np.quantile([ms for s in samples for ms in s["latency"]], .95)),
                worker_wait_p95_ms=1000 * float(np.quantile([ms for s in samples for ms in s["worker_wait"]], .95)),
                arrival_lag_max_ms=1000 * max(ms for s in samples for ms in s["arrival_lag"]),
                throughput=len(outputs) / sum(s["duration"] for s in samples),
                **{key: sum(s["work"].get(key, 0) for s in samples)
                   for key in ("dynamic_launches", "merged_launches", "dynamic_groups", "serial", "fixed")},
            )
    write_json(output, dict(
        dataset=dataset, job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), workers=WORKERS,
        extra_packed_backbone_bytes=model_bytes, samples=len(chosen), repeats=REPEATS,
        cohort_sha256=sha256("\n".join(chosen).encode()).hexdigest(), code_sha256=digest(Path(__file__)),
        vector_protocol_sha256=digest(root / "qwen-embedding-torch-v1/protocol.json"),
        embedding_revision=provenance["revision"], reranker_revision=reader.provenance["revision"], rates=results,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("KDD", "SciRepEval"))
    for name in ("embedding", "reranker", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
