"""Test shared-prefix execution and bounded priority under loopback search load."""

import argparse
from collections import deque
import json
import os
from pathlib import Path
from queue import Queue
from time import perf_counter

import torch

from expert_search.gpu_owned_vector_index import GpuOwnedVectorIndex
from expert_search.owned_vector_index import ExactOwnedVectorIndex
from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.retrieval.enterprise_cross_precision_load import trace, workload
from experiments.retrieval.enterprise_first_query import evaluate, read_rows
from experiments.retrieval.enterprise_gap_verification import DEVELOPMENT, SOURCE_INVENTORY
from experiments.retrieval.enterprise_prefix_fork import ComparisonParser


ARMS = ("copy_fifo", "broadcast_fifo", "broadcast_aged_length")
ORDERS = (ARMS, ARMS[::-1])
SEEDS = (20261119, 20261120)
RATE = 3.0
PRIORITY_SECONDS = .75
REFERENCE = "f9faebee50953ebf775e8dfaeaa13971d7c79b5fc6e8fe27e66f99bc32993fdd"


class AgedLengthQueue(Queue):
    def _get(self):
        pending: deque = self.queue
        oldest = pending[0]
        if oldest is None or perf_counter() - oldest[1] >= PRIORITY_SECONDS:
            return pending.popleft()
        # Queue's get() already holds the mutex. Ties preserve arrival order.
        winner = min((i for i, item in enumerate(pending) if item is not None),
                     key=lambda i: len(pending[i][0]))
        selected = pending[winner]
        del pending[winner]
        return selected


def run(queries, corpus, models, reference_result, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output in a Slurm CUDA allocation")
    if (digest(queries), digest(corpus), digest(reference_result)) != (DEVELOPMENT, SOURCE_INVENTORY, REFERENCE):
        raise ValueError("Use the frozen development data and serial first-query reference")
    pinned = json.loads(reference_result.read_bytes())["checkpoints"]
    requests, records = read_rows(queries), read_rows(corpus)
    texts = [row["text"] for row in requests]
    sources = {row["id"]: row for row in records}
    if len(requests) != 80 or len(sources) != 52667 or len(records) != len(sources):
        raise ValueError("Unexpected cohort")
    torch.backends.cuda.matmul.allow_tf32 = False
    path = models / "qwen3-embedding-0.6b"
    checkpoint = verify_checkpoint(path)
    embedding = QwenTorchEmbedding(path)
    started = perf_counter()
    data = embedding.encode([row["text"] for row in records])
    reference = ExactOwnedVectorIndex(data, list(sources), [[row["expert_id"]] for row in records])
    index = GpuOwnedVectorIndex(reference)
    index_seconds = perf_counter() - started
    print(f"Indexed {len(sources)} source records in {index_seconds:.1f}s", flush=True)
    parser = ComparisonParser(models / "qwen3-4b-instruct-2507")
    verifier = Reranker(models / "qwen3-reranker-0.6b")
    models_used = dict(embedding=checkpoint["revision"], verifier=verifier.provenance["revision"],
                       parser=parser.readers["bf16"].provenance["revision"])
    if models_used != pinned:
        raise ValueError("Model revisions changed")
    longest = set(sorted(range(len(texts)), key=lambda i: (-len(texts[i]), i))[:len(texts)//4])
    strata = dict(longest_quartile=longest, remaining=set(range(len(texts))) - longest)
    traces = []
    for seed, order in zip(SEEDS, ORDERS, strict=True):
        indices, times = trace(len(texts), RATE, seed)
        responses, report = {}, {}
        for arm in order:
            variant = "reference" if arm == "copy_fifo" else "broadcast_prefix"
            search = lambda text: evaluate(text, variant, parser, embedding, index, sources, verifier)
            selector = AgedLengthQueue if arm == "broadcast_aged_length" else Queue
            responses[arm], report[arm] = workload(texts, indices, times, search,
                                                  pending_factory=selector, strata=strata)
            if report[arm]["completed"] != len(texts):
                raise ValueError(f"Incomplete trace for {arm}")
            print(seed, arm, len(responses[arm]), "completed", flush=True)
        mismatches = {arm: sum(a != b for a, b in zip(responses[ARMS[0]], responses[arm], strict=True))
                      for arm in ARMS}
        traces.append(dict(seed=seed, nominal_rate=RATE, execution_order=order, mismatches=mismatches, arms=report))
    write_json(output, dict(
        requests=len(requests), sources=len(sources), experts=len(reference.owners), index_seconds=index_seconds,
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), code_sha256=digest(Path(__file__)),
        checkpoints=models_used, priority_seconds=PRIORITY_SECONDS, traces=traces,
        reference_sha256=digest(reference_result),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("queries", "corpus", "models", "reference_result", "output"):
        parser.add_argument(arg, type=Path)
    run(**vars(parser.parse_args()))
