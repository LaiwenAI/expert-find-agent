"""Benchmark read-only prefix sharing during the complete enterprise first query."""

import argparse
from copy import copy
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn.functional import linear
from transformers.cache_utils import DynamicCache, DynamicLayer

from expert_search.gpu_owned_vector_index import GpuOwnedVectorIndex
from expert_search.owned_vector_index import ExactOwnedVectorIndex
from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.retrieval.enterprise_cross_precision_search import Parser
from experiments.retrieval.enterprise_first_query import evaluate, read_rows
from experiments.retrieval.enterprise_gap_verification import DEVELOPMENT, SOURCE_INVENTORY
from experiments.retrieval.enterprise_guarded_parsing import GuardedReader
from experiments.retrieval.enterprise_isolated_parsing import padded, spans


ARMS = ("reference", "broadcast_prefix")


def fork_prefix(prefix, rows):
    # A dynamic full-attention layer rebinds keys/values on append, leaving its
    # input storage untouched. Fork metadata before exposing broadcast views.
    if type(prefix) is not DynamicCache or prefix.offloading:
        raise ValueError("Expected a resident dynamic prefix cache")
    fork = copy(prefix)
    fork.layers = [copy(layer) for layer in prefix.layers]
    for layer in fork.layers:
        if type(layer) is not DynamicLayer or not layer.is_initialized or layer.keys.shape[0] != 1:
            raise ValueError("Expected initialized singleton full-attention layers")
        layer.keys = layer.keys.expand(rows, -1, -1, -1)
        layer.values = layer.values.expand(rows, -1, -1, -1)
    return fork


class ForkReader(GuardedReader):
    broadcast = False
    prefix_bytes = 0
    forks = 0

    def cached(self, front, suffixes, prefix_cache):
        if not self.broadcast:
            return super().cached(front, suffixes, prefix_cache)
        cache = fork_prefix(prefix_cache, len(suffixes))
        self.forks += 1
        self.prefix_bytes += sum(
            (layer.keys.numel() + layer.values.numel()) * layer.keys.element_size()
            for layer in cache.layers
        )
        ids, mask, positions = padded(suffixes, self.tokenizer.pad_token_id, len(front))
        prefix_mask = torch.ones((len(suffixes), len(front)), dtype=torch.bool, device="cuda")
        hidden = self.backbone(input_ids=ids, attention_mask=torch.cat((prefix_mask, mask), 1),
                               position_ids=positions, past_key_values=cache, use_cache=True)
        if prefix_cache.get_seq_length() != len(front):
            raise AssertionError("Shared prefix cache was changed")
        return linear(hidden.last_hidden_state[:, -1].contiguous(), self.head), ids.numel()


class ComparisonParser(Parser):
    def __init__(self, checkpoint):
        self.readers = {kind: ForkReader(checkpoint) for kind in ("bf16", "fp16")}
        half = self.readers["fp16"]
        half.model.half()
        half.head = half.model.get_output_embeddings().weight[half.labels].contiguous()
        if half.head.dtype != torch.float16:
            raise ValueError("Unexpected FP16 reader")

    def evaluate(self, text, clauses, method, cutoff):
        for reader in self.readers.values():
            reader.broadcast = method == "broadcast_prefix"
            reader.forks = reader.prefix_bytes = 0
        return super().evaluate(text, clauses, "consensus", cutoff)

    def counters(self):
        return {key: sum(getattr(reader, key) for reader in self.readers.values())
                for key in ("forks", "prefix_bytes")}


def summary(records):
    changes = dict(
        labels=sum(not np.array_equal(a["labels"], b["labels"]) for a, b in records),
        qualifications=sum(a["qualifications"] != b["qualifications"] for a, b in records),
        retrieval=sum(a["retrieved"] != b["retrieved"] for a, b in records),
        decisions=sum(a["decisions"] != b["decisions"] for a, b in records),
        fallback_count=sum(a["rechecked"] != b["rechecked"] for a, b in records),
    )
    arms = {}
    for i, arm in enumerate(ARMS):
        rows = [pair[i] for pair in records]
        arms[arm] = dict(
            **{name: sum(row[name] for row in rows) for name in ("supported", "pairs", "rechecked",
                                                                "forks", "prefix_bytes")},
            p50_p95_ms={key: (1000 * np.quantile([row["times"][key] for row in rows], [.5, .95])).tolist()
                        for key in rows[0]["times"]},
            transient_peak_p95_bytes=float(np.quantile([row["peak_bytes"] for row in rows], .95)),
        )
    rng = np.random.default_rng(1819)
    samples = rng.integers(len(records), size=(4999, len(records)))
    ref, fork = (np.array([pair[i]["times"]["total"] for pair in records]) for i in range(2))
    diffs = np.quantile(fork[samples], .95, axis=1) - np.quantile(ref[samples], .95, axis=1)
    return dict(changed_requests=changes, arms=arms, delta_total_p95_ms_ci95=(1000*np.quantile(diffs, [.025, .975])).tolist())


def run(queries, corpus, models, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a fresh output and Slurm CUDA")
    if (digest(queries), digest(corpus)) != (DEVELOPMENT, SOURCE_INVENTORY):
        raise ValueError("Use the pinned development corpus only")
    requests, documents = read_rows(queries), read_rows(corpus)
    sources = {row["id"]: row for row in documents}
    if len(requests) != 80 or len(sources) != 52667 or len(sources) != len(documents):
        raise ValueError("Unexpected request or source inventory")
    torch.backends.cuda.matmul.allow_tf32 = False
    model_path = models / "qwen3-embedding-0.6b"
    provenance = verify_checkpoint(model_path)
    embedding = QwenTorchEmbedding(model_path)
    parser = ComparisonParser(models / "qwen3-4b-instruct-2507")
    diagnostic = "Find an expert in optical inspection. Prior experience testing polymer films is required."
    clauses = spans(diagnostic)
    traces = [parser.evaluate(diagnostic, clauses, arm, 2.0) for arm in ARMS]
    if not np.array_equal(traces[0]["labels"], traces[1]["labels"]) or not parser.counters()["forks"]:
        raise AssertionError("Warm startup did not reproduce the reference output through a real fork")
    print("Parser cache fork checked; embedding pinned development inventory.", flush=True)
    started = perf_counter()
    vectors = embedding.encode([row["text"] for row in documents])
    reference = ExactOwnedVectorIndex(vectors, list(sources), [[row["expert_id"]] for row in documents])
    index = GpuOwnedVectorIndex(reference)
    index_seconds = perf_counter() - started
    verifier = Reranker(models / "qwen3-reranker-0.6b")
    for arm in ARMS:
        evaluate(diagnostic, arm, parser, embedding, index, sources, verifier)
    rows = []
    for i, request in enumerate(requests):
        result = {}
        for arm in (ARMS if i % 2 == 0 else ARMS[::-1]):
            torch.cuda.reset_peak_memory_stats()
            resident = torch.cuda.memory_allocated()
            row = evaluate(request["text"], arm, parser, embedding, index, sources, verifier)
            result[arm] = dict(row, **parser.counters(), peak_bytes=torch.cuda.max_memory_allocated() - resident)
        rows.append(tuple(result[arm] for arm in ARMS))
        if (i + 1) % 10 == 0:
            print(i + 1, "development requests evaluated", flush=True)
    write_json(output, dict(
        development_sha256=DEVELOPMENT, corpus_sha256=SOURCE_INVENTORY, code_sha256=digest(Path(__file__)),
        requests=len(rows), sources=len(sources), experts=len(reference.owners), index_seconds=index_seconds,
        tag=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        checkpoints=dict(embedding=provenance["revision"], verifier=verifier.provenance["revision"],
                         parser=parser.readers["bf16"].provenance["revision"]),
        **summary(rows),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("queries", "corpus", "models", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
