"""Reuse only fixed instructions in the live enterprise qualification parser."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.gpu_owned_vector_index import GpuOwnedVectorIndex
from expert_search.owned_vector_index import ExactOwnedVectorIndex
from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.retrieval import enterprise_shared_qualification as search
from experiments.retrieval.enterprise_cross_precision import evaluate, prefix_cache
from experiments.retrieval.enterprise_gap_verification import DEVELOPMENT, SOURCE_INVENTORY
from experiments.retrieval.enterprise_isolated_parsing import shared_prompt, spans
from experiments.retrieval.enterprise_prefix_fork import ComparisonParser, fork_prefix
from experiments.retrieval.enterprise_routed_review import digest, rows, write


ARMS = ("request_prefill", "static_prefix")
PREAMBLE_PROBES = ("1 Probe.", "Z Example.", "甲示例。", "? Other.")


class StaticParser(ComparisonParser):
    @torch.inference_mode()
    def __init__(self, checkpoint):
        super().__init__(checkpoint)
        tokenizer = self.readers["bf16"].tokenizer
        fronts = [shared_prompt(tokenizer, text, spans(text))[0] for text in PREAMBLE_PROBES]
        common = []
        for column in zip(*fronts):
            if len(set(column)) != 1:
                break
            common.append(column[0])
        if not common or any(common == front for front in fronts):
            raise ValueError("Need a nonempty static instructional prefix")
        self.fixed = common
        start = perf_counter()
        self.caches = {reader: prefix_cache(reader, self.fixed) for reader in self.readers.values()}
        torch.cuda.synchronize()
        self.startup_ms = (perf_counter() - start) * 1000
        self.mode = "request_prefill"

    def prefill(self, reader, front):
        if self.mode != "static_prefix" or len(front) <= len(self.fixed) or front[:len(self.fixed)] != self.fixed:
            if self.mode == "static_prefix":
                self.run_counts["prefix_miss"] += 1
            return prefix_cache(reader, front)
        cache = self.caches[reader]
        child = fork_prefix(cache, 1)
        remaining = torch.tensor([front[len(self.fixed):]], device="cuda")
        result = reader.backbone(input_ids=remaining, past_key_values=child, use_cache=True).past_key_values
        if result.get_seq_length() != len(front) or cache.get_seq_length() != len(self.fixed):
            raise ValueError("Request cache was incomplete or mutated its static prefix")
        self.run_counts["prefix_hits"] += 1
        self.run_counts["prefix_tokens_avoided"] += len(self.fixed)
        return result

    @torch.inference_mode()
    def evaluate(self, text, clauses, method, cutoff):
        if method != "broadcast_prefix" or cutoff != 2.:
            raise ValueError("Unexpected enterprise parser contract")
        start = perf_counter()
        for reader in self.readers.values():
            reader.broadcast = True
            reader.forks = reader.prefix_bytes = 0
        self.run_counts = Counter()
        result = evaluate(text, clauses, self.readers, "consensus", prefill=self.prefill)
        self.run_labels = tuple(map(int, result["labels"]))
        self.run_counts.update(
            rechecked=result.get("rechecked", 0), fp16_checks=result.get("fp16_block_checks", 0))
        self.ms = (perf_counter() - start) * 1000
        return dict(labels=result["labels"], rechecked=result.get("rechecked", 0))


def request(text, arm, fixture):
    parser = fixture[0]
    parser.mode = arm
    result = search.one(text, "automatic", *fixture)
    return dict(result, labels=parser.run_labels, parser_ms=parser.ms, parser_counts=dict(parser.run_counts))


def summarize(observations):
    reference = [r["request_prefill"] for r in observations]
    reference_time = np.asarray([row["elapsed_ms"] for row in reference])
    draws = np.random.default_rng(20261123).integers(len(reference), size=(6000, len(reference)))
    report = {}
    for arm in ARMS:
        current = [r[arm] for r in observations]
        elapsed = np.asarray([r["elapsed_ms"] for r in current])
        delta = np.quantile(elapsed[draws], .95, axis=1) - np.quantile(reference_time[draws], .95, axis=1)
        report[arm] = dict(
            queries=len(current),
            issued=sum(bool(row["initial"] or row["extra"]) for row in current),
            parser_counts=dict(sum((Counter(r["parser_counts"]) for r in current), Counter())),
            parser_p50_p95_ms=np.quantile([r["parser_ms"] for r in current], [.50, .95]).tolist(),
            live_p50_p95_ms=np.quantile(elapsed, [.50, .95]).tolist(),
            paired_live_p95_delta_ci95=np.quantile(delta, [.025,.975]).tolist(),
            changes={key: sum(a[key] != b[key] for a,b in zip(current, reference, strict=True))
                     for key in ("labels", "qualifications", "retrieved", "ranked", "decisions", "initial", "extra", "invoked")},
            recheck_count_changes=sum(r["parser_counts"].get("rechecked", 0) != b["parser_counts"].get("rechecked", 0)
                                      for r,b in zip(current, reference, strict=True)))
    return report


def run(queries, corpus, roles, vectors, metadata, models, output):
    search.role.require_cuda(output)
    if (digest(queries), digest(corpus)) != (DEVELOPMENT, SOURCE_INVENTORY):
        raise ValueError("Expected the pinned development-only data")
    torch.backends.cuda.matmul.allow_tf32 = False
    sources = {row["id"]: row for row in rows(corpus)}
    meta = json.loads(metadata.read_bytes())
    embedding_path = models / "qwen3-embedding-0.6b"
    provenance = verify_checkpoint(embedding_path)
    if (SOURCE_INVENTORY, provenance["revision"], digest(vectors), len(sources)) != tuple(
            meta[key] for key in ("corpus_sha256", "checkpoint", "vectors_sha256", "rows")):
        raise ValueError("Cached index provenance differs")
    index = GpuOwnedVectorIndex(ExactOwnedVectorIndex(
        np.load(vectors, allow_pickle=False), list(sources), [[r["expert_id"]] for r in sources.values()]))
    parser = StaticParser(models / "qwen3-4b-instruct-2507")
    verifier = search.EncodedReader(Reranker(models / "qwen3-reranker-0.6b"))
    fixture = (parser, QwenTorchEmbedding(embedding_path), index, sources,
               search.role.load_role_index(roles, SOURCE_INVENTORY, sources), verifier)
    for arm in ARMS:
        request("Find a specialist in optical calibration. Prior sensor testing is required.", arm, fixture)
    observations = []
    for i, row in enumerate(rows(queries)):
        observations.append({arm: request(row["text"], arm, fixture) for arm in (ARMS if i % 2 == 0 else ARMS[::-1])})
        if (i + 1) % 20 == 0:
            print(f"Completed {i+1} private development requests", flush=True)
    resident_bytes = sum((layer.keys.numel() + layer.values.numel()) * layer.keys.element_size()
                         for cache in parser.caches.values() for layer in cache.layers)
    write(output, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
                       code_sha256=digest(Path(__file__)), static_prefix_tokens=len(parser.fixed),
                       static_startup_ms=parser.startup_ms, static_resident_bytes=resident_bytes,
                       metrics=summarize(observations)))


if __name__ == "__main__":
    arg = argparse.ArgumentParser(description=__doc__)
    for name in ("queries", "corpus", "roles", "vectors", "metadata", "models", "output"):
        arg.add_argument(name, type=Path)
    run(**vars(arg.parse_args()))
