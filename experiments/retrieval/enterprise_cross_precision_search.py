"""Run independent and guarded clause readers through the same live expert search."""

import argparse
from itertools import permutations
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.gpu_owned_vector_index import GpuOwnedVectorIndex
from expert_search.owned_vector_index import ExactOwnedVectorIndex
from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.retrieval.enterprise_cross_precision import evaluate as parse
from experiments.retrieval.enterprise_first_query import evaluate, read_rows, summarize
from experiments.retrieval.enterprise_gap_verification import DEVELOPMENT, SOURCE_INVENTORY
from experiments.retrieval.enterprise_guarded_parsing import GuardedReader


ARMS = ("independent", "bf16_guard", "consensus")


class Parser:
    def __init__(self, checkpoint):
        self.readers = {kind: GuardedReader(checkpoint) for kind in ("bf16", "fp16")}
        half = self.readers["fp16"]
        half.model.half()
        half.head = half.model.get_output_embeddings().weight[half.labels].contiguous()
        if half.head.dtype != torch.float16:
            raise ValueError("Invalid FP16 reader")

    @torch.inference_mode()
    def evaluate(self, text, clauses, method, cutoff):
        if cutoff != 2.0:
            raise ValueError("Unexpected qualification threshold")
        route = "bf16_independent" if method == "independent" else method
        result = parse(text, clauses, self.readers, route)
        return dict(labels=result["labels"], rechecked=result.get("rechecked", 0))


def run(queries, corpus, models, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a new GPU run")
    if (digest(queries), digest(corpus)) != (DEVELOPMENT, SOURCE_INVENTORY):
        raise ValueError("Changed development input")
    requests, records = read_rows(queries), read_rows(corpus)
    sources = {row["id"]: row for row in records}
    if len(requests) != 80 or len(sources) != len(records) or len(records) != 52667:
        raise ValueError("Unexpected cohort")
    torch.backends.cuda.matmul.allow_tf32 = False
    embedding_path = models / "qwen3-embedding-0.6b"
    checkpoint = verify_checkpoint(embedding_path)
    embedding = QwenTorchEmbedding(embedding_path)
    started = perf_counter()
    vectors = embedding.encode([row["text"] for row in records])
    reference = ExactOwnedVectorIndex(vectors, list(sources), [[row["expert_id"]] for row in records])
    index = GpuOwnedVectorIndex(reference)
    index_seconds = perf_counter() - started
    print(f"Indexed {len(sources)} sources", flush=True)
    reader = Parser(models / "qwen3-4b-instruct-2507")
    verifier = Reranker(models / "qwen3-reranker-0.6b")
    warmup = "Find an expert in optical inspection. Prior experience testing polymer films is required."
    for name in ARMS:
        evaluate(warmup, name, reader, embedding, index, sources, verifier)
    orders = tuple(permutations(ARMS))
    rows = []
    for i, request in enumerate(requests):
        order = orders[i % len(orders)]
        rows.append({name: evaluate(request["text"], name, reader, embedding, index, sources, verifier)
                     for name in order})
        if (i+1) % 10 == 0:
            print(i+1, "complete", flush=True)
    baseline = np.asarray([row["bf16_guard"]["times"]["total"] for row in rows])
    candidate = np.asarray([row["consensus"]["times"]["total"] for row in rows])
    rng = np.random.default_rng(41459)
    draws = rng.integers(len(rows), size=(4999, len(rows)))
    ratio = np.quantile(baseline[draws], .95, axis=1) / np.quantile(candidate[draws], .95, axis=1)
    write_json(output, dict(
        development_sha256=DEVELOPMENT, corpus_sha256=SOURCE_INVENTORY,
        script_sha256=digest(Path(__file__)), parser_sha256=digest(Path(__file__).with_name("enterprise_cross_precision.py")),
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), requests=len(rows), sources=len(sources),
        experts=len(reference.owners), index_seconds=index_seconds,
        model_revisions={"embedding":checkpoint["revision"], "verifier":verifier.provenance["revision"],
                         "parser":reader.readers["bf16"].provenance["revision"]},
        arms=summarize(rows, ARMS),
        bf16_guard_over_consensus_p95_ci95=np.quantile(ratio, [.025, .975]).tolist(),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("queries", "corpus", "models", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
