"""Validate that fixed-backend packed verification can share read-only CUDA parameters."""

import argparse
from concurrent.futures import Future
from copy import deepcopy
from functools import partial
from hashlib import sha256
import os
from pathlib import Path

import torch
from transformers import AttentionInterface

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.retrieval.cof_global_profile_pooling import ProfileIndex, read_data, select
from experiments.retrieval.cof_ragged_concurrent import REQUESTS, PackedBroker, replay, serve
from experiments.retrieval.cof_ragged_frontier import PACKED_ATTENTION, packed_attention, packed_forward
from experiments.retrieval.cof_verification_microbatch import Command


def packed_view(backbone, *, share_weights):
    memo = {id(param): param for param in backbone.parameters()} if share_weights else {}
    packed = deepcopy(backbone, memo).eval()
    packed.config._attn_implementation = PACKED_ATTENTION
    if packed.config is backbone.config or backbone.config._attn_implementation == PACKED_ATTENTION:
        raise AssertionError("Packed and canonical attention configurations must be independent")
    return packed


def measure_storage(original, copy):
    parameters = dict(original.named_parameters())
    extra = 0
    shared = 0
    for name, param in copy.named_parameters():
        size = param.numel() * param.element_size()
        if parameters[name].data_ptr() == param.data_ptr():
            shared += size
        else:
            extra += size
    return dict(shared_parameter_bytes=shared, extra_parameter_bytes=extra)


def verify_merged_broker(reader, model, weights):
    groups = [[{"input_ids": reader.tokenizer.encode(text, add_special_tokens=False)}]
              for text in ("Independent review of image retrieval.", "Evidence for interleaving two tasks.")]
    expected = [packed_forward(model, weights, group)[0] for group in groups]

    def forward(rows):
        values, _ = packed_forward(model, weights, rows)
        return values, max(len(row["input_ids"]) for row in rows)

    broker = PackedBroker(forward)
    futures = [Future() for _ in groups]
    for rows, future in zip(groups, futures, strict=True):
        broker.queue.put(Command("score", rows, future))
    with broker:
        received = [future.result(timeout=15)[0] for future in futures]
    if broker.work["merged_launches"] != 1:
        raise AssertionError("Expected actual cross-request coalescing")
    error = max(abs(a - b) for left, right in zip(expected, received, strict=True)
                for a, b in zip(left, right, strict=True))
    return dict(merged_launches=1, requests=2, max_absolute_margin_error=error)


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
        copied = packed_view(base, share_weights=False)
        shared = packed_view(base, share_weights=True)
    storage = {"copied": measure_storage(base, copied), "shared": measure_storage(base, shared)}
    isolation = verify_merged_broker(reader, shared, weights)
    if storage["shared"]["extra_parameter_bytes"] or storage["copied"]["shared_parameter_bytes"]:
        raise AssertionError("Unexpected parameter aliasing")
    chosen = sorted(select(dataset, ids, texts), key=lambda q: sha256(f"packed-load-v1:{q}".encode()).digest())[:REQUESTS]
    assert len(chosen) == REQUESTS

    def run_trace(arm, packed, sequence):
        def query_fn(method, arrived, q, *, broker):
            return serve(method, arrived, texts[q], texts, encoder, index, reader, weights, packed, broker)
        return replay(12, sequence, arm, query_fn, packed, weights)

    def matches(a, b):
        return sum(x == y for x, y in zip(a["outputs"], b["outputs"], strict=True))

    for mode, view in (("padded4", base), ("packed8_merge", copied), ("packed8_merge", shared)):
        run_trace(mode, view, chosen[:6])
    reference = run_trace("padded4", base, chosen)
    copy_result = run_trace("packed8_merge", copied, chosen)
    shared_result = run_trace("packed8_merge", shared, chosen)
    write_json(output, dict(
        dataset=dataset, job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        requests=len(chosen), offered_rps=12, storage=storage, forced_merge_probe=isolation,
        shared_vs_copied=matches(shared_result, copy_result), shared_vs_padded=matches(shared_result, reference),
        copied_vs_padded=matches(copy_result, reference),
        shared_merged_launches=shared_result["work"].get("merged_launches", 0),
        copied_merged_launches=copy_result["work"].get("merged_launches", 0),
        code_sha256=digest(Path(__file__)),
        vector_protocol_sha256=digest(root / "qwen-embedding-torch-v1/protocol.json"),
        embedding_revision=provenance["revision"], reranker_revision=reader.provenance["revision"],
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("KDD", "SciRepEval"))
    for name in ("embedding", "reranker", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
