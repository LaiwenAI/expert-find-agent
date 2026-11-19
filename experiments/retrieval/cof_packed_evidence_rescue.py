"""Spend packed-verifier savings on source expansion at fixed expert ranks."""

import argparse
from hashlib import sha256
import os
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import torch
from transformers import AttentionInterface

from expert_search.deferred_verification import resolve_shared_frontier
from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import paired_interval
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.retrieval.cof_binary_head import BinaryWaveReader
from experiments.retrieval.cof_frontier_wave import SUPPORT, WaveScheduler
from experiments.retrieval.cof_global_profile_pooling import ProfileIndex, read_data, select
from experiments.retrieval.cof_live_certified_batching import GUARD_MARGIN, TOP
from experiments.retrieval.cof_ragged_frontier import PACKED_ATTENTION, packed_attention, packed_forward
from experiments.retrieval.cof_ragged_shared_weights import measure_storage, packed_view


ARMS = ("padded3", "packed3", "packed4", "packed_rescue")


class SharedPackedReader(BinaryWaveReader):
    def __init__(self, reader, query, texts, universe, weights, packed):
        self.packed = packed
        super().__init__(reader, query, texts, universe, "two_rows_batched_tokenization", weights)

    def _forward(self, batch, dynamic):
        if not dynamic or self.packed is None or len(batch) == 1:
            return super()._forward(batch, dynamic)
        values, tokens = packed_forward(self.packed, self.weight, batch)
        self.padded_tokens += tokens
        return values


def certified_support(wave, docs):
    count = 0
    for doc in docs:
        if doc in wave.recheck_cache:
            count += wave.recheck_cache[doc] >= 0
        else:
            count += wave.score_cache.get(doc, 0) > GUARD_MARGIN
    return count


def resolve(candidates, wave, width):
    scheduler = WaveScheduler(wave, candidates, width, False)
    result = resolve_shared_frontier(
        candidates, scheduler.score, scheduler.recheck, wave.length,
        support=SUPPORT, top=TOP, width=width, band=GUARD_MARGIN, cost_aware=False,
    )
    # Restore fixed retrieval order after verification. This compares only attachment.
    citations = dict(result["selected"])
    return [[expert, citations[expert]] for expert, _ in candidates]


def verify(reader, packed, weights, candidates, texts, query, truth, arm):
    begin = perf_counter()
    initial = [(expert, docs[:3]) for expert, docs in candidates]
    padded = arm == "padded3"
    wave = SharedPackedReader(reader, query, texts, {doc for _, docs in candidates for doc in docs},
                              weights, None if padded else packed)
    width = 4 if padded else 8
    n_rescues = 0
    if arm == "packed4":
        final = resolve(candidates, wave, width)
    else:
        final = resolve(initial, wave, width)
        if arm == "packed_rescue":
            selected = dict(final)
            extra = {
                expert for expert, docs in candidates
                if len(docs) > 3 and not selected[expert] and certified_support(wave, docs[:3]) == 1
            }
            n_rescues = len(extra)
            if extra:
                expanded = [(expert, docs if expert in extra else docs[:3]) for expert, docs in candidates]
                final = resolve(expanded, wave, width)
                if any(old_docs and old_docs != new_docs for (_, old_docs), (_, new_docs) in zip(
                        selected.items(), final, strict=True)):
                    raise AssertionError("Rescue replaced already attached citations")
    return dict(
        ms=(perf_counter() - begin) * 1000, outputs=final, rescues=n_rescues,
        cited=sum(bool(docs) for _, docs in final),
        cited_positives=sum(truth.get(expert, 0) > 0 and bool(docs) for expert, docs in final),
        cited_negatives=sum(expert in truth and truth[expert] <= 0 and bool(docs) for expert, docs in final),
        cited_unknown=sum(expert not in truth and bool(docs) for expert, docs in final),
        scored_docs=wave.scored_documents, forwarded=wave.score_calls + wave.recheck_calls,
    )


def summarize(rows):
    out = {}
    for arm in ARMS:
        current = [row[arm] for row in rows]
        paired = {}
        for base in ("padded3", "packed3", "packed4"):
            reference = [row[base] for row in rows]
            paired[base] = {field: paired_interval([a[field] - b[field]
                                                    for a, b in zip(current, reference, strict=True)])
                            for field in ("ms", "cited_positives")}
            paired[base]["exact"] = sum(a["outputs"] == b["outputs"]
                                        for a, b in zip(current, reference, strict=True))
        out[arm] = dict(
            **{field: sum(r[field] for r in current)
               for field in ("cited", "cited_positives", "cited_negatives", "cited_unknown", "rescues")},
            mean_scored_docs=mean(r["scored_docs"] for r in current),
            mean_forwards=mean(r["forwarded"] for r in current),
            verifier_p95_ms=float(np.quantile([r["ms"] for r in current], .95)),
            complete_p95_ms=float(np.quantile([r["ms"] + rows[i]["retrieve_ms"]
                                               for i, r in enumerate(current)], .95)),
            contrasts=paired,
        )
    return out


def run(dataset, embedding, reranker, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    provenance = verify_checkpoint(embedding)
    root, papers, _, reference, ids, _, labels = read_data(dataset, provenance)
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
    if storage["extra_parameter_bytes"]:
        raise AssertionError("Expected shared parameter storage")
    chosen = select(dataset, ids, texts)
    warm = next(q for q in ids if q not in chosen)
    rows = []
    for i, q in enumerate([warm, *chosen]):
        began = perf_counter()
        candidates = index.search_query(texts[q], encoder, "mean_max", depth=4)[:TOP]
        latency = (perf_counter() - began) * 1000
        offset = i % len(ARMS)
        result = {
            arm: verify(reader, packed, weights, candidates, texts, texts[q], labels[q], arm)
            for arm in ARMS[offset:] + ARMS[:offset]
        }
        if i:
            rows.append(dict(result, retrieve_ms=latency))
        if i and i % 25 == 0:
            print(f"{dataset}: {i}/{len(chosen)}", flush=True)
    write_json(output, dict(
        dataset=dataset, job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), queries=len(chosen),
        shared_weights=storage, cohort_sha256=sha256("\n".join(chosen).encode()).hexdigest(),
        code_sha256=digest(Path(__file__)), vector_protocol_sha256=digest(root / "qwen-embedding-torch-v1/protocol.json"),
        embedding_revision=provenance["revision"], reranker_revision=reader.provenance["revision"], arms=summarize(rows),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("KDD", "SciRepEval"))
    for name in ("embedding", "reranker", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
