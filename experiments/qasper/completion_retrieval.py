"""Measure semantic and lexical shortlists for one missing QASPER paragraph."""

import argparse
from hashlib import sha256
import os
from pathlib import Path
import random
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import paired_interval
from experiments.followir.core17_reranking import write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.qasper.adaptive_evidence import digest, evidence_sets, offline, query_input, read_data
from experiments.qasper.evidence_gap_audit import choose
from experiments.qasper.remaining_evidence import TASK, excerpt, holdout, shortlists, terms


BUDGETS = (4, 8)
METHODS = ("lexical", "gap", "dense", "anchored", "residual", "hybrid")
ANCHORED_TASK = (
    "Retrieve an additional passage that supplies evidence still needed to answer the question. "
    "The evidence in the query has already been found."
)


def dense_order(scores, candidates):
    return sorted(candidates, key=lambda i: (-scores[i], i))


def hybrid_order(a, b):
    second = {idx: 1 / (10 + rank) for rank, idx in enumerate(b, start=1)}
    weighted = [(1 / (10 + rank) + second[idx], idx) for rank, idx in enumerate(a, start=1)]
    return [idx for _, idx in sorted(weighted, key=lambda row: (-row[0], row[1]))]


def observe(encoder, reader, qa, pid, metadata, vectors, rng):
    paragraphs, sections, left, right = metadata
    ids = {paragraph: idx for idx, paragraph in enumerate(paragraphs)}
    refs = {frozenset(ids[p] for p in ref) for ref in evidence_sets(qa, paragraphs)}
    view = holdout(pid, refs)
    if view is None:
        raise ValueError("Expected an incomplete anchor set for every eligible question")
    shown, witnesses = view
    question = qa["question"]
    excerpts = "\n".join(excerpt(reader.tokenizer, paragraphs[i], 128) for i in shown)
    query = question + "\nEvidence already available:\n" + excerpts

    t0 = perf_counter()
    words = [set(terms(paragraph)) for paragraph in paragraphs]
    pools = shortlists(question, words, shown, include_rankings=True)
    candidates = pools["anchored_full"]
    lex_ms = (perf_counter() - t0) * 1000

    bank = vectors[left:right]
    encoding_ms = {}
    encoded = {}
    inputs = {
        "dense": query_input(question),
        "anchored": f"Instruct: {ANCHORED_TASK}\nQuery: {query}",
    }
    for name in rng.sample(list(inputs), len(inputs)):
        start = perf_counter()
        encoded[name] = encoder.encode([inputs[name]])[0]
        encoding_ms[name] = (perf_counter() - start) * 1000

    start = perf_counter()
    dense = dense_order(bank @ encoded["dense"], candidates)
    dense_ms = encoding_ms["dense"] + (perf_counter() - start) * 1000
    start = perf_counter()
    anchored = dense_order(bank @ encoded["anchored"], candidates)
    anchored_ms = encoding_ms["anchored"] + (perf_counter() - start) * 1000
    start = perf_counter()
    # Remove the query component explained by the centroid of already available passages.
    centroid = bank[shown].mean(axis=0)
    centroid /= max(float(np.linalg.norm(centroid)), 1e-9)
    projection = max(0.0, float(encoded["dense"] @ centroid))
    residual = dense_order(bank @ (encoded["dense"] - projection * centroid), candidates)
    residual_ms = encoding_ms["dense"] + (perf_counter() - start) * 1000
    start = perf_counter()
    hybrid = hybrid_order(anchored, pools["gap_ranked"])
    hybrid_ms = anchored_ms + lex_ms + (perf_counter() - start) * 1000

    orders = {"lexical": pools["question_ranked"], "gap": pools["gap_ranked"],
              "dense": dense, "anchored": anchored, "residual": residual, "hybrid": hybrid}
    retrieval_ms = {"lexical": lex_ms, "gap": lex_ms, "dense": dense_ms,
                    "anchored": anchored_ms, "residual": residual_ms, "hybrid": hybrid_ms}
    chunks = [f"{section}\n{paragraph}" for section, paragraph in zip(sections, paragraphs, strict=True)]
    result = {}
    arms = [(method, budget) for method in METHODS for budget in BUDGETS]
    for method, budget in rng.sample(arms, len(arms)):
        pool = orders[method][:budget]
        start = perf_counter()
        margins, clipped = reader.score_pairs([(query, chunks[i]) for i in pool], task=TASK)
        selected = max(zip(margins, pool, strict=True), key=lambda row: (row[0], -row[1]))[1]
        rerank_ms = (perf_counter() - start) * 1000
        result[method, budget] = dict(hit=selected in witnesses, reachable=bool(witnesses.intersection(pool)),
                                      size=len(pool), clipped=clipped, rerank_ms=rerank_ms,
                                      retrieval_ms=retrieval_ms[method])
    return result


def summarize(rows):
    result = {}
    for budget in BUDGETS:
        result[budget] = {}
        for method in METHODS:
            arm = [row[method, budget] for row in rows]
            reachable = sum(r["reachable"] for r in arm)
            hits = sum(r["hit"] for r in arm)
            contrasts = {}
            for baseline in ("gap", "dense"):
                refs = [row[baseline, budget] for row in rows]
                contrasts[baseline] = {
                    metric: paired_interval([int(a[metric]) - int(b[metric])
                                             for a, b in zip(arm, refs, strict=True)])
                    for metric in ("reachable", "hit")
                }
            result[budget][method] = dict(
                reachable=reachable, hit=hits, shortlist_misses=len(rows) - reachable,
                selection_misses=reachable - hits, mean_pairs=mean(r["size"] for r in arm),
                truncations=sum(r["clipped"] for r in arm),
                retrieval_p95_ms=float(np.quantile([r["retrieval_ms"] for r in arm], .95)),
                estimated_serial_p95_ms=float(np.quantile([r["retrieval_ms"] + r["rerank_ms"] for r in arm], .95)),
                contrasts=contrasts,
            )
    return result


def run(data, embedding, reranker, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    embed_info = verify_checkpoint(embedding)
    reader = Reranker(reranker)
    if (embed_info["repository"] != "Qwen/Qwen3-Embedding-0.6B"
            or reader.provenance["repository"] != "Qwen/Qwen3-Reranker-0.6B"):
        raise ValueError("Expected frozen Qwen3 checkpoints")
    encoder = QwenTorchEmbedding(embedding)
    encoder.model.config.use_cache = False
    reader.score_pairs([("Warm up", "A paragraph")] * 8, task=TASK)
    encoder.encode([query_input("Warm up")])
    rng = random.Random(493)
    splits = {split: read_data(data, split) for split in ("dev", "test")}
    if set(splits["dev"]).intersection(splits["test"]):
        raise ValueError("Expected disjoint QASPER papers")
    results = {}
    for split, papers in splits.items():
        panel = choose(papers, multi=True)
        meta, _, vectors, index = offline({pid: papers[pid] for pid, _ in panel}, encoder)
        print(f"{split}: indexed {len(panel)} papers", flush=True)
        rows = []
        for idx, (pid, qa) in enumerate(panel, start=1):
            rows.append(observe(encoder, reader, qa, pid, meta[pid], vectors, rng))
            if idx % 25 == 0:
                print(f"{split}: {idx}/{len(panel)}", flush=True)
        results[split] = dict(papers=len(rows), index=index, vector_bytes=vectors.nbytes,
                              panel_sha256=sha256("\n".join(qa["question_id"] for _, qa in panel).encode()).hexdigest(),
                              arms=summarize(rows))
    write_json(output, dict(dataset="QASPER v0.3", job=os.environ["SLURM_JOB_ID"],
                            gpu=torch.cuda.get_device_name(), code_sha256=digest(Path(__file__)),
                            embedding=embed_info["revision"], reranker=reader.provenance["revision"],
                            results=results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data", "embedding", "reranker", "output"):
        parser.add_argument(key, type=Path)
    run(**vars(parser.parse_args()))
