"""Audit global CoF retrieval, evidence promotion, and profile-size calibration."""

import argparse
from hashlib import sha256
import os
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from expert_search.deferred_verification import defer_and_rank, resolve_shared_frontier
from expert_search.gpu_owned_vector_index import GpuOwnedVectorIndex
from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import paired_interval
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.retrieval.cof_compact_profile_shortlist import read_data
from experiments.retrieval.cof_live_certified_batching import GUARD_MARGIN, POOL, TOP, WIDTH, WaveReader
from experiments.retrieval.cof_shared_frontier import panel

ARMS = ("dense_only", "dense_promote_two", "dense_attach_two", "mean_attach_two", "calibrated_attach_two")


def empirical_tail(scores):
    _, indices, counts = np.unique(-scores, return_inverse=True, return_counts=True)
    # Give tied scores the same conservative tail rank.
    return np.cumsum(counts)[indices] / (len(scores) + 1)


def best_tail_probability(tail, counts):
    """Adjust up to three ordered source matches for the size of each profile."""
    best = np.ones(len(counts))
    for col in range(tail.shape[1]):
        valid = counts > col
        prob = tail[valid, col]
        size = counts[valid]
        log_off = np.log1p(-prob)
        value = -np.expm1(size * log_off)
        mass = np.exp(size * log_off)
        for j in range(1, col + 1):
            mass *= (size - j + 1) * (prob / (j * (1 - prob)))
            value -= mass
        best[valid] = np.minimum(best[valid], np.maximum(value, 1e-15))
    return best


class CandidateIndex:
    def __init__(self, reference):
        self.gpu = GpuOwnedVectorIndex(reference)
        self.reference = reference
        self.group = np.repeat(np.arange(len(reference.owner_ids)), np.diff(
            np.append(reference.owner_starts, len(reference.owner_rows))))
        self.counts = np.bincount(self.group)

    def search(self, vector, route):
        if route == "dense":
            hits = self.gpu.search_vector(vector, k=POOL)["hits"]
            return [(row["id"], [e["id"] for e in row["evidence"]]) for row in hits]
        scores = torch.mv(self.gpu.vectors, torch.as_tensor(vector, device="cuda")).cpu().numpy()
        reference = self.reference
        indices = reference.owner_rows
        ranked = indices[np.lexsort((indices, -scores[indices], self.group))]
        columns = np.arange(3)
        mask = columns < self.counts[:, None]
        positions = np.minimum(columns, self.counts[:, None] - 1) + reference.owner_starts[:, None]
        sources = ranked[positions]
        if route == "mean":
            owner_score = -np.sum(np.where(mask, scores[sources], 0), axis=1) / mask.sum(axis=1)
        else:
            owner_score = best_tail_probability(empirical_tail(scores)[sources], self.counts)
        tie_break = -scores[sources[:, 0]]
        best = np.lexsort((np.arange(len(owner_score)), tie_break, owner_score))[:POOL]
        return [(reference.owner_ids[e], [reference.source_ids[j] for j in sources[e, mask[e]]]) for e in best]


def run_request(index, encoder, verifier, texts, query, truth, arm):
    start = perf_counter()
    vector = encoder.encode([query], query=True)[0]
    route = arm.split("_", 1)[0]
    hits = index.search(vector, route)
    retrieved = perf_counter()
    calls = 0
    if arm == "dense_only":
        final = [(expert, []) for expert, _ in hits[:TOP]]
    else:
        pool = hits if arm == "dense_promote_two" else hits[:TOP]
        wave = WaveReader(verifier, query, texts, {doc for _, docs in pool for doc in docs},
                          dynamic=True, cache=True)
        if arm == "dense_promote_two":
            final = defer_and_rank(pool, wave.score, wave.recheck, support=2, top=TOP,
                                   width=WIDTH, band=GUARD_MARGIN)["selected"]
        else:
            assessed = resolve_shared_frontier(
                pool, wave.score, wave.recheck, wave.length, support=2, top=TOP, width=WIDTH,
                band=GUARD_MARGIN, cost_aware=False)["selected"]
            citations = dict(assessed)
            final = [(expert, citations[expert]) for expert, _ in pool]
        calls = wave.score_calls + wave.recheck_calls
    end = perf_counter()
    positives = {expert for expert, score in truth.items() if score > 0}
    judged = set(truth)
    candidate_ids = {expert for expert, _ in hits}
    returned = {expert for expert, _ in final}
    promoted_out = {expert for expert, _ in hits[:TOP]} - returned
    promoted_in = returned - {expert for expert, _ in hits[:TOP]}
    return dict(ms=(end-start)*1000, retrieval_ms=(retrieved-start)*1000,
                positive_candidates=len(positives & candidate_ids),
                all_positives=len(positives), candidate_hit=int(bool(positives & candidate_ids)),
                oracle_p5=min(len(positives & candidate_ids), TOP) / TOP,
                p5=len(positives & returned) / TOP, judged_p5=len(judged & returned) / TOP,
                cited_p5=sum(expert in positives and bool(docs) for expert, docs in final) / TOP,
                citation_coverage=sum(bool(docs) for _, docs in final) / TOP, calls=calls,
                positives_added=len(promoted_in & positives), positives_removed=len(promoted_out & positives))


def summary(rows):
    report = {}
    for arm in ARMS:
        values = [row[arm] for row in rows]
        metrics = dict(p50_p95_ms=np.quantile([v["ms"] for v in values], [.5, .95]).tolist(),
                       retrieve_p95_ms=float(np.quantile([v["retrieval_ms"] for v in values], .95)))
        for key in ("candidate_hit", "oracle_p5", "p5", "judged_p5", "cited_p5", "citation_coverage", "calls"):
            metrics[key] = mean(row[key] for row in values)
        for key in ("positives_added", "positives_removed"):
            metrics[key] = sum(row[key] for row in values)
        metrics["micro_candidate_recall"] = (
            sum(row["positive_candidates"] for row in values) / sum(row["all_positives"] for row in values))
        metrics["p5_delta_vs_dense_only"] = paired_interval([row[arm]["p5"] - row["dense_only"]["p5"] for row in rows])
        report[arm] = metrics
    return report


def run(args):
    if args.output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a fresh output path and Slurm CUDA")
    provenance = verify_checkpoint(args.embedding)
    torch.backends.cuda.matmul.allow_tf32 = False
    root, papers, _, reference, ids, _, labels = read_data("KDD", provenance)
    texts = {pid: (paper.get("title", "") + "\n" + paper.get("abstract", "")).strip()
             for pid, paper in papers.items()}
    index = CandidateIndex(reference)
    encoder = QwenTorchEmbedding(args.embedding, batch_size=1)
    encoder.model.config.use_cache = False
    verifier = Reranker(args.reranker)
    chosen = panel(ids, texts)
    for arm in ARMS:
        qid = next(qid for qid in ids if qid not in chosen)
        run_request(index, encoder, verifier, texts, texts[qid], labels[qid], arm)
    rows = []
    for i, qid in enumerate(chosen):
        offset = i % len(ARMS)
        rows.append({arm: run_request(index, encoder, verifier, texts, texts[qid], labels[qid], arm)
                     for arm in ARMS[offset:] + ARMS[:offset]})
        if (i + 1) % 30 == 0:
            print(f"Completed {i + 1} requests", flush=True)
    write_json(args.output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), queries=len(rows),
        experts=len(reference.owner_ids), sources=len(reference.source_ids), shortlist=POOL,
        cohort_sha256=sha256("\n".join(chosen).encode()).hexdigest(),
        embedding_revision=provenance["revision"], reranker_revision=verifier.provenance["revision"],
        script_sha256=digest(Path(__file__)),
        inputs={name: digest(root / name) for name in ("qwen-embedding-torch-v1/protocol.json",
                                                      "KDD_queries_test_hard.json")},
        arms=summary(rows),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("embedding", "reranker", "output"):
        parser.add_argument(name, type=Path)
    run(parser.parse_args())
