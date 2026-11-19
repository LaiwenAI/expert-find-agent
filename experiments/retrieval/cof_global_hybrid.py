"""Compare live lexical and dense retrieval over the full CoF KDD reviewer pool."""

import argparse
from hashlib import sha256
import os
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from expert_search.array_bm25 import ArrayBM25
from expert_search.deferred_verification import resolve_shared_frontier
from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import paired_interval
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.retrieval.cof_compact_profile_shortlist import read_data
from experiments.retrieval.cof_global_candidate_audit import CandidateIndex
from experiments.retrieval.cof_live_certified_batching import GUARD_MARGIN, POOL, TOP, WIDTH, WaveReader
from experiments.retrieval.cof_shared_frontier import panel

ARMS = ("dense", "sparse", "source_fusion", "expert_fusion", "union")
RRF_CONSTANT = 60
DEFERRED_ARMS = ("dense", "union", "deferred_union")


def score_reciprocals(values, active=None):
    inverse = np.unique(-values, return_inverse=True, return_counts=True)[1:]
    bins, counts = inverse
    ranks = np.cumsum(counts) - counts + 1
    reciprocal = 1 / (RRF_CONSTANT + ranks[bins])
    return reciprocal if active is None else np.where(active, reciprocal, 0)


def selected_reciprocals(values, rows, active=None):
    """Evaluate full-inventory competition ranks only at requested source rows."""
    ranks = len(values) + 1 - np.searchsorted(np.sort(values), values[rows], side="right")
    reciprocal = 1 / (RRF_CONSTANT + ranks)
    return reciprocal if active is None else np.where(active[rows], reciprocal, 0)


def reserved_union(ordered, dense_expert, sparse_expert):
    half = POOL // 2
    dense_top = np.argsort(-dense_expert, kind="stable")[:half]
    sparse_top = np.argsort(-sparse_expert, kind="stable")[:half]
    included = set(list(dict.fromkeys([*dense_top, *sparse_top, *ordered]))[:POOL])
    return [expert for expert in ordered if expert in included]


class HybridIndex(CandidateIndex):
    def __init__(self, reference, lexical):
        super().__init__(reference)
        if set(lexical.ids) != set(reference.source_ids):
            raise ValueError("Lexical and dense source inventories disagree")
        self.lexical = lexical
        self.row_for_source = {doc: i for i, doc in enumerate(lexical.ids)}
        self.source_order = np.array([self.row_for_source[doc] for doc in reference.source_ids])
        self.columns = np.arange(3)
        self.positions = np.minimum(self.columns, self.counts[:, None] - 1) + reference.owner_starts[:, None]
        self.mask = self.columns < self.counts[:, None]

    def source_rows(self, source_scores):
        indices = self.reference.owner_rows
        ranked = indices[np.lexsort((indices, -source_scores[indices], self.group))]
        return ranked[self.positions]

    def sparse_scores(self, query):
        values, matches = self.lexical.document_arrays(query)
        return values[self.source_order], matches[self.source_order]

    def sparse_top_three(self, scores):
        # Zero padding and unmatched rows contribute zero to the top-three mean.
        if np.any(scores < 0):
            rows = self.source_rows(scores)
            return np.sum(np.where(self.mask, scores[rows], 0), axis=1) / 3
        owner_rows = self.reference.owner_rows
        active = scores[owner_rows] > 0
        groups, rows = self.group[active], owner_rows[active]
        order = np.lexsort((rows, -scores[rows], groups))
        groups, rows = groups[order], rows[order]
        unique, starts, counts = np.unique(groups, return_index=True, return_counts=True)
        contribution = np.zeros((len(self.counts), 3))
        if len(unique):
            positions = starts[:, None] + np.minimum(self.columns, counts[:, None] - 1)
            contribution[unique] = np.where(self.columns < counts[:, None], scores[rows[positions]], 0)
        return contribution.sum(axis=1) / 3

    def deferred_union(self, vector, sparse, matched):
        ref = self.reference
        dense = torch.mv(self.gpu.vectors, torch.as_tensor(vector, device="cuda")).cpu().numpy()
        dense_expert = np.maximum.reduceat(dense[ref.owner_rows], ref.owner_starts)
        sparse_expert = self.sparse_top_three(sparse)
        expert_rrf = score_reciprocals(dense_expert) + score_reciprocals(sparse_expert, sparse_expert > 0)
        ordered = np.argsort(-expert_rrf, kind="stable")
        experts = reserved_union(ordered, dense_expert, sparse_expert)

        # Compute exact citation ranks for the shortlisted experts' sources.
        owned = [ref.owner_rows[ref.owner_starts[e]:ref.owner_starts[e] + self.counts[e]] for e in experts]
        selected = np.unique(np.concatenate(owned))
        fused = selected_reciprocals(dense, selected) + selected_reciprocals(sparse, selected, matched)
        hits = []
        for expert, rows in zip(experts, owned, strict=True):
            scores = fused[np.searchsorted(selected, rows)]
            winning = rows[np.lexsort((rows, -scores))[:3]]
            hits.append((ref.owner_ids[expert], [ref.source_ids[row] for row in winning]))
        return hits

    def search_query(self, query, encoder, arm):
        if arm != "sparse":
            dense_vector = encoder.encode([query], query=True)[0]
        if arm == "dense":
            return self.search(dense_vector, "dense")

        sparse, matched = self.sparse_scores(query)
        if arm == "deferred_union":
            return self.deferred_union(dense_vector, sparse, matched)
        sparse_rows = self.source_rows(sparse)
        sparse_expert = np.sum(np.where(self.mask, sparse[sparse_rows], 0), axis=1) / 3
        if arm == "sparse":
            return self.hits(np.argsort(-sparse_expert, kind="stable")[:POOL], sparse_rows)

        dense = torch.mv(self.gpu.vectors, torch.as_tensor(dense_vector, device="cuda")).cpu().numpy()
        source_rrf = score_reciprocals(dense) + score_reciprocals(sparse, matched)
        fused_rows = self.source_rows(source_rrf)
        if arm == "source_fusion":
            return self.hits(np.argsort(-source_rrf[fused_rows[:, 0]], kind="stable")[:POOL], fused_rows)

        ref = self.reference
        dense_expert = np.maximum.reduceat(dense[ref.owner_rows], ref.owner_starts)
        expert_rrf = score_reciprocals(dense_expert) + score_reciprocals(sparse_expert, sparse_expert > 0)
        ordered = np.argsort(-expert_rrf, kind="stable")
        if arm == "union":
            ordered = reserved_union(ordered, dense_expert, sparse_expert)
        return self.hits(ordered[:POOL], fused_rows)

    def hits(self, experts, rows):
        ref = self.reference
        return [(ref.owner_ids[e], [ref.source_ids[j] for j in rows[e, self.mask[e]]]) for e in experts]


def request(index, encoder, verifier, texts, query, truth, arm):
    start = perf_counter()
    hits = index.search_query(query, encoder, arm)
    retrieved = perf_counter()
    pool = hits[:TOP]
    reader = WaveReader(verifier, query, texts, {doc for _, docs in pool for doc in docs}, dynamic=True, cache=True)
    outputs = resolve_shared_frontier(
        pool, reader.score, reader.recheck, reader.length, support=2, top=TOP, width=WIDTH,
        band=GUARD_MARGIN, cost_aware=False,
    )["selected"]
    finished = perf_counter()
    positive = {expert for expert, val in truth.items() if val > 0}
    candidates = {expert for expert, _ in hits}
    known = {expert for expert, _ in pool}
    return dict(
        candidates=positive & candidates,
        positives=len(positive),
        p5=len(positive & known) / TOP,
        judged=len(known & truth.keys()) / TOP,
        cited_p5=sum(expert in positive and bool(docs) for expert, docs in outputs) / TOP,
        citation_coverage=sum(bool(docs) for _, docs in outputs) / TOP,
        calls=reader.score_calls + reader.recheck_calls,
        ranked_sources=hits,
        returned_citations=outputs,
        retrieval_ms=1000 * (retrieved - start),
        complete_ms=1000 * (finished - start),
    )


def summary(rows, arms=ARMS):
    out = {}
    for arm in arms:
        values = [r[arm] for r in rows]
        out[arm] = dict(
            recall10=sum(len(r["candidates"]) for r in values) / sum(r["positives"] for r in values),
            oracle_p5=mean(min(TOP, len(r["candidates"])) / TOP for r in values),
            hit10=mean(bool(r["candidates"]) for r in values),
            added=sum(len(r[arm]["candidates"] - r["dense"]["candidates"]) for r in rows),
            dropped=sum(len(r["dense"]["candidates"] - r[arm]["candidates"]) for r in rows),
            p5_delta_vs_dense=paired_interval([r[arm]["p5"] - r["dense"]["p5"] for r in rows]),
            **{key: mean(r[key] for r in values) for key in ("p5", "judged", "cited_p5", "citation_coverage", "calls")},
            **{key: np.quantile([r[key] for r in values], [0.5, 0.95]).tolist()
               for key in ("retrieval_ms", "complete_ms")},
        )
    if "deferred_union" in arms:
        out["deferred_union"]["ranked_source_agreement_vs_union"] = sum(
            row["deferred_union"]["ranked_sources"] == row["union"]["ranked_sources"] for row in rows)
        out["deferred_union"]["citation_agreement_vs_union"] = sum(
            row["deferred_union"]["returned_citations"] == row["union"]["returned_citations"] for row in rows)
    return out


def run(args):
    if args.output.exists() or not args.lexical.is_file() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a lexical index, a new output path and Slurm CUDA")
    provenance = verify_checkpoint(args.embedding)
    torch.backends.cuda.matmul.allow_tf32 = False
    root, papers, _, reference, ids, _, labels = read_data("KDD", provenance)
    texts = {pid: (paper.get("title", "") + "\n" + paper.get("abstract", "")).strip() for pid, paper in papers.items()}
    start = perf_counter()
    index = HybridIndex(reference, ArrayBM25(args.lexical))
    load_seconds = perf_counter() - start
    print(f"Loaded both indexes for {len(reference.source_ids)} sources in {load_seconds:.1f}s", flush=True)
    encoder = QwenTorchEmbedding(args.embedding, batch_size=1)
    encoder.model.config.use_cache = False
    verifier = Reranker(args.reranker)
    chosen = panel(ids, texts)
    sample = next(qid for qid in ids if qid not in chosen)
    arms = DEFERRED_ARMS if args.deferred_union else ARMS
    for arm in arms:
        request(index, encoder, verifier, texts, texts[sample], labels[sample], arm)
    rows = []
    for i, qid in enumerate(chosen):
        offset = i % len(arms)
        rows.append({arm: request(index, encoder, verifier, texts, texts[qid], labels[qid], arm)
                     for arm in arms[offset:] + arms[:offset]})
        if (i + 1) % 30 == 0:
            print(f"Completed {i + 1} queries", flush=True)
    write_json(args.output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(), queries=len(rows),
        experts=len(reference.owners), sources=len(reference.source_ids), index_load_seconds=load_seconds,
        cohort_sha256=sha256("\n".join(chosen).encode()).hexdigest(),
        script_sha256=digest(Path(__file__)), lexical_index_sha256=digest(args.lexical),
        vector_protocol_sha256=digest(root / "qwen-embedding-torch-v1/protocol.json"),
        hard_label_sha256=digest(root / "KDD_queries_test_hard.json"),
        embedding_revision=provenance["revision"], reranker_revision=verifier.provenance["revision"],
        arms=summary(rows, arms),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("embedding", "reranker", "lexical", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--deferred-union", action="store_true")
    run(parser.parse_args())
