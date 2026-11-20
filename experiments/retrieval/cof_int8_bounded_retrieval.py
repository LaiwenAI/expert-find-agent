"""Eliminate exact source checks with conservative int8 similarity bounds."""

import argparse
from hashlib import sha256
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.gpu_owned_vector_index import GpuOwnedVectorIndex
from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.followir.core17_reranking import digest, write_json
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.retrieval.cof_compact_profile_shortlist import read_data
from experiments.retrieval.cof_global_profile_pooling import select
from experiments.retrieval.cof_live_certified_batching import POOL

LEVELS = 127
CITATIONS = 3


def quantize(vectors):
    values = np.asarray(vectors, dtype=np.float64)
    peak = np.max(np.abs(values), axis=1)
    scale = np.where(peak == 0, 1., peak / LEVELS)
    quantized = np.rint(values / scale[:, None]).astype(np.int8)
    reconstructed = quantized * scale[:, None]
    return (quantized, scale, np.linalg.norm(values - reconstructed, axis=1),
            np.linalg.norm(reconstructed, axis=1), np.linalg.norm(values, axis=1))


def bounds(products, scales, residuals, norms, query_scale, query_error, query_quant_norm, query_norm, dimensions):
    approximation = products.astype(np.float64) * scales * query_scale
    # The first two terms cover quantizing both operands. The last term
    # conservatively admits the FP32 exact-route dot-product roundoff.
    rounding = (2 * dimensions + 1) * np.finfo(np.float32).eps / 2
    allowance = norms * (query_error + rounding / (1 - rounding) * query_norm) + residuals * query_quant_norm
    lower = np.nextafter(approximation - allowance, -np.inf)
    upper = np.nextafter(approximation + allowance, np.inf)
    return lower, upper


def candidates(reference, lower, upper, k):
    rows, starts = reference.owner_rows, reference.owner_starts
    owner_lower = np.maximum.reduceat(lower[rows], starts)
    owner_upper = np.maximum.reduceat(upper[rows], starts)
    threshold = np.partition(owner_lower, -k)[-k]
    owners = np.flatnonzero(owner_upper >= threshold)
    retained = {}
    for i in owners:
        owned = np.asarray(reference.owners[reference.owner_ids[i]])
        count = min(len(owned), CITATIONS)
        cutoff = np.partition(lower[owned], -count)[-count]
        retained[i] = owned[upper[owned] >= cutoff]
    return retained


class BoundedIndex:
    def __init__(self, reference):
        self.exact = GpuOwnedVectorIndex(reference)
        self.reference = reference
        n, d = reference.vectors.shape
        if d % 32 or d * LEVELS**2 > np.iinfo(np.int32).max or n < POOL:
            raise ValueError("Unsupported integer matrix geometry")
        columns = np.empty(((n + 31) // 32 * 32, d), dtype=np.int8)
        columns[n:] = 0
        self.scales, self.residuals, self.norms = (np.empty(n, dtype=np.float64) for _ in range(3))
        for start in range(0, n, 4096):
            stop = min(start + 4096, n)
            signed, scale, error, _, norm = quantize(reference.vectors[start:stop])
            columns[start:stop] = signed
            self.scales[start:stop] = scale
            self.residuals[start:stop] = error
            self.norms[start:stop] = norm
        self.codes = torch.as_tensor(columns, device="cuda")
        self.dimension = d
        self.rows = n

    @torch.inference_mode()
    def bounded(self, query):
        first = perf_counter()
        signed, scale, error, q_quant_norm, q_norm = quantize(query[None, :])
        query_device = torch.as_tensor(signed, device="cuda")
        # Repeat a column to meet the CUDA integer GEMM shape constraint.
        query_matrix = query_device.T.expand(-1, 8).contiguous()
        products = torch._int_mm(self.codes, query_matrix)[:self.rows, 0].contiguous().cpu().numpy()
        lower, upper = bounds(products, self.scales, self.residuals, self.norms,
                              scale[0], error[0], q_quant_norm[0], q_norm[0], self.dimension)
        coarse_ms = (perf_counter() - first) * 1000
        shortlist = candidates(self.reference, lower, upper, POOL)
        candidate_ms = (perf_counter() - first) * 1000 - coarse_ms

        selected = np.unique(np.concatenate(list(shortlist.values())))
        dense_query = torch.as_tensor(query, device="cuda")
        storage = self.exact.vectors.index_select(0, torch.as_tensor(selected, device="cuda"))
        scores = torch.mv(storage, dense_query).cpu().numpy()
        score_by_row = dict(zip(selected, scores, strict=True))
        ranked = []
        for i, owned in shortlist.items():
            ordered = sorted(owned, key=lambda row: (-score_by_row[row], self.reference.source_ids[row]))[:CITATIONS]
            owner = self.reference.owner_ids[i]
            ranked.append((owner, [self.reference.source_ids[row] for row in ordered], float(score_by_row[ordered[0]])))
        ranked.sort(key=lambda item: (-item[2], item[0]))
        return [(person, sources) for person, sources, _ in ranked[:POOL]], len(shortlist), len(selected), coarse_ms, candidate_ms

    def full(self, query):
        hits = self.exact.search_vector(query, k=POOL)["hits"]
        return [(item["id"], [doc["id"] for doc in item["evidence"]]) for item in hits]


def run(args):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Use a fresh output on Slurm CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    provenance = verify_checkpoint(args.embedding)
    root, papers, _, reference, ids, _, labels = read_data(args.dataset, provenance)
    texts = {pid: (p.get("title", "") + "\n" + p.get("abstract", "")).strip() for pid, p in papers.items()}
    chosen = select(args.dataset, ids, texts)
    begun = perf_counter()
    index = BoundedIndex(reference)
    build_seconds = perf_counter() - begun
    embedding = QwenTorchEmbedding(args.embedding, batch_size=1)

    def score(text, method):
        started = perf_counter()
        vector = embedding.encode([text], query=True)[0]
        ready = perf_counter()
        if method == "full":
            result, owners, sources, coarse, nomination = index.full(vector), len(reference.owner_ids), index.rows, 0., 0.
        else:
            result, owners, sources, coarse, nomination = index.bounded(vector)
        done = perf_counter()
        return dict(ranked=result, owners=owners, exact_sources=sources, coarse_ms=coarse, nomination_ms=nomination,
                    retrieval_ms=(done - ready)*1000, full_ms=(done - started)*1000)

    warm = next(q for q in ids if q not in chosen)
    score(texts[warm], "full")
    score(texts[warm], "bounded")
    rows = []
    for i, qid in enumerate(chosen):
        order = ("full", "bounded") if i % 2 == 0 else ("bounded", "full")
        data = {mode: score(texts[qid], mode) for mode in order}
        rows.append(data)
        if (i + 1) % 30 == 0:
            print(f"Ran paired search on {i+1} requests.", flush=True)
    found = len(rows)
    quality = {}
    truth = [labels[q] for q in chosen]
    for mode in ("full", "bounded"):
        quality[mode] = {
            "known_positive_top10": sum(sum(ground.get(owner, 0) > 0
                                             for owner, _ in row[mode]["ranked"])
                                        for row, ground in zip(rows, truth, strict=True)),
            "mean_owners_exact": float(np.mean([row[mode]["owners"] for row in rows])),
            "mean_sources_exact": float(np.mean([row[mode]["exact_sources"] for row in rows])),
            **{key + "_p95": float(np.quantile([row[mode][key] for row in rows], .95))
               for key in ("full_ms", "retrieval_ms", "coarse_ms", "nomination_ms")},
        }
    write_json(args.output, dict(
        dataset=args.dataset, queries=found, source_rows=index.rows, reviewers=len(reference.owner_ids),
        dense_exact_outputs=sum(row["full"]["ranked"] == row["bounded"]["ranked"] for row in rows),
        expert_order_agreement=sum([e for e, _ in row["full"]["ranked"]] ==
                                  [e for e, _ in row["bounded"]["ranked"]] for row in rows),
        cohort_sha256=sha256("\n".join(chosen).encode()).hexdigest(),
        labels_sha256=digest(root / f"{args.dataset}_queries_test_hard.json"),
        index=dict(build_seconds=build_seconds, fp32_bytes=index.exact.vectors.numel()*4,
                   int8_bytes=index.codes.numel(), dimensions=index.dimension),
        model_revision=provenance["revision"], gpu=torch.cuda.get_device_name(),
        arms=quality, script_sha256=digest(Path(__file__)),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("KDD", "SciRepEval"))
    parser.add_argument("embedding", type=Path)
    parser.add_argument("output", type=Path)
    run(parser.parse_args())
