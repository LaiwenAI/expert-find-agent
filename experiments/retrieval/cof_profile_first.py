"""Compare profile-first and full-source search with a shared final scoring rule."""

import argparse
from hashlib import sha256
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.retrieval.cof_global_profile_pooling import (
    POOL, TOP, ProfileIndex, read_data, request, select, summarize,
)


ARMS = ("full_blend", "centroid32", "two_centers32", "union16", "union32", "union64")
BUDGETS = {"centroid32": 32, "two_centers32": 32, "union16": 16, "union32": 32, "union64": 64}


def shortlist(centroid, modes, arm):
    count = BUDGETS[arm]
    standard = np.argsort(-centroid, kind="stable")[:count]
    if arm.startswith("centroid"):
        return standard
    split = np.argsort(-modes, kind="stable")[:count]
    if arm.startswith("two_centers"):
        return split
    half = count // 2
    picks = list(dict.fromkeys([*standard[:half], *split[:half], *standard, *split]))[:count]
    return np.asarray(picks, dtype=np.int64)


class ProfileFirstIndex(ProfileIndex):
    def search_query(self, query, encoder, arm):
        vector = encoder.encode([query], query=True)[0]
        ref = self.reference
        device_query = torch.as_tensor(vector, device="cuda")
        centroid = self.centroids @ vector
        if arm == "full_blend":
            positions = np.arange(len(ref.owner_ids))
            raw = torch.mv(self.gpu.vectors, device_query).cpu().numpy()
            self.last_rows = len(raw)
        else:
            modes = None if arm == "centroid32" else (self.mode_centers @ vector).reshape(-1, 2).max(axis=1)
            positions = shortlist(centroid, modes, arm)
            owners = [ref.owners[ref.owner_ids[i]] for i in positions]
            rows = np.unique(np.concatenate(owners))
            device_rows = torch.as_tensor(rows, device="cuda")
            exact = torch.mv(self.gpu.vectors.index_select(0, device_rows), device_query).cpu().numpy()
            raw = np.empty(len(ref.source_ids), dtype=np.float32)
            raw[rows] = exact
            self.last_rows = len(rows)
        owners = [ref.owners[ref.owner_ids[i]] for i in positions]
        maxima = (np.maximum.reduceat(raw[ref.owner_rows], ref.owner_starts) if arm == "full_blend" else
                  np.asarray([raw[docs].max() for docs in owners]))
        values = .5 * (centroid[positions] + maxima)
        order = np.lexsort((positions, -values))[:POOL]
        hits = []
        for i in order:
            owner = ref.owner_ids[positions[i]]
            docs = sorted(owners[i], key=lambda p: (-float(raw[p]), ref.source_ids[p]))[:3]
            hits.append((owner, [ref.source_ids[doc] for doc in docs]))
        return hits


def evaluate(index, encoder, verifier, texts, query, truth, arm):
    result = request(index, encoder, verifier, texts, query, truth, arm)
    result["scored_sources"] = index.last_rows
    return result


def aggregate(rows, labels):
    # Existing aggregate computes paired and missing-label contrasts against its primary arm.
    normalized = [dict(row, dense=row["full_blend"]) for row in rows]
    results = summarize(normalized, labels, arms=ARMS)
    for name, outcome in results.items():
        arm = [row[name] for row in rows]
        reference = [row["full_blend"] for row in rows]
        outcome["exact_expert_and_source_matches"] = sum(
            a["ranked_sources"] == b["ranked_sources"] for a, b in zip(arm, reference, strict=True))
        outcome["exact_citation_matches"] = sum(
            a["returned_citations"] == b["returned_citations"] for a, b in zip(arm, reference, strict=True))
        outcome["mean_scored_sources"] = float(np.mean([row["scored_sources"] for row in arm]))
    return results


def run(dataset, embedding, reranker, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    provenance = verify_checkpoint(embedding)
    root, papers, _, reference, ids, _, labels = read_data(dataset, provenance)
    texts = {pid: (paper.get("title", "") + "\n" + paper.get("abstract", "")).strip()
             for pid, paper in papers.items()}
    start = perf_counter()
    index = ProfileFirstIndex(reference)
    index_seconds = perf_counter() - start
    encoder = QwenTorchEmbedding(embedding, batch_size=1)
    encoder.model.config.use_cache = False
    verifier = Reranker(reranker)
    chosen = select(dataset, ids, texts)
    warm = next(q for q in ids if q not in chosen)
    for arm in ARMS:
        evaluate(index, encoder, verifier, texts, texts[warm], labels[warm], arm)
    rows = []
    for i, q in enumerate(chosen):
        offset = i % len(ARMS)
        rows.append({arm: evaluate(index, encoder, verifier, texts, texts[q], labels[q], arm)
                     for arm in ARMS[offset:] + ARMS[:offset]})
        if (i + 1) % 25 == 0:
            print(f"{dataset}: {i + 1}/{len(chosen)}", flush=True)
    write_json(output, dict(
        dataset=dataset, job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        experts=len(reference.owner_ids), sources=len(reference.source_ids), queries=len(chosen),
        index_seconds=index_seconds, cohort_sha256=sha256("\n".join(chosen).encode()).hexdigest(),
        known_positives=sum(sum(v > 0 for v in labels[q].values()) for q in chosen),
        code_sha256=digest(Path(__file__)),
        vector_protocol_sha256=digest(root / "qwen-embedding-torch-v1/protocol.json"),
        embedding_revision=provenance["revision"], reranker_revision=verifier.provenance["revision"],
        arms=aggregate(rows, [labels[q] for q in chosen]),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("KDD", "SciRepEval"))
    for key in ("embedding", "reranker", "output"):
        parser.add_argument(key, type=Path)
    run(**vars(parser.parse_args()))
