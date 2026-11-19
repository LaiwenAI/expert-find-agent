"""Compare fixed profile context and owned-source evidence in full CoF retrieval."""

import argparse
from hashlib import sha256
import os
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from expert_search.cof_benchmark import load
from expert_search.gpu_owned_vector_index import GpuOwnedVectorIndex
from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import paired_interval
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.retrieval.cof_compact_profile_shortlist import read_data, title_tokens
from experiments.retrieval.cof_global_hybrid import request
from experiments.retrieval.cof_live_certified_batching import POOL, TOP
from experiments.retrieval.cof_shared_frontier import panel


ARMS = ("dense", "centroid", "top_three", "mean_max", "two_centers", "two_centers_max")


def unit(vector):
    norm = np.linalg.norm(vector)
    return vector / norm if norm else np.zeros_like(vector)


def centers(rows):
    """Build one normalized center and two spherical clusters without query access."""
    center = unit(rows.mean(axis=0))
    if len(rows) == 1:
        return center, np.stack([center, center])
    first = rows[(rows @ center).argmin()]
    second = rows[(rows @ first).argmin()]
    split = np.stack([first, second])
    for _ in range(6):
        groups = (rows @ split.T).argmax(axis=1)
        for label in range(2):
            members = rows[groups == label]
            if len(members):
                split[label] = unit(members.mean(axis=0))
    return center, split


class ProfileIndex:
    def __init__(self, reference):
        self.reference = reference
        self.gpu = GpuOwnedVectorIndex(reference)
        self.counts = np.diff(np.append(reference.owner_starts, len(reference.owner_rows)))
        self.groups = np.repeat(np.arange(len(self.counts)), self.counts)
        self.positions = reference.owner_starts[:, None] + np.minimum(
            np.arange(3), self.counts[:, None] - 1)
        self.mask = np.arange(3) < self.counts[:, None]
        profiles = [centers(reference.vectors[reference.owners[owner]]) for owner in reference.owner_ids]
        self.centroids = np.stack([row[0] for row in profiles])
        self.mode_centers = np.stack([row[1] for row in profiles]).reshape(-1, self.centroids.shape[-1])

    def search_query(self, query, encoder, arm, *, depth=3):
        if type(depth) is not int or depth < 1 or (depth != 3 and arm not in {"mean_max", "two_centers_max"}):
            raise ValueError("Custom source depth requires a blended ranker")
        vector = encoder.encode([query], query=True)[0]
        if arm == "dense":
            hits = self.gpu.search_vector(vector, k=POOL)["hits"]
            return [(row["id"], [e["id"] for e in row["evidence"]]) for row in hits]

        ref = self.reference
        if arm in {"centroid", "two_centers"}:
            rank_scores = self.profile_scores(vector, arm)
            owners = [ref.owner_ids[i] for i in np.argsort(-rank_scores, kind="stable")[:POOL]]
            hits = self.gpu.search_vector(vector, k=POOL, eligible=owners)["hits"]
            sources = {hit["id"]: [e["id"] for e in hit["evidence"]] for hit in hits}
            return [(owner, sources[owner]) for owner in owners]

        scores = torch.mv(self.gpu.vectors, torch.as_tensor(vector, device="cuda")).cpu().numpy()
        ordered_sources = None
        if arm == "top_three":
            entries = ref.owner_rows
            ordered_sources = entries[np.lexsort((entries, -scores[entries], self.groups))][self.positions]
            rank_scores = np.where(self.mask, scores[ordered_sources], 0).sum(axis=1) / self.mask.sum(axis=1)
        else:
            maxima = np.maximum.reduceat(scores[ref.owner_rows], ref.owner_starts)
            rank_scores = .5 * (self.profile_scores(vector, arm) + maxima)

        hits = []
        for i in np.argsort(-rank_scores, kind="stable")[:POOL]:
            owner = ref.owner_ids[i]
            if ordered_sources is None:
                docs = sorted(ref.owners[owner], key=lambda doc: (-float(scores[doc]), ref.source_ids[doc]))[:depth]
            else:
                docs = ordered_sources[i, self.mask[i]]
            hits.append((owner, [ref.source_ids[doc] for doc in docs]))
        return hits

    def profile_scores(self, vector, arm):
        if arm.startswith("two_centers"):
            return (self.mode_centers @ vector).reshape(-1, 2).max(axis=1)
        return self.centroids @ vector


def summarize(rows, truths):
    report = {}
    reference = [row["dense"] for row in rows]
    for arm in ARMS:
        result = [row[arm] for row in rows]
        gains, losses, lower, upper = [], [], [], []
        for current, base, truth in zip(result, reference, truths, strict=True):
            selected = {e for e, _ in current["ranked_sources"][:TOP]}
            original = {e for e, _ in base["ranked_sources"][:TOP]}
            gained, dropped = selected - original, original - selected
            positive_delta = sum(truth.get(e, 0) > 0 for e in gained) - sum(truth.get(e, 0) > 0 for e in dropped)
            lower.append((positive_delta - len(dropped - truth.keys())) / TOP)
            upper.append((positive_delta + len(gained - truth.keys())) / TOP)
            gains.append(len(current["candidates"] - base["candidates"]))
            losses.append(len(base["candidates"] - current["candidates"]))
        report[arm] = dict(
            known_positive_recall_at_10=sum(len(row["candidates"]) for row in result) / sum(row["positives"] for row in result),
            known_positive_p5_delta=paired_interval(
                [row["p5"] - base["p5"] for row, base in zip(result, reference, strict=True)]),
            partial_label_p5_delta_bounds=[mean(lower), mean(upper)],
            known_positives_added_at_10=sum(gains), known_positives_lost_at_10=sum(losses),
            **{name: mean(row[name] for row in result)
               for name in ("p5", "judged", "cited_p5", "citation_coverage", "calls")},
            **{name: np.quantile([row[name] for row in result], [.5, .95]).tolist()
               for name in ("retrieval_ms", "complete_ms")},
        )
    return report


def select(dataset, ids, texts):
    if dataset == "KDD":
        return panel(ids, texts)
    _, nips, _, labels = load("NIPS")
    nips_titles = {tuple(title_tokens(nips[q]["title"])) for q in labels["hard"]}
    return [q for q in ids if tuple(title_tokens(texts[q].split("\n", 1)[0])) not in nips_titles]


def run(dataset, embedding, reranker, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require new output and Slurm CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    provenance = verify_checkpoint(embedding)
    root, papers, _, reference, ids, _, labels = read_data(dataset, provenance)
    texts = {pid: (paper.get("title", "") + "\n" + paper.get("abstract", "")).strip()
             for pid, paper in papers.items()}
    start = perf_counter()
    index = ProfileIndex(reference)
    index_seconds = perf_counter() - start
    encoder = QwenTorchEmbedding(embedding, batch_size=1)
    encoder.model.config.use_cache = False
    verifier = Reranker(reranker)
    chosen = select(dataset, ids, texts)
    warm = next(q for q in ids if q not in chosen)
    for arm in ARMS:
        request(index, encoder, verifier, texts, texts[warm], labels[warm], arm)
    rows = []
    for i, q in enumerate(chosen):
        offset = i % len(ARMS)
        rows.append({arm: request(index, encoder, verifier, texts, texts[q], labels[q], arm)
                     for arm in ARMS[offset:] + ARMS[:offset]})
        if (i + 1) % 30 == 0:
            print("Completed requests:", i + 1, flush=True)
    write_json(output, dict(
        dataset=dataset, job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        experts=len(reference.owner_ids), sources=len(reference.source_ids), queries=len(chosen),
        index_seconds=index_seconds, known_positives=sum(sum(v > 0 for v in labels[q].values()) for q in chosen),
        cohort_sha256=sha256("\n".join(chosen).encode()).hexdigest(), hard_label_sha256=digest(root / f"{dataset}_queries_test_hard.json"),
        script_sha256=digest(Path(__file__)), reference_request_sha256=digest(Path(request.__code__.co_filename)),
        vector_protocol_sha256=digest(root / "qwen-embedding-torch-v1/protocol.json"),
        embedding_revision=provenance["revision"], reranker_revision=verifier.provenance["revision"],
        arms=summarize(rows, [labels[q] for q in chosen]),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("KDD", "SciRepEval"))
    for name in ("embedding", "reranker", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
