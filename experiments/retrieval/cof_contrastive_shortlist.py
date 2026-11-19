"""Compare corpus-centered expert retrieval with a query-adaptive shortlist."""

import argparse
from hashlib import sha256
import os
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import torch
from torch.nn.functional import normalize

from expert_search.gpu_owned_vector_index import GpuOwnedVectorIndex
from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.retrieval import cof_global_profile_pooling as profiles
from experiments.retrieval.cof_compact_profile_shortlist import read_data
from experiments.retrieval.cof_global_hybrid import request, score_reciprocals
from experiments.retrieval.cof_live_certified_batching import POOL

ARMS = ("dense", "mean_max", "centered", "fixed_fusion", "selective_fusion")


def best(scores, count):
    return np.argsort(-scores, kind="stable")[:count]


class ContrastiveIndex:
    def __init__(self, reference):
        self.reference = reference
        self.gpu = GpuOwnedVectorIndex(reference)
        self.background = self.gpu.vectors.mean(dim=0)
        self.centered = normalize(self.gpu.vectors - self.background, dim=1)
        self.raw_profiles = self.profile_centers(self.gpu.vectors)
        self.centered_profiles = self.profile_centers(self.centered)
        self.expanded = False

    def profile_centers(self, vectors):
        return torch.stack([
            normalize(vectors[self.reference.owners[owner]].mean(dim=0), dim=0)
            for owner in self.reference.owner_ids
        ])

    def scores(self, vector, vectors, centers):
        ref = self.reference
        sources = torch.mv(vectors, vector).cpu().numpy()
        source_max = np.maximum.reduceat(sources[ref.owner_rows], ref.owner_starts)
        context = torch.mv(centers, vector).cpu().numpy()
        return sources, source_max, .5 * (context + source_max)

    def search_query(self, query, encoder, arm):
        self.expanded = False
        vector = encoder.encode([query], query=True)[0]
        if arm == "dense":
            hits = self.gpu.search_vector(vector, k=POOL)["hits"]
            return [(row["id"], [e["id"] for e in row["evidence"]]) for row in hits]
        vector = torch.as_tensor(vector, device=self.gpu.vectors.device)
        raw, source_max, blend = self.scores(vector, self.gpu.vectors, self.raw_profiles)
        expert_scores, citations = blend, raw
        agrees = len(set(best(source_max, 3)) & set(best(blend, 3))) >= 2
        if arm in {"centered", "fixed_fusion"} or (arm == "selective_fusion" and not agrees):
            self.expanded = True
            residual = normalize(vector - self.background, dim=0)
            centered, _, centered_blend = self.scores(residual, self.centered, self.centered_profiles)
            if arm == "centered":
                expert_scores, citations = centered_blend, centered
            else:
                expert_scores = score_reciprocals(blend) + score_reciprocals(centered_blend)
                citations = raw + centered
        ref = self.reference
        result = []
        for i in best(expert_scores, POOL):
            owner = ref.owner_ids[i]
            docs = sorted(ref.owners[owner], key=lambda j: (-float(citations[j]), ref.source_ids[j]))[:3]
            result.append((owner, [ref.source_ids[j] for j in docs]))
        return result


def run(dataset, embedding, reranker, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and a Slurm CUDA allocation")
    torch.backends.cuda.matmul.allow_tf32 = False
    provenance = verify_checkpoint(embedding)
    root, papers, _, reference, ids, _, labels = read_data(dataset, provenance)
    texts = {pid: (paper.get("title", "") + "\n" + paper.get("abstract", "")).strip()
             for pid, paper in papers.items()}
    start = perf_counter()
    index = ContrastiveIndex(reference)
    torch.cuda.synchronize()
    index_seconds = perf_counter() - start
    encoder = QwenTorchEmbedding(embedding, batch_size=1)
    encoder.model.config.use_cache = False
    verifier = Reranker(reranker)
    chosen = profiles.select(dataset, ids, texts)

    def evaluate(qid, arm):
        item = request(index, encoder, verifier, texts, texts[qid], labels[qid], arm)
        item["expanded"] = index.expanded
        return item

    warmup = next(qid for qid in ids if qid not in chosen)
    for arm in ARMS:
        evaluate(warmup, arm)
    rows = []
    for i, qid in enumerate(chosen):
        offset = i % len(ARMS)
        rows.append({arm: evaluate(qid, arm) for arm in ARMS[offset:] + ARMS[:offset]})
        if (i + 1) % 30 == 0:
            print(f"{dataset} completed {i + 1} of {len(chosen)} queries", flush=True)

    report = profiles.summarize(rows, [labels[qid] for qid in chosen], arms=ARMS)
    for arm in ARMS:
        report[arm]["expanded_fraction"] = mean(row[arm]["expanded"] for row in rows)
    write_json(output, dict(
        dataset=dataset, job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        experts=len(reference.owner_ids), sources=len(reference.source_ids), queries=len(chosen),
        cohort_sha256=sha256("\n".join(chosen).encode()).hexdigest(), index_seconds=index_seconds,
        known_positives=sum(sum(value > 0 for value in labels[qid].values()) for qid in chosen),
        script_sha256=digest(Path(__file__)), reference_request_sha256=digest(Path(request.__code__.co_filename)),
        profile_runner_sha256=digest(Path(profiles.__file__)),
        hard_label_sha256=digest(root / f"{dataset}_queries_test_hard.json"),
        vector_protocol_sha256=digest(root / "qwen-embedding-torch-v1/protocol.json"),
        embedding_revision=provenance["revision"], reranker_revision=verifier.provenance["revision"],
        arms=report,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("KDD", "SciRepEval"))
    for key in ("embedding", "reranker", "output"):
        parser.add_argument(key, type=Path)
    run(**vars(parser.parse_args()))
