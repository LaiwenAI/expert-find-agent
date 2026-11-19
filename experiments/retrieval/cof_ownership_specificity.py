"""Separate owner-aware candidate ordering from owner-aware citation sourcing."""

import argparse
from collections import Counter
from hashlib import sha256
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.statistics import paired_interval
from experiments.retrieval import cof_global_profile_pooling as baseline


ARMS = ("dense", "mean_max", "specific_rank", "specific_citations", "specific_both")
DEGREE_PENALTY = .025


class OwnershipIndex(baseline.ProfileIndex):
    def __init__(self, reference):
        super().__init__(reference)
        self.degree = np.bincount(reference.owner_rows, minlength=len(reference.source_ids))
        if np.any(self.degree < 1):
            raise ValueError("Source without a reviewer association")
        self.source_degree = dict(zip(reference.source_ids, self.degree, strict=True))
        self.penalty = (DEGREE_PENALTY * np.log2(self.degree)).astype(np.float32)

    def search_query(self, query, encoder, arm):
        if arm in ("dense", "mean_max"):
            return super().search_query(query, encoder, arm)
        vector = encoder.encode([query], query=True)[0]
        ref = self.reference
        scores = torch.mv(self.gpu.vectors, torch.as_tensor(vector, device="cuda")).cpu().numpy()
        adjusted = scores - self.penalty
        rank_sources = adjusted if arm in ("specific_rank", "specific_both") else scores
        cite_sources = adjusted if arm in ("specific_citations", "specific_both") else scores
        maxima = np.maximum.reduceat(rank_sources[ref.owner_rows], ref.owner_starts)
        combined = .5 * (self.centroids @ vector + maxima)
        hits = []
        for owner_index in np.argsort(-combined, kind="stable")[:baseline.POOL]:
            owner = ref.owner_ids[owner_index]
            docs = sorted(ref.owners[owner], key=lambda j: (-float(cite_sources[j]), ref.source_ids[j]))[:3]
            hits.append((owner, [ref.source_ids[j] for j in docs]))
        return hits


def audit(rows, truths, degree):
    controls = [r["mean_max"] for r in rows]
    report = baseline.summarize(rows, truths, ARMS)
    for arm in ARMS:
        counts = Counter()
        effect = []
        for row, truth, control in zip(rows, truths, controls, strict=True):
            result = row[arm]
            before = {owner for owner, _ in control["ranked_sources"]}
            after = {owner for owner, _ in result["ranked_sources"]}
            top = {owner for owner, _ in result["ranked_sources"][:baseline.TOP]}
            old_top = {owner for owner, _ in control["ranked_sources"][:baseline.TOP]}
            cited = {owner for owner, docs in result["returned_citations"] if docs}
            counts["new_positive_top10"] += len(result["candidates"] - control["candidates"])
            counts["new_positive_reaches_top5"] += len((result["candidates"] - control["candidates"]) & top)
            counts["new_positive_reaches_cited_top5"] += len(
                (result["candidates"] - control["candidates"]) & top & cited)
            counts["lost_positive_top10"] += len(control["candidates"] - result["candidates"])
            counts["unknown_top10_added"] += len(after - before - truth.keys())
            counts["unknown_top10_lost"] += len(before - after - truth.keys())
            counts["unknown_top5"] += len(top - truth.keys())
            counts["unknown_cited_top5"] += len(cited - truth.keys())
            counts["known_negative_cited_top5"] += sum(truth.get(owner) == 0 for owner in cited)
            proposed = Counter(doc for _, docs in result["ranked_sources"][:baseline.TOP] for doc in docs)
            returned = Counter(doc for _, docs in result["returned_citations"] for doc in docs)
            counts["proposed_source_slots"] += proposed.total()
            counts["proposed_shared_source_slots"] += sum(n for doc, n in proposed.items() if degree[doc] > 1)
            counts["cited_source_slots"] += returned.total()
            counts["cited_shared_source_slots"] += sum(n for doc, n in returned.items() if degree[doc] > 1)
            counts["duplicated_citation_slots_in_response"] += sum(n - 1 for n in returned.values())
            counts["same_top5_identities"] += int(top == old_top)
            effect.append(result["cited_p5"] - control["cited_p5"])
        report[arm]["ownership_audit"] = dict(counts)
        report[arm]["cited_positive_difference_vs_equal_blend"] = paired_interval(effect)
    return report


def run(dataset, embedding, reranker, output):
    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require a fresh CUDA Slurm run")
    torch.backends.cuda.matmul.allow_tf32 = False
    provenance = baseline.verify_checkpoint(embedding)
    root, papers, _, reference, ids, _, labels = baseline.read_data(dataset, provenance)
    texts = {pid: (p.get("title", "") + "\n" + p.get("abstract", "")).strip() for pid, p in papers.items()}
    index = OwnershipIndex(reference)
    encoder = baseline.QwenTorchEmbedding(embedding, batch_size=1)
    encoder.model.config.use_cache = False
    verifier = baseline.Reranker(reranker)
    chosen = baseline.select(dataset, ids, texts)
    warm = next(q for q in ids if q not in chosen)
    for arm in ARMS:
        baseline.request(index, encoder, verifier, texts, texts[warm], labels[warm], arm)
    rows = []
    for i, q in enumerate(chosen):
        offset = i % len(ARMS)
        rows.append({arm: baseline.request(index, encoder, verifier, texts, texts[q], labels[q], arm)
                     for arm in ARMS[offset:] + ARMS[:offset]})
        if (i + 1) % 30 == 0:
            print(f"{dataset}: completed {i+1}/{len(chosen)}", flush=True)
    report = audit(rows, [labels[q] for q in chosen], index.source_degree)
    if report["specific_citations"]["ownership_audit"]["same_top5_identities"] != len(chosen):
        raise ValueError("Citation-only intervention changed the ranker")
    baseline.write_json(output, dict(
        dataset=dataset, gpu=torch.cuda.get_device_name(), job=os.environ["SLURM_JOB_ID"],
        experts=len(reference.owner_ids), sources=len(index.degree), shared_sources=int((index.degree > 1).sum()),
        known_positives=sum(sum(v > 0 for v in labels[q].values()) for q in chosen), queries=len(chosen),
        cohort_sha256=sha256("\n".join(chosen).encode()).hexdigest(),
        vector_protocol_sha256=baseline.digest(root / "qwen-embedding-torch-v1/protocol.json"),
        hard_label_sha256=baseline.digest(root / f"{dataset}_queries_test_hard.json"),
        script_sha256=baseline.digest(Path(__file__)), embedding_revision=provenance["revision"],
        reranker_revision=verifier.provenance["revision"], degree_penalty=DEGREE_PENALTY, arms=report,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("KDD", "SciRepEval"))
    for name in ("embedding", "reranker", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
