"""Compare version-aware source replacement at a fixed CoF nomination and read cap."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from expert_search.gpu_owned_vector_index import GpuOwnedVectorIndex
from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import paired_interval
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.retrieval.cof_compact_profile_shortlist import read_data
from experiments.retrieval.cof_global_hybrid import request
from experiments.retrieval.cof_global_profile_pooling import select
from experiments.retrieval.cof_live_certified_batching import POOL, TOP
from experiments.retrieval.cof_profile_depth_review import fingerprints

CAP = 3
WINDOW = 12
ARMS = ("raw", "version_prefix", "version_refill", "title_prefix", "title_refill")
# OpenAlex and the authors' references link each pair of versions. These
# two families were discovered on the same panels, so replay is descriptive.
FAMILIES = (
    ("1029409534", "2018715584"),
    ("2030451946", "2418808916"),
)


def choose(ordered, identities, size=CAP):
    selected, seen = [], set()
    for doc in ordered:
        group = identities.get(doc, doc)
        if group not in seen:
            selected.append(doc)
            seen.add(group)
        if len(selected) == size:
            break
    return selected


def duplicates(documents, identities):
    return len({identities.get(doc, doc) for doc in documents}) < len(documents)


class FamilyIndex:
    def __init__(self, reference, texts):
        self.gpu = GpuOwnedVectorIndex(reference)
        self.titles = {doc: fingerprints(texts[doc], doc)[0] for doc in reference.source_ids}
        self.families = {member: group[0] for group in FAMILIES for member in group}
        for first, second in FAMILIES:
            if first in self.titles and second in self.titles and self.titles[first] != self.titles[second]:
                raise ValueError("Verified version pair no longer has its pinned common title")

    def search_query(self, query, model, arm):
        vector = model.encode([query], query=True)[0]
        depth = WINDOW if arm.endswith("_refill") else CAP
        top = self.gpu.search_vector(vector, k=POOL, evidence_limit=depth)["hits"]
        result = [(row["id"], [item["id"] for item in row["evidence"]]) for row in top]
        if arm == "raw":
            return result
        identities = self.families if arm.startswith("version") else self.titles
        return [(owner, choose(docs, identities)) if i < TOP else (owner, docs[:CAP])
                for i, (owner, docs) in enumerate(result)]


def measure(index, encoder, reranker, texts, query, labels, arm):
    reply = request(index, encoder, reranker, texts, query, labels, arm)
    counts = Counter()
    for owner, docs in reply["returned_citations"]:
        if not docs:
            continue
        status = "unjudged" if owner not in labels else "positive" if labels[owner] > 0 else "negative"
        counts["issued"] += 1
        counts[f"issued_{status}"] += 1
        for key, identities in (("confirmed", index.families), ("title", index.titles)):
            duplicate = duplicates(docs, identities)
            counts[f"duplicates_{key}"] += duplicate
            if not duplicate:
                counts[f"distinct_{key}"] += 1
                counts[f"distinct_{key}_{status}"] += 1
    reply["counts"] = counts
    return reply


def report(rows):
    totals = {}
    for arm in ARMS:
        samples = [row[arm] for row in rows]
        counts = Counter()
        changed = reduced = 0
        for base, sample in zip(rows, samples, strict=True):
            counts.update(sample["counts"])
            before, after = base["raw"]["ranked_sources"][:TOP], sample["ranked_sources"][:TOP]
            changed += sum(a[1] != b[1] for a, b in zip(before, after, strict=True))
            reduced += sum(len(a[1]) < len(b[1]) for a, b in zip(after, before, strict=True))
        totals[arm] = dict(
            **counts,
            changed_source_slots=changed, fewer_sources_than_raw_slots=reduced,
            mean_verifier_calls=mean(x["calls"] for x in samples),
            complete_p50_p95_ms=np.quantile([row["complete_ms"] for row in samples], [.5, .95]).tolist(),
            retrieval_p50_p95_ms=np.quantile([row["retrieval_ms"] for row in samples], [.5, .95]).tolist(),
            distinct_title_positive_p5_delta=paired_interval([
                (row[arm]["counts"]["distinct_title_positive"] - row["raw"]["counts"]["distinct_title_positive"]) / TOP
                for row in rows
            ]),
        )
    return totals


def run(dataset, embedding, reranker, reference, output):
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or output.exists():
        raise ValueError("Use a fresh output on a Slurm CUDA node")
    torch.backends.cuda.matmul.allow_tf32 = False
    checkpoint = verify_checkpoint(embedding)
    root, papers, _, source_index, eligible, _, labels = read_data(dataset, checkpoint)
    texts = {pid: (p.get("title", "") + "\n" + p.get("abstract", "")).strip() for pid, p in papers.items()}
    tested = select(dataset, eligible, texts)
    baseline = json.loads(reference.read_bytes())
    cohort = sha256("\n".join(tested).encode()).hexdigest()
    judgments = digest(root / f"{dataset}_queries_test_hard.json")
    if (dataset, cohort, judgments) != (baseline["dataset"], baseline["cohort_sha256"], baseline["labels_sha256"]):
        raise ValueError("Changed evaluation cohort")
    index = FamilyIndex(source_index, texts)
    encoder = QwenTorchEmbedding(embedding, batch_size=1)
    verifier = Reranker(reranker)
    if (checkpoint["revision"], verifier.provenance["revision"]) != (
            baseline["revisions"]["embedding"], baseline["revisions"]["reranker"]):
        raise ValueError("Changed model revisions")
    warm = next(q for q in eligible if q not in tested)
    for arm in ARMS:
        measure(index, encoder, verifier, texts, texts[warm], labels[warm], arm)
    rows = []
    for i, qid in enumerate(tested):
        offset = i % len(ARMS)
        arms = {arm: measure(index, encoder, verifier, texts, texts[qid], labels[qid], arm)
                for arm in ARMS[offset:] + ARMS[:offset]}
        ranks = [[owner for owner, _ in row["ranked_sources"]] for row in arms.values()]
        if any(rank != ranks[0] for rank in ranks):
            raise AssertionError("A source arm changed expert nominations")
        rows.append(arms)
        if (i + 1) % 30 == 0:
            print(f"Completed {i + 1}/{len(tested)} requests.", flush=True)
    results = report(rows)
    raw_pos = round(baseline["arms"]["dense"]["cited_p5"] * len(tested) * TOP)
    if results["raw"].get("issued_positive", 0) != raw_pos:
        raise AssertionError("Original positive citation counts did not reproduce")
    write_json(output, dict(
        dataset=dataset, queries=len(tested), cohort_sha256=cohort, labels_sha256=judgments,
        gpu=torch.cuda.get_device_name(), job=os.environ["SLURM_JOB_ID"], script_sha256=digest(Path(__file__)),
        source_index_sha256=digest(Path(GpuOwnedVectorIndex.search_vector.__code__.co_filename)),
        baseline_sha256=digest(reference), source_window=WINDOW, citation_cap=CAP,
        confirmed_family_members_in_catalog=sum(doc in index.titles for group in FAMILIES for doc in group),
        arms=results,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("KDD", "SciRepEval"))
    for name in ("embedding", "reranker", "reference", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
