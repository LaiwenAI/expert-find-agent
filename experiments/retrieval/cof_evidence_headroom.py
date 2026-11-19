"""Compute exact paired headroom for choosing sources and ranking confirmed reviewers."""

import argparse
from collections import Counter
from hashlib import sha256
from itertools import combinations
import json
import os
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from experiments.retrieval import cof_followup_value as prior


REFERENCE = "9341b1f1da94d4c2681d7ed898d4507e840366f7a1592bb53f6a1eb7db39a0a7"
DATASETS = (*prior.DATASETS, "NIPS")
CHAIN = ("baseline", "fixed_ranking", "free_ranking", "full_pool", "shortlist", "all_known")


def reachable(request):
    """Map each reachable confirmed-reviewer mask to its minimum follow-up reads."""
    positives = [doc for doc in request.sources if request.margins[doc] >= 0.]
    bits = {doc: 1 << i for i, doc in enumerate(positives)}
    first = sum(bits[doc] for doc in request.initial if doc in bits)
    extra = [bit for doc, bit in bits.items() if doc not in request.initial]
    owner_masks = [sum(bits[doc] for doc in docs if doc in bits) for _, docs in request.hits]
    masks = {}
    reads = 0
    for depth in range(min(prior.FOLLOWUP, len(extra)) + 1):
        for choice in combinations(extra, depth):
            checked = first | sum(choice)
            confirmed = sum(1 << i for i, mask in enumerate(owner_masks) if (checked & mask).bit_count() >= 2)
            masks.setdefault(confirmed, depth)
            reads += 1
    full = sum(1 << i for i, mask in enumerate(owner_masks) if mask.bit_count() >= 2)
    return masks, full, reads


def fixed_ranking(mask, size):
    confirmed, unconfirmed = [], []
    for i in range(size):
        (confirmed if mask & (1 << i) else unconfirmed).append(i)
    return (confirmed + unconfirmed)[:prior.TOP]


def assess(request):
    candidate_ids = [expert for expert, _ in request.hits]
    labels = [request.labels.get(expert) for expert in candidate_ids]
    observed, _ = prior.select_documents(request, "one_hit", None)
    base = request.quality(observed)
    citations = sum(base[f"cited_{kind}"] for kind in ("positive", "negative", "unknown"))
    risk_cap = base["cited_negative"]
    masks, full, read_subsets = reachable(request)
    known_positive = sum(1 << i for i, label in enumerate(labels) if label == 1)
    known_negative = sum(1 << i for i, label in enumerate(labels) if label == 0)
    baseline_mask = sum(1 << i for i, flag in enumerate(request.ranks(observed)[1]) if flag)
    if baseline_mask not in masks:
        raise AssertionError("Baseline state is absent from reachable masks")
    cap_positive, matched_fixed, matched_free = -1, -1, -1
    minimum_at_max = prior.FOLLOWUP
    for mask, cost in masks.items():
        # Free ranking can admit any verified candidate, with unseen ownership
        # evidence still forbidden. Counts of unsupported candidates fill no cited slot.
        positive_count = (mask & known_positive).bit_count()
        cap_positive = max(cap_positive, min(prior.TOP, positive_count))
        nonnegative_count = (mask & ~known_negative).bit_count()
        if mask.bit_count() >= citations and max(0, citations - nonnegative_count) <= risk_cap:
            matched_free = max(matched_free, min(citations, positive_count))
        rank = fixed_ranking(mask, len(labels))
        cited = [i for i in rank if mask & (1 << i)]
        if len(cited) == citations and sum(labels[i] == 0 for i in cited) <= risk_cap:
            score = sum(labels[i] == 1 for i in cited)
            if score > matched_fixed:
                matched_fixed, minimum_at_max = score, cost
            elif score == matched_fixed:
                minimum_at_max = min(minimum_at_max, cost)

    full = min(prior.TOP, (full & known_positive).bit_count())
    shortlist = min(prior.TOP, known_positive.bit_count())
    all_known = min(prior.TOP, sum(label == 1 for label in request.labels.values()))
    chain = [base["cited_positive"], matched_fixed, matched_free, min(citations, full),
             min(citations, shortlist), min(citations, all_known)]
    if chain != sorted(chain) or not (cap_positive <= full <= shortlist <= all_known):
        raise AssertionError("Headroom layers are not nested")
    return dict(zip(CHAIN, chain, strict=True)), dict(
        baseline_citations=citations, baseline_cited_negative=risk_cap,
        verified_within_budget=cap_positive, verified_full_pool=full, positive_shortlist=shortlist,
        positive_all_known=all_known, explored_subsets=read_subsets, confirmed_masks=len(masks),
        min_reads_at_matched_fixed_max=minimum_at_max,
    )


def measure(data):
    records = [assess(request) for _, request in data]
    layers = {stage: sum(row[stage] for row, _ in records) for stage in CHAIN}
    gap_counts = {}
    for before, after in zip(CHAIN[:-1], CHAIN[1:], strict=True):
        deltas = [row[after] - row[before] for row, _ in records]
        gap_counts[f"{before}_to_{after}"] = dict(
            additional=sum(deltas), queries=sum(n > 0 for n in deltas),
            p5_delta=prior.paired_interval([n / prior.TOP for n in deltas]),
        )
    layer_sums = Counter()
    for _, item in records:
        layer_sums.update(item)
    return dict(matched=layers, matched_increment=gap_counts, unrestricted=dict(layer_sums),
                mean_masks=mean(row["confirmed_masks"] for _, row in records))


def run(embedding, reranker, reference, output):
    if (output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available()
            or prior.digest(reference) != REFERENCE):
        raise ValueError("Require a fresh result, pinned reference and Slurm CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    provenance = prior.verify_checkpoint(embedding)
    encoder = prior.QwenTorchEmbedding(embedding, batch_size=1)
    encoder.model.config.use_cache = False
    scorer = prior.Reranker(reranker)
    old = json.loads(reference.read_text())
    if (old["embedding_revision"] != provenance["revision"]
            or old["reranker_revision"] != scorer.provenance["revision"]):
        raise ValueError("Changed checkpoints")
    report = {}
    for dataset in DATASETS:
        kwargs = {"select_queries": lambda _dataset, ids, _texts: ids} if dataset == "NIPS" else {}
        root, data, experts, sources, truncated, _ = prior.collect(dataset, encoder, scorer, provenance, **kwargs)
        if dataset == "SIGIR":
            data = sorted(data, key=lambda row: sha256(f"cof-followup-v1:{row[0]}".encode()).digest())
            data = data[2 * len(data) // 3:]
        expected = old["report"][dataset]
        ids = sha256("\n".join(qid for qid, _ in data).encode()).hexdigest()
        label_digest = prior.digest(root / f"{dataset}_queries_test_hard.json")
        if ids != expected["qids_sha256"] or label_digest != expected["labels_sha256"]:
            raise ValueError(f"Changed panel inputs for {dataset}")
        metrics = measure(data)
        control = expected["report"]["one_hit"]
        if (metrics["matched"]["baseline"] != control["cited_positive"]
                or metrics["unrestricted"]["baseline_cited_negative"] != control["cited_negative"]
                or metrics["unrestricted"]["baseline_citations"] != sum(
                    control[f"cited_{kind}"] for kind in ("positive", "negative", "unknown"))):
            raise ValueError(f"Changed reference routing for {dataset}")
        report[dataset] = dict(metrics, queries=len(data), experts=experts, sources=sources, qids_sha256=ids,
                               labels_sha256=label_digest, truncated_pairs=truncated)
    prior.write_json(output, dict(job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
                                  script_sha256=prior.digest(Path(__file__)), reference_sha256=REFERENCE,
                                  embedding_revision=provenance["revision"],
                                  reranker_revision=scorer.provenance["revision"],
                                  followup_budget=prior.FOLLOWUP, report=report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("embedding", "reranker", "reference", "output"):
        parser.add_argument(argument, type=Path)
    run(**vars(parser.parse_args()))
