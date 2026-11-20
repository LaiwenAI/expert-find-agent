"""Compare fixed and distilled read budgets on a new community."""

import argparse
from collections import Counter, defaultdict
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import clustered_paired_interval
from experiments.community.distilled_first_query_reads import (
    EvidenceIndex, fit, paired_delta, question_text,
)
from experiments.community.evaluate_crossvalidated_witness import connected_groups
from experiments.community.first_query_owner_routing import load_panel
from experiments.community.held_window_routing import independent
from experiments.community.witness_matched_issuance import cross_site_overlap
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint


ARMS = {"dense3": 3, "dense4": 4, "dense5": 5, "dense10": 10, "adaptive": None}


def confidence(index, model, text, candidates, similarities):
    center, scale, weights = model
    rows = index.features(text, candidates, similarities)[3:5]
    return float(np.max((rows - center) / scale @ weights[1:] + weights[0]))


def calibrate(queries, index, encoder, model):
    confidence_values = []
    for query in queries:
        text = question_text(query)
        candidates, similarities = index.nominate(text, encoder)
        confidence_values.append(confidence(index, model, text, candidates, similarities))
    return float(np.median(confidence_values))


def evaluate(queries, index, encoder, reader, model, threshold, offset):
    arms = tuple(ARMS)
    records = defaultdict(list)
    complete = Counter()
    tiers = Counter()
    budget_per_query = []
    for i, query in enumerate([queries[0], *queries]):
        text = question_text(query)
        known = {index.ids[doc] for doc in query["answers"]}
        shift = (i + offset) % len(arms)
        choices = {}
        for arm in arms[shift:] + arms[:shift]:
            start = perf_counter()
            candidates, similarities = index.nominate(text, encoder)
            budget = ARMS[arm]
            if arm == "adaptive":
                budget = 5 if confidence(index, model, text, candidates, similarities) > threshold else 3
            inspected = candidates[:budget]
            scores, clipped = index.read(text, inspected, reader)
            descending = np.argsort(-np.asarray(scores), kind="stable")
            issued = [inspected[j] for j in descending if scores[j] > 0][:5]
            elapsed = (perf_counter() - start) * 1000
            if i:
                records[arm].append((issued, known, inspected, clipped, elapsed))
                choices[arm] = (budget, candidates)
        if i:
            _, candidates = choices["dense10"]
            first_hit = next((i for i, doc in enumerate(candidates, 1) if doc in known), 0)
            tier = next((label for bound, label in ((3, "top3"), (5, "rank4to5"), (10, "rank6to10"))
                         if 0 < first_hit <= bound), "absent")
            tiers[tier] += 1
            for arm in arms:
                returned, *_ = records[arm][-1]
                complete[arm] += bool(known.intersection(returned))
            budget_per_query.append(choices["adaptive"][0])

    groups = connected_groups(queries)
    # Random assignment assigns the same realized number of five-read requests
    # without consulting question features or labels. Replay observed live fixed arms.
    small = np.array([len(gold.intersection(issued)) for issued, gold, *_ in records["dense3"]])
    large = np.array([len(gold.intersection(issued)) for issued, gold, *_ in records["dense5"]])
    extra = large - small
    five_read_queries = budget_per_query.count(5)
    rng = np.random.default_rng(20261121)
    draws = [small.sum() + extra[rng.choice(len(queries), size=five_read_queries, replace=False)].sum()
             for _ in range(1000)]
    results = {}
    for arm, rows in records.items():
        results[arm] = dict(
            pairs=sum(len(inspected) for _, _, inspected, _, _ in rows),
            known_available=sum(len(gold.intersection(inspected)) for _, gold, inspected, _, _ in rows),
            known_issued=sum(len(gold.intersection(issued)) for issued, gold, *_ in rows),
            issued=sum(len(issued) for issued, *_ in rows),
            clipped=sum(row[-2] for row in rows),
            p50_p95_ms=np.quantile([row[-1] for row in rows], [.50, .95]).tolist(),
            versus_dense4=paired_delta(rows, records["dense4"], groups),
            versus_dense4_matched_issuance=paired_delta(rows, records["dense4"], groups, matched=True),
            versus_dense5=paired_delta(rows, records["dense5"], groups),
        )
    return dict(
        queries=len(queries), clusters=len(set(groups)), rank_tiers=dict(tiers),
        queries_with_known_issued=dict(complete), arms=results, adaptive_five_reads=five_read_queries,
        randomized_budget_replay=dict(known_mean=float(np.mean(draws)),
                                      empirical_95=np.quantile(draws, [.025, .975]).tolist()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("development", "later_stats", "mathoverflow", "target", "embedding", "reranker", "output"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--order-offset", type=int, default=0)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require Slurm CUDA and a fresh output path")
    dev, dev_hash = load_panel(args.development)
    later, _ = load_panel(args.later_stats)
    overflow, _ = load_panel(args.mathoverflow)
    later, _ = independent(dev, later)
    held, held_hash = load_panel(args.target)
    held, exclusions = cross_site_overlap((dev, later, overflow), held)
    provenance = verify_checkpoint(args.embedding)
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    reader = Reranker(args.reranker)
    dev_index = EvidenceIndex(dev, encoder)
    model, training = fit(dev["queries"], dev_index, encoder, reader)
    threshold = calibrate(dev["queries"], dev_index, encoder, model)
    held_index = EvidenceIndex(held, encoder)
    report = evaluate(held["queries"], held_index, encoder, reader, model, threshold, args.order_offset)
    write_json(args.output, dict(
        **report, retained_catalog_sources=len(held_index.docs), catalog_accounts=len(set(held_index.owners)),
        cross_site_exclusions=exclusions, training=training, threshold=threshold,
        dev_snapshot_sha256=dev_hash, held_snapshot_sha256=held_hash, script_sha256=digest(Path(__file__)),
        embedding_revision=provenance["revision"], reranker_revision=reader.provenance["revision"],
        gpu=torch.cuda.get_device_name(), order_offset=args.order_offset,
    ))


if __name__ == "__main__":
    main()
