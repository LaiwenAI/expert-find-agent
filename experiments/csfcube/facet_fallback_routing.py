"""Estimate when query-vector evidence needs cross-encoder verification."""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import clustered_paired_interval
from experiments.csfcube.frozen_confirmation import load
from experiments.csfcube.live_facet_gate import FacetIndex, act, assignments, collect, returned
from experiments.csfcube.paired_facet_confirmation import key
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint


ARMS = ("dense4", "facet4", "cosine", "hash", "margin", "disagreement", "learned")
ROUTES = ("hash", "margin", "disagreement", "learned")
TARGET_FALLBACK = .30


def hashed(question):
    value = key(question["query_id"], question["facet"], "facet-fallback-v1")
    return int.from_bytes(value[:8], "big") / 2**64


def cosine_choice(cosine, policy):
    return returned(cosine > policy["cosine"])


def signals(cosine, policy):
    order = np.argsort(-cosine, kind="stable")[:4]
    picks = cosine_choice(cosine, policy)
    closest = float(np.abs(cosine[:4] - policy["cosine"]).min())
    disagreement = 4 - int(np.sum(order < 4))
    return np.array([len(picks), float(np.max(cosine) - policy["cosine"]), closest,
                     float(np.std(cosine)), disagreement])


def utility(grades):
    # Reward complete request-level strong retrieval and charge issuing
    # a definitely irrelevant source. Keep weak relevance separate.
    return float(any(g >= 2 for g in grades)) - .5 * float(0 in grades)


def fit_router(features, outcomes):
    center = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 1e-8)
    z = (features - center) / scale
    weight = np.linalg.solve(z.T @ z + 5 * np.eye(z.shape[1]), z.T @ outcomes)
    return center, scale, weight


def priorities(features, jitter, model):
    center, scale, weights = model
    return {
        "hash": jitter,
        "margin": -features[2] + 1e-8 * jitter,
        "disagreement": features[4] + 1e-8 * jitter,
        "learned": float(((features - center) / scale) @ weights) + 1e-8 * jitter,
    }


def train(questions, samples, folds, policies):
    routes = {}
    for fold in sorted(policies):
        policy = policies[fold]["reuse"]
        embargo = {doc for q, sample, f in zip(questions, samples, folds, strict=True) if f == fold
                   for doc in (q["query_id"], *sample["docs"])}
        kept = [(q, sample) for q, sample, f in zip(questions, samples, folds, strict=True)
                if f != fold and embargo.isdisjoint([q["query_id"], *sample["docs"]])]
        if len(kept) < 10:
            raise ValueError("Not enough nonoverlapping requests to fit a router")
        features, outcomes, randomness = [], [], []
        for question, sample in kept:
            similarity, values, grades = (sample[name] for name in ("reuse", "values", "grades"))
            checked, accepted = act("facet4", similarity, policy)
            accepted[checked] = values[checked] > policy["reader"]
            base = grades[cosine_choice(similarity, policy)]
            fallback = grades[returned(accepted)]
            features.append(signals(similarity, policy))
            outcomes.append(utility(fallback) - utility(base))
            randomness.append(hashed(question))
        model = fit_router(np.stack(features), np.asarray(outcomes))
        fitted = [priorities(x, jitter, model) for x, jitter in zip(features, randomness, strict=True)]
        cutoffs = {name: float(np.quantile([row[name] for row in fitted], 1 - TARGET_FALLBACK))
                   for name in ROUTES}
        routes[fold] = dict(model=model, cutoffs=cutoffs, train=len(kept),
                            favorable=sum(value > 0 for value in outcomes),
                            unfavorable=sum(value < 0 for value in outcomes))
    return routes


def evaluate(questions, samples, index, encoder, reader, offset):
    folds, policies = assignments(questions, samples)
    routers = train(questions, samples, folds, policies)
    stored = defaultdict(list)
    for i in range(-1, len(questions)):
        j = max(0, i)
        question, sample, fold = questions[j], samples[j], folds[j]
        policy = policies[fold]["reuse"]
        model = routers[fold]
        shift = (j + offset) % len(ARMS)
        for arm in ARMS[shift:] + ARMS[:shift]:
            start = perf_counter()
            focal, docs, similarity = index.nominate(question, encoder, "none" if arm == "dense4" else "reuse")
            cosine = similarity.get("reuse")
            fallback = arm in ("dense4", "facet4")
            if arm in ROUTES:
                scores = priorities(signals(cosine, policy), hashed(question), model["model"])
                fallback = scores[arm] > model["cutoffs"][arm]
            if fallback:
                checked, mask = act("dense4" if arm == "dense4" else "facet4", cosine, policy)
                values, clipped = index.check(reader, focal, [docs[k] for k in checked], question["facet"])
                mask[checked] = values > policy["reader"]
                picks = returned(mask)
            else:
                checked, clipped = [], 0
                picks = cosine_choice(cosine, policy)
            json.dumps({"sources": [docs[k] for k in picks]})
            ms = (perf_counter() - start) * 1000
            if docs != sample["docs"]:
                raise ValueError("An arm changed the fixed shortlist")
            if i >= 0:
                stored[arm].append(dict(grades=sample["grades"][picks].tolist(), calls=int(fallback),
                                        checked=len(checked), clipped=clipped, ms=ms))
    return summarize(questions, folds, stored, routers)


def flags(records):
    return np.array([any(g >= 2 for g in row["grades"]) for row in records], dtype=np.int64)


def summarize(questions, folds, stored, routers):
    groups = [q["query_id"] for q in questions]
    dense, cosine, verify = (flags(stored[arm]) for arm in ("dense4", "cosine", "facet4"))
    reports = {}
    for arm, records in stored.items():
        strong = flags(records)
        grade = Counter(g for row in records for g in row["grades"])
        item = dict(strong_queries=int(strong.sum()), issued_by_grade=[grade[g] for g in range(4)],
                    coverage=sum(bool(row["grades"]) for row in records),
                    fallback_queries=sum(row["calls"] for row in records), checked=sum(row["checked"] for row in records),
                    clipped=sum(row["clipped"] for row in records),
                    warm_p50_p95_ms=np.quantile([row["ms"] for row in records], [.50, .95]).tolist(),
                    delta_dense4=clustered_paired_interval(strong - dense, groups),
                    delta_cosine=clustered_paired_interval(strong - cosine, groups))
        if arm in ROUTES:
            allocation = {f: [j for j, fold in enumerate(folds) if f == fold] for f in sorted(set(folds))}
            budgets = {f: sum(records[j]["calls"] for j in chosen) for f, chosen in allocation.items()}
            rng = np.random.default_rng(20261121)
            marginal = verify - cosine
            hits = [cosine.sum() + sum(marginal[rng.choice(pool, size=budgets[f], replace=False)].sum()
                                      for f, pool in allocation.items()) for _ in range(2000)]
            item["same_budget_random"] = dict(mean=float(np.mean(hits)),
                                              central_95=np.quantile(hits, [.025, .975]).tolist(),
                                              per_fold_budgets=budgets)
        reports[arm] = item
    return dict(arms=reports, fit={
        k: {**{name: item for name, item in params.items() if name != "model"},
            "model": [arr.tolist() for arr in params["model"]]} for k, params in routers.items()
    }, queries=len(questions), query_groups=len(set(groups)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "provenance", "embedding", "reranker", "baseline", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--order-offset", type=int, default=0)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or args.output.exists():
        raise ValueError("Require Slurm CUDA and a fresh output path")
    _, questions, sources = load(args.data)
    expected = json.loads(args.provenance.read_bytes())
    embedding = verify_checkpoint(args.embedding)
    reader = Reranker(args.reranker)
    if (sources != expected["source_hashes"] or embedding["revision"] != expected["embedding"]["revision"]
        or reader.provenance["revision"] != expected["reranker"]["revision"]):
        raise ValueError("CSFCube inputs changed")
    encoder = QwenTorchEmbedding(args.embedding)
    encoder.model.config.use_cache = False
    index = FacetIndex(args.data, questions, encoder)
    samples = collect(questions, index, encoder, reader)
    result = evaluate(questions, samples, index, encoder, reader, args.order_offset)
    prior = json.loads(args.baseline.read_bytes())["runs"][0]["arms"]
    for arm, previous in (("dense4", "dense4"), ("facet4", "reuse4"), ("cosine", "reuse_cosine")):
        for measure in ("strong_queries", "issued_by_grade"):
            if result["arms"][arm][measure] != prior[previous][measure]:
                raise ValueError(f"Earlier {previous} result did not reproduce")
    write_json(args.output, dict(
        **result, order_offset=args.order_offset, source_hashes=sources,
        embedding_revision=embedding["revision"], reranker_revision=reader.provenance["revision"],
        baseline_sha256=digest(args.baseline), script_sha256=digest(Path(__file__)), gpu=torch.cuda.get_device_name(),
    ))


if __name__ == "__main__":
    main()
