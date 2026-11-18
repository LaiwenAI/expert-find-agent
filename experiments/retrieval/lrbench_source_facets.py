"""Measure joint scientific evidence across owned papers on released LR-Bench pairs."""

import argparse
from collections import defaultdict
import importlib.util
from itertools import combinations
import json
import os
from pathlib import Path
import re
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import clustered_paired_interval
from experiments.followir.core17_reranking import digest, write_json
from experiments.followir.core17_verifier import Reranker
from experiments.litsearch.dense_cache import verify_checkpoint
from experiments.retrieval.lrbench_frozen_baselines import ASSET_HASHES, paired_stats, read_split


TASKS = {
    "general": "Does the candidate publication demonstrate expertise relevant to reviewing the query paper?",
    "methods": "Does the candidate publication demonstrate methods useful for reviewing the query paper?",
    "problem": "Does the candidate publication address the research problem or domain of the query paper?",
}
ARMS = ("dense_max", "dense_top3", "general_first", "general_max", "general_mean",
        "facet_single", "facet_pair")
DEPTH = 3
GPU_BATCH = 16


def normalized_title(document):
    return tuple(re.findall(r"\w+", document.partition("\n")[0].casefold()))


def scores(dense, features):
    general, methods, problem = features.T
    facets = np.column_stack((methods, problem))
    single = facets.min(axis=1)
    best_single = int(np.argmax(single))
    best_pair, pair_score = (best_single,), float(single[best_single])
    for i, j in combinations(range(len(facets)), 2):
        candidate = float(np.minimum(*np.maximum(facets[i], facets[j])))
        if candidate > pair_score:
            best_pair, pair_score = (i, j), candidate
    return {
        "dense_max": float(dense[0]), "dense_top3": float(dense.mean()),
        "general_first": float(general[0]), "general_max": float(general.max()),
        "general_mean": float(general.mean()), "facet_single": float(single[best_single]),
        "facet_pair": pair_score,
    }, best_pair


def select_documents(uses, documents, queries, doc_vectors, query_vectors):
    output = {}
    for query, profile in uses:
        if not profile or any(normalized_title(part) == normalized_title(query) for part in profile):
            raise ValueError("Empty or title-overlapping owned publication history")
        values = doc_vectors[[documents[doc] for doc in profile]] @ query_vectors[queries[query]]
        ranked = np.argsort(-values, kind="stable")[:DEPTH]
        output[query, profile] = ([documents[profile[i]] for i in ranked], values[ranked])
    return output


def collect(reader, by_query, docs):
    values = {}
    checks = defaultdict(int)
    for query, pool in by_query.items():
        unique = sorted(pool)
        for start in range(0, len(unique), GPU_BATCH):
            batch = unique[start:start + GPU_BATCH]
            features = []
            for task in TASKS.values():
                margins, truncated = reader.score(query, docs, batch, task=task)
                checks["truncated"] += truncated
                checks["document_task_pairs"] += len(batch)
                features.append(margins)
            for doc, row in zip(batch, np.asarray(features, dtype=np.float64).T, strict=True):
                values[query, doc] = row
    return values, dict(checks)


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("A Slurm CUDA allocation is required")
    for name, expected in ASSET_HASHES.items():
        if digest(args.release / name) != expected:
            raise ValueError(f"Changed input asset: {name}")
    inputs = {split: read_split(args.release, split) for split in ("pc", "rc")}
    uses = dict.fromkeys(variant for pairs, _ in inputs.values() for pair in pairs for variant in pair)
    queries = {q: i for i, q in enumerate(dict.fromkeys(q for q, _ in uses))}
    documents = {d: i for i, d in enumerate(dict.fromkeys(d for _, profile in uses for d in profile))}
    embed = verify_checkpoint(args.embedding)
    encoder = QwenTorchEmbedding(args.embedding, max_length=512, batch_size=32)
    encoder.model.config.use_cache = False
    reader = Reranker(args.reranker)
    start = perf_counter()
    doc_vectors = encoder.encode(list(documents))
    query_vectors = encoder.encode(list(queries), query=True)
    shortlist = select_documents(uses, documents, queries, doc_vectors, query_vectors)
    by_query = defaultdict(set)
    for (query, _), (chosen, _) in shortlist.items():
        by_query[query].update(chosen)
    indexed = perf_counter()
    features, checks = collect(reader, by_query, dict(enumerate(documents)))
    scored = perf_counter()
    judgments = {
        key: scores(dense, np.stack([features[key[0], pid] for pid in chosen]))
        for key, (chosen, dense) in shortlist.items()
    }
    spec = importlib.util.spec_from_file_location("rate_evaluator", args.release / "rate-evaluation_script.py")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    report, rows = {}, {}
    for split, (pairs, reference) in inputs.items():
        predictions = {"rate": reference}
        for arm in ARMS:
            predictions[arm] = [
                dict(ref, positive_score=judgments[pos][0][arm], negative_score=judgments[neg][0][arm])
                for (pos, neg), ref in zip(pairs, reference, strict=True)
            ]
        accuracy = paired_stats(evaluator, split, predictions)
        improvements = {}
        for method, base in (("general_max", "general_first"), ("facet_pair", "facet_single"),
                             ("facet_pair", "dense_top3")):
            differences = [
                int(left["positive_score"] > left["negative_score"])
                - int(right["positive_score"] > right["negative_score"])
                for left, right in zip(predictions[method], predictions[base], strict=True)
            ]
            improvements[f"{method}_minus_{base}"] = clustered_paired_interval(
                differences, [row["anchor_id"] for row in reference]
            )
        variants = {variant for pair in pairs for variant in pair}
        report[split] = dict(accuracy=accuracy, improvements=improvements,
                             unique_profiles=len(variants),
                             selected_two_sources=sum(len(judgments[key][1]) == 2 for key in variants),
                             mean_sources=mean(len(shortlist[key][0]) for key in variants))
        rows[split] = predictions
    write_json(args.output, dict(
        job_id=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        script_sha256=digest(Path(__file__)), input_sha256=ASSET_HASHES, embedding=embed,
        reranker=reader.provenance, query_profile_pairs=len(uses), checks=checks,
        embedding_and_shortlisting_seconds=indexed-start, verifier_chunked_seconds=scored-indexed,
        report=report, predictions=rows,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("release", "embedding", "reranker", "output"):
        parser.add_argument(name, type=Path)
    run(parser.parse_args())
