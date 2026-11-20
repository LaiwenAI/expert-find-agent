"""Estimate whether routing to titles transfers across CoF collections."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np

from expert_search.cof_benchmark import load
from expert_search.cof_data import DATASETS
from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.credit import real_title_runtime as prior
from experiments.credit import topic_role_postings as provenance


CACHE_ENCODER_SHA256 = "e2b1e9dd238ba49d343b3a3a0e6f441b225062a1012219bad33c0d9e691fb9d9"


def source_text(paper):
    return (paper["title"] + "\n" + paper["abstract"]).strip()


def frozen_vectors(root, papers, profile_ids, query_ids, embedding):
    cache = root / "qwen-embedding-torch-v1"
    protocol = json.loads((cache / "protocol.json").read_bytes())
    summary = json.loads((cache / "summary.json").read_bytes())
    items = [
        dict(id=prefix + pid, text=source_text(papers[pid]), query=is_query)
        for prefix, ids, is_query in (("doc:", profile_ids, False), ("query:", query_ids, True))
        for pid in ids
    ]
    path = cache / "vectors.npy"
    digest = provenance.digest(path)
    if (digest != summary["vectors_sha256"]
            or protocol["input_sha256"] != sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()
            or protocol["encoder_sha256"] != CACHE_ENCODER_SHA256
            or protocol["checkpoint"] != embedding
            or protocol["max_length"] != 512):
        raise ValueError(f"Cached embedding provenance mismatch: {root.name}")
    matrix = np.load(path, allow_pickle=False, mmap_mode="r")
    if matrix.shape != (len(items), 1024) or matrix.dtype != np.float32:
        raise ValueError(f"Unexpected cached embedding shape or dtype: {root.name}")
    return matrix, digest


def top_eight(scores):
    boundary = np.partition(scores, -8)[-8]
    indices = np.flatnonzero(scores >= boundary)
    return indices[np.argsort(-scores[indices], kind="stable")[:8]]


def measure(dataset, encoder, checkpoint):
    root, papers, profiles, labels = load(dataset)
    manifest = json.loads((root / "manifest.json").read_bytes())
    owners = {}
    for reviewer, profile in profiles.items():
        for pid in set(profile["papers"]):
            owners.setdefault(pid, set()).add(reviewer)
    profile_ids, query_ids = sorted(owners), sorted(labels["soft"])
    matrix, digest = frozen_vectors(root, papers, profile_ids, query_ids, checkpoint)
    positions = [i for i, pid in enumerate(profile_ids) if papers[pid]["title"].strip()]
    source_ids = [profile_ids[i] for i in positions]
    if not source_ids:
        return [], dict(source_rows=0, profile_sources=len(profile_ids), excluded_reason="no_source_titles")
    keys = {prior.title_key(papers[pid]["title"]) for pid in source_ids}
    excluded = set(manifest["query_ids_in_profiles"] + manifest["exact_query_texts_in_profiles"])
    titles = encoder.encode([papers[pid]["title"] for pid in source_ids])
    full = matrix[positions]
    rows = []
    for index, qid in enumerate(query_ids):
        key = prior.title_key(papers[qid]["title"])
        if not key or qid in excluded or key in keys:
            continue
        vector = matrix[len(profile_ids) + index]
        scores = full @ vector
        ranked = top_eight(scores)
        first, second = ranked[:2]
        title_scores = titles @ vector
        title_scores[first] = -np.inf
        extra = np.argmax(title_scores)
        features = [
            scores[first], scores[first] - scores[second], scores[second] - scores[ranked[-1]],
            np.log1p(len(papers[qid]["abstract"].split())),
        ]
        outcomes = {}
        for rubric, cases in labels.items():
            gold = cases[qid]
            outcomes[rubric] = []
            for pair in ((first, second), (first, extra)):
                candidates = set.union(*(owners[source_ids[i]] for i in pair))
                outcomes[rubric].append([
                    int(any(gold.get(reviewer, 0) > 0 for reviewer in candidates)),
                    sum(reviewer in gold for reviewer in candidates),
                    len(candidates),
                ])
        rows.append(dict(title=key, features=features, outcomes=outcomes, changed=bool(extra != second)))
    return rows, dict(
        source_rows=len(source_ids), queries_before_cross_corpus_filter=len(rows),
        vectors_sha256=digest, manifest_sha256=provenance.digest(root / "manifest.json"),
    )


def predictors(train, test):
    x = np.array([row["features"] for row in train], dtype=np.float64)
    holdout = np.array([row["features"] for row in test], dtype=np.float64)
    center = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-12)
    normalized = (x - center) / scale
    delta = np.mean([
        [row["outcomes"][rubric][1][0] - row["outcomes"][rubric][0][0] for rubric in ("soft", "hard")]
        for row in train
    ], axis=1)
    weights = np.linalg.solve(normalized.T @ normalized + np.eye(x.shape[1]) * len(x) * 0.25,
                              normalized.T @ delta)
    train_prediction = delta.mean() + normalized @ weights
    utility = delta.mean() + ((holdout - center) / scale) @ weights
    minimum_utility = max(0., float(np.quantile(train_prediction, .75)))
    margin_cutoff = float(np.quantile(x[:, 1], .25))
    return {
        "never": np.zeros(len(test), dtype=bool),
        "always": np.ones(len(test), dtype=bool),
        "low_margin": holdout[:, 1] < margin_cutoff,
        "learned_gain": utility > minimum_utility,
    }, dict(utility_cutoff=minimum_utility, margin_cutoff=margin_cutoff, weights=weights.tolist())


def evaluate(rows, routes):
    rng = np.random.default_rng(20261120)
    sampled = rng.integers(0, len(rows), size=(10000, len(rows)))
    tables = {}
    for rubric in ("soft", "hard"):
        measurements = np.array([row["outcomes"][rubric] for row in rows])
        difference = measurements[:, 1, 0] - measurements[:, 0, 0]
        tables[rubric] = {}
        for name, gate in routes.items():
            observed = measurements[np.arange(len(rows)), gate.astype(int)]
            delta = difference * gate
            tables[rubric][name] = dict(
                scans=int(gate.sum()), known_positive_queries=int(observed[:, 0].sum()),
                judged_pairs=int(observed[:, 1].sum()), returned_pairs=int(observed[:, 2].sum()),
                gained=int((delta > 0).sum()), lost=int((delta < 0).sum()), net=int(delta.sum()),
                random_same_scan_count_expected_net=float(difference.mean() * gate.sum()),
                paired_percentile_interval=np.quantile(delta[sampled].mean(axis=1), [.025, .975]).tolist(),
            )
    return tables


def run(embedding, output):
    import torch

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require fresh output and Slurm CUDA")
    checkpoint = provenance.verify_checkpoint(embedding)
    encoder = QwenTorchEmbedding(embedding, max_length=128, batch_size=128)
    encoder.model.config.use_cache = False
    collected = {}
    metadata = {}
    for dataset in DATASETS:
        collected[dataset], metadata[dataset] = measure(dataset, encoder, checkpoint)
        print(f"{dataset}: {len(collected[dataset])} queries", flush=True)
    excluded_panels = {panel: metadata[panel] for panel, rows in collected.items() if not rows}
    title_counts = Counter(key for rows in collected.values() for key in {row["title"] for row in rows})
    panels = {panel: [row for row in rows if title_counts[row["title"]] == 1]
              for panel, rows in collected.items() if rows}
    def score_panel(panel, test, train):
        routes, fitting = predictors(train, test)
        return dict(
            **metadata[panel], queries=len(test), training_queries=len(train),
            changed_source_sets=sum(row["changed"] for row in test),
            fitting=fitting, reviewer_proxy=evaluate(test, routes),
        )

    results = {
        panel: score_panel(panel, test, [row for name, rows in panels.items() if name != panel for row in rows])
        for panel, test in panels.items()
    }
    grouped_results = {}
    for panel, test in collected.items():
        if not test:
            continue
        test_titles = {row["title"] for row in test}
        train = [row for name, rows in collected.items() if name != panel
                 for row in rows if row["title"] not in test_titles]
        grouped_results[panel] = score_panel(panel, test, train)
    provenance.write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        script_sha256=provenance.digest(Path(__file__)), embedding_revision=checkpoint["revision"],
        cached_encoder_sha256=CACHE_ENCODER_SHA256,
        known_exploratory_panel="SciRepEval", excluded_panels=excluded_panels, panel_results=results,
        training_only_overlap_filter=grouped_results,
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("embedding", type=Path)
    cli.add_argument("output", type=Path)
    run(**vars(cli.parse_args()))
