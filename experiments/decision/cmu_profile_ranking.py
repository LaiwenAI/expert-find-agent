"""Grouped out-of-fold expert-fit scoring with resident dense profiles."""

import argparse
from collections import defaultdict
from hashlib import sha256
from itertools import combinations
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from expert_search.cmu_gold import pairwise_loss
from expert_search.retrieval import tokens
from experiments.decision.cmu_bounded_reranking import read_dataset


FOLDS = 5
ARMS = ("maximum", "centroid", "top_three", "mean_max", "learned_pooling", "learned_relative")
BACKGROUND_FLOOR = 0.025


def grouped_folds(cases, queries):
    memberships = defaultdict(list)
    parent = {reviewer: reviewer for reviewer, _, _ in cases}

    def find(reviewer):
        while parent[reviewer] != reviewer:
            parent[reviewer] = parent[parent[reviewer]]
            reviewer = parent[reviewer]
        return reviewer

    titles = {}
    for reviewer, paper, _ in cases:
        title = tuple(tokens(queries[paper].split("\n", 1)[0]))
        first = titles.setdefault(title, reviewer)
        a, b = sorted((find(first), find(reviewer)))
        parent[b] = a
    for i, (reviewer, _, _) in enumerate(cases):
        memberships[find(reviewer)].append(i)
    groups = sorted(memberships.values(), key=lambda ids: (-len(ids), ids[0]))
    folds = [[] for _ in range(FOLDS)]
    sizes = [0] * FOLDS
    for group in groups:
        fold = sizes.index(min(sizes))
        folds[fold].append(group)
        sizes[fold] += len(group)
    if min(sizes) == 0:
        raise ValueError("Not enough independent components")
    return folds


def make_features(owned, centroid, query, background=None):
    scores = owned @ query
    top = np.partition(scores, -min(len(scores), 3))[-3:]
    maximum = float(top.max())
    top_three = float(top.mean())
    center = float(centroid @ query)
    values = [maximum, center, top_three, float(scores.std())]
    if background is None:
        return values
    reference = background @ query
    location = float(reference.mean())
    spread = max(float(reference.std()), BACKGROUND_FLOOR)
    upper = float(np.quantile(reference, 0.5 ** (1 / len(scores))))
    return values + [location, (maximum - upper) / spread, (top_three - location) / spread,
                     (float(scores.mean()) - location) / spread]


def preference_data(indices, features, cases, labels):
    reviewers = defaultdict(list)
    for i in indices:
        reviewers[cases[i][0]].append(i)
    examples, answers = [], []
    for reviewer, group in reviewers.items():
        for first, second in combinations(group, 2):
            delta = labels[reviewer][cases[first][1]] - labels[reviewer][cases[second][1]]
            if delta == 0:
                continue
            diff = features[first] - features[second]
            examples.extend((diff, -diff))
            answers.extend((delta > 0, delta < 0))
    return np.stack(examples), np.asarray(answers, dtype=np.float64)


def pair_records(predictions, labels, component):
    grouped = defaultdict(lambda: [0., 0.])
    for reviewer, scores in predictions.items():
        bucket = grouped[component[reviewer]]
        for a, b in combinations(scores, 2):
            delta = labels[reviewer][a] - labels[reviewer][b]
            if not delta:
                continue
            sign = np.sign(scores[a] - scores[b])
            bucket[0] += abs(delta) * (0.5 if sign == 0 else float(sign * delta < 0))
            bucket[1] += abs(delta)
    return grouped


def paired_interval(arm, reference, labels, component):
    records = pair_records(arm, labels, component)
    baseline = pair_records(reference, labels, component)
    groups = sorted(records)
    difference = np.array([records[g][0] - baseline[g][0] for g in groups])
    weights = np.array([records[g][1] for g in groups])
    rng = np.random.default_rng(20261119)
    resample = rng.integers(len(groups), size=(10000, len(groups)))
    estimates = difference[resample].sum(axis=1) / weights[resample].sum(axis=1)
    return dict(delta=float(difference.sum() / weights.sum()),
                quantiles=np.quantile(estimates, [.025, .975]).tolist(), components=len(groups))


def run(root, embedding, output):
    import torch
    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.decision.merit_lightweight_decision import fit, logit
    from experiments.followir.core17_reranking import write_json
    from experiments.litsearch.dense_cache import verify_checkpoint

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require new output and a Slurm CUDA allocation")

    cases, annotations, queries, published = read_dataset(root)
    folds = grouped_folds(cases, queries)
    components = {cases[i][0]: ids[0] for fold in folds for ids in fold for i in ids}
    encoder_meta = verify_checkpoint(embedding)
    encoder = QwenTorchEmbedding(embedding, batch_size=24)
    encoder.model.config.use_cache = False
    start = perf_counter()
    publication_ids = sorted(published)
    vectors = dict(zip(publication_ids, encoder.encode([published[doc] for doc in publication_ids]), strict=True))
    resident = {}
    for owned in {row[2] for row in cases}:
        rows = np.stack([vectors[doc] for doc in owned])
        centroid = rows.mean(axis=0)
        resident[owned] = (rows, centroid / np.linalg.norm(centroid))
    index_time = perf_counter() - start
    encoder.batch_size = 1

    def encode(paper):
        return encoder.encode([queries[paper]], query=True)[0]

    predictions = {arm: {reviewer: {} for reviewer in annotations} for arm in ARMS}
    times = defaultdict(list)
    fold_reports = []
    for groups in folds:
        held = sorted(i for group in groups for i in group)
        fit_indices = sorted(set(range(len(cases))) - set(held))
        fit_documents = sorted({doc for i in fit_indices for doc in cases[i][2]}
                               - {cases[i][1] for i in held})
        held_titles = {tuple(tokens(queries[cases[i][1]].split("\n", 1)[0])) for i in held}
        fit_documents = [doc for doc in fit_documents
                         if tuple(tokens(published[doc].split("\n", 1)[0])) not in held_titles]
        background = np.stack([vectors[doc] for doc in fit_documents])

        fit_started = perf_counter()
        features = {}
        for i in fit_indices:
            _, paper, owned = cases[i]
            features[i] = np.asarray(make_features(*resident[owned], encode(paper), background))
        pair_features, pair_labels = preference_data(fit_indices, features, cases, annotations)
        raw_fit = fit(pair_features[:, :4], pair_labels)
        relative_fit = fit(pair_features, pair_labels)
        fit_seconds = perf_counter() - fit_started

        def request(paper, owned, arm):
            start = perf_counter()
            query = encode(paper)
            rows, center = resident[owned]
            if arm == "maximum":
                value = (rows @ query).max()
            elif arm == "centroid":
                value = center @ query
            elif arm == "mean_max":
                value = 0.5 * ((rows @ query).max() + center @ query)
            elif arm == "top_three":
                scores = rows @ query
                value = np.partition(scores, -min(len(scores), 3))[-3:].mean()
            elif arm == "learned_pooling":
                value = logit(make_features(rows, center, query), raw_fit)
            else:
                value = logit(make_features(rows, center, query, background), relative_fit)
            return float(value), 1000 * (perf_counter() - start)

        for arm in ARMS:
            request(*cases[fit_indices[0]][1:], arm)
        for offset, i in enumerate(held):
            reviewer, paper, owned = cases[i]
            index = offset % len(ARMS)
            for arm in ARMS[index:] + ARMS[:index]:
                value, elapsed = request(paper, owned, arm)
                predictions[arm][reviewer][paper] = value
                times[arm].append(elapsed)
        holdout = {cases[i][0]: annotations[cases[i][0]] for i in held}
        fold_reports.append(dict(
            train=len(fit_indices), held=len(held), held_reviewers=len(holdout), held_components=len(groups),
            background_publications=len(fit_documents), fit_seconds=fit_seconds,
            loss={arm: pairwise_loss(predictions[arm], holdout)["weighted_pairwise_loss"] for arm in ARMS},
        ))
        print("Completed fold", len(fold_reports), flush=True)
    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        projection_sha256=sha256(json.dumps([cases, annotations, queries, published],
                                             sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        embedding_revision=encoder_meta["revision"], reviewers=len(annotations), cases=len(cases),
        publications=len(published), index_seconds=index_time, folds=fold_reports,
        results={arm: pairwise_loss(predictions[arm], annotations) | dict(
            p50_p95_ms=np.quantile(times[arm], [.5, .95]).tolist(),
            delta_vs_maximum=paired_interval(predictions[arm], predictions["maximum"], annotations, components),
            delta_vs_learned_pooling=paired_interval(predictions[arm], predictions["learned_pooling"],
                                                     annotations, components),
        ) for arm in ARMS},
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "embedding", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
