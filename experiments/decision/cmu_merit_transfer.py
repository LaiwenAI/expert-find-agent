"""Test transfer of CMU within-reviewer preference heads to judged MERIT candidates."""

import argparse
from collections import defaultdict
from hashlib import sha256
from itertools import combinations
import json
import os
from pathlib import Path

import numpy as np

from expert_search.retrieval import tokens
from experiments.decision.cmu_bounded_reranking import read_dataset
from experiments.decision.cmu_profile_ranking import make_features, preference_data


ARMS = ("maximum", "centroid", "top_three", "mean_max", "learned_pooling", "learned_relative")


def title(text):
    return tuple(tokens(text.split("\n", 1)[0]))


def prepare_external(cases, source_titles, source_profiles):
    duplicates = defaultdict(set)
    profiles = {}
    queries = {}
    excluded = 0
    for case in cases:
        key = title(case.query)
        # When records repeat a target and exact profile text, treat them as one view.
        sources = tuple(sorted({doc for doc in case.papers if title(doc) != key}))
        if (key in source_titles or tuple(sorted(title(doc) for doc in sources)) in source_profiles
                or len(sources) < 2):
            excluded += 1
            continue
        target = key, sources
        queries[target] = min(queries.get(target, case.query), case.query)
        profiles[target] = sources
        duplicates[target].add(case.label)
    conflicting = sum(len(labels) != 1 for labels in duplicates.values())
    return [(key, queries[key, sources], profiles[key, sources], next(iter(labels)))
            for (key, sources), labels in sorted(duplicates.items()) if len(labels) == 1], excluded, conflicting


def pair_totals(scores, labels):
    error = 0.
    pairs = 0
    for (score_a, label_a), (score_b, label_b) in combinations(zip(scores, labels, strict=True), 2):
        if label_a == label_b:
            continue
        signed = (score_a - score_b) * (label_a - label_b)
        error += float(signed < 0) if signed else .5
        pairs += 1
    return error, pairs


def metrics(rows, scores):
    groups = defaultdict(list)
    classes = [row[3] for row in rows]
    for i, row in enumerate(rows):
        groups[row[0]].append(i)
    mixed = [indices for indices in groups.values() if len({classes[i] for i in indices}) == 2]
    errors, pairs = zip(*(pair_totals([scores[i] for i in selected], [classes[i] for i in selected])
                          for selected in mixed), strict=True)
    top = []
    for selected in mixed:
        best = max(scores[i] for i in selected)
        winners = [classes[i] for i in selected if scores[i] == best]
        top.append(float(np.mean(winners)))
    total, count = pair_totals(scores, classes)
    return dict(global_pairwise_loss=total / count, global_pairs=count,
                within_target_loss=sum(errors) / sum(pairs), within_target_pairs=sum(pairs),
                mixed_targets=len(mixed), top_one_on_mixed_targets=float(np.mean(top)))


def run(root, merit, embedding, output):
    import torch
    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    from experiments.decision.merit_evidence_dispatch import read_cases
    from experiments.decision.merit_lightweight_decision import fit, logit
    from experiments.followir.core17_reranking import digest, write_json
    from experiments.litsearch.dense_cache import verify_checkpoint

    if output.exists() or not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("Require new output and Slurm CUDA")
    cases, annotations, queries, documents = read_dataset(root)
    cmu_titles = {title(doc) for doc in (*queries.values(), *documents.values())}
    cmu_profiles = {tuple(sorted(title(documents[d]) for d in owned)) for _, _, owned in cases}
    rows, excluded, conflicting = prepare_external(read_cases(merit), cmu_titles, cmu_profiles)
    provenance = verify_checkpoint(embedding)
    model = QwenTorchEmbedding(embedding, batch_size=24)
    model.model.config.use_cache = False
    source_texts = sorted(set(documents.values()) | {doc for _, _, docs, _ in rows for doc in docs})
    source_vectors = dict(zip(source_texts, model.encode(source_texts), strict=True))
    query_texts = sorted(set(queries.values()) | {query for _, query, _, _ in rows})
    query_vectors = dict(zip(query_texts, model.encode(query_texts, query=True), strict=True))
    background = np.stack([source_vectors[doc] for doc in dict.fromkeys(documents.values())])

    def features(owned, query):
        vectors = np.stack([source_vectors[doc] for doc in owned])
        centroid = vectors.mean(axis=0)
        return make_features(vectors, centroid / np.linalg.norm(centroid), query_vectors[query], background)

    cmu_features = {i: np.asarray(features([documents[doc] for doc in owned], queries[pid]))
                    for i, (_, pid, owned) in enumerate(cases)}
    pair_features, labels = preference_data(range(len(cases)), cmu_features, cases, annotations)
    raw = fit(pair_features[:, :4], labels)
    relative = fit(pair_features, labels)
    scores = defaultdict(list)
    for _, query, owned, _ in rows:
        view = features(owned, query)
        values = (*view[:3], .5 * (view[0] + view[1]), logit(view[:4], raw), logit(view, relative))
        for arm, value in zip(ARMS, values, strict=True):
            scores[arm].append(value)
    write_json(output, dict(
        job=os.environ["SLURM_JOB_ID"], gpu=torch.cuda.get_device_name(),
        script_sha256=digest(Path(__file__)), profile_ranker_sha256=digest(Path(__file__).with_name("cmu_profile_ranking.py")),
        cmu_projection_sha256=sha256(json.dumps([cases, annotations, queries, documents],
                                                sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        merit_test_sha256=digest(merit), embedding_revision=provenance["revision"],
        cmu_fit_cases=len(cases), merit_candidate_views=len(rows),
        merit_targets=len({row[0] for row in rows}), merit_positive_views=sum(row[3] for row in rows),
        excluded_before_dedup=excluded, conflicting_duplicate_views=conflicting,
        results={arm: metrics(rows, scores[arm]) for arm in ARMS},
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "merit", "embedding", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
