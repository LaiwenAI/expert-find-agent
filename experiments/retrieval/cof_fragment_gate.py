"""Measure budgeted fragment scoring with venue-separated calibration."""

import argparse
import json
import os
from pathlib import Path
import sqlite3
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from expert_search.cof_benchmark import load
from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import paired_interval
from experiments.retrieval.cof_complementary_source_pairs import deliver, facets, owned_candidates, select
from experiments.retrieval.cof_online_ownership import EMBEDDING, TOP_K, prepare, sha
from experiments.retrieval.cof_query_encoding_budget import compare_labels

CALIBRATION = ('SIGIR', 'KDD')
TEST = ('NIPS', 'SciRepEval')
RATE = .25
SEED = 20261123
FEATURES = ('query_terms', 'candidate_coverage', 'top_score', 'last_kept_score', 'boundary_gap',
            'first_gap', 'top5_spread', 'initial_mean_profile_sources')


def normalized(text):
    return ' '.join(text.casefold().split())


def overlaps(dataset, excluded_ids, excluded_titles):
    root, papers, profiles, labels = load(dataset)
    manifest = json.loads((root / 'manifest.json').read_text())
    sources = {p for r in profiles.values() for p in r['papers']}
    present = set(manifest['query_ids_in_profiles']) | set(manifest['exact_query_texts_in_profiles'])
    source_titles = {normalized(papers[p]['title']) for p in sources}
    q = {p for p in labels['soft'] if p in present | excluded_ids | sources
         or normalized(papers[p]['title']) in source_titles | excluded_titles}
    blocked_sources = {p for p in sources if p in excluded_ids
                       or normalized(papers[p]['title']) in excluded_titles}
    return q, blocked_sources


def cheap_features(text, pools, compact, index):
    scores = [score for _, score, _ in pools]
    return np.array([len(text.split()), len(compact), scores[0], scores[4], scores[4] - scores[5],
                     scores[0] - scores[1], scores[0] - scores[4],
                     mean(len(index.owners[owner]) for owner, _, _ in pools[:TOP_K])], dtype=np.float64)


def build_arm(method, ranker, conn, encoder, text, pools, compact):
    started = perf_counter()
    matrix = None
    fragments = facets(text) if method == 'pair_bottleneck' else []
    if fragments:
        subqueries = torch.as_tensor(encoder.encode(fragments, query=True), device='cuda')
        docs = ranker.vectors.index_select(0, torch.as_tensor(compact, device='cuda'))
        matrix = (docs @ subqueries.T).cpu().numpy()
    ranking = select(pools, compact, matrix, method)[:TOP_K]
    payload, citations, chars = deliver(ranking, ranker, conn)
    elapsed = (perf_counter() - started) * 1000
    if not payload:
        raise ValueError('Empty output')
    return {'experts': [row[0] for row in ranking], 'citations': citations,
            'characters': chars, 'branch_ms': elapsed}


def corpus(dataset, out, checkpoint, encoder, excluded_ids, excluded_titles):
    omit, removed = overlaps(dataset, excluded_ids, excluded_titles)
    ranker, db, queries, qrels, inventory = prepare(dataset, out, checkpoint, exclude_queries=omit,
                                                   exclude_sources=removed)
    records = []
    try:
        with sqlite3.connect(db) as connection:
            for i, (key, text) in enumerate(sorted(queries.items())):
                started = perf_counter()
                pools, compact = owned_candidates(ranker, encoder.encode([text], query=True)[0])
                ready = perf_counter()
                features = cheap_features(text, pools, compact, ranker.index)
                gate_features_ms = (perf_counter() - ready) * 1000
                order = ('dense', 'pair_bottleneck')
                arms = {m: build_arm(m, ranker, connection, encoder, text, pools, compact)
                        for m in (order if i % 2 == 0 else order[::-1])}
                comparisons = {
                    grade: compare_labels(arms['dense']['experts'], arms['pair_bottleneck']['experts'], labels[key])
                    for grade, labels in qrels.items()
                }
                records.append({'query': key, 'features': features.tolist(), 'base_ms': (ready - started) * 1000,
                                'feature_ms': gate_features_ms, 'arms': arms, 'comparisons': comparisons})
    finally:
        db.unlink()
    _, papers, _, _ = load(dataset)
    inventory.update({'excluded_queries': len(omit), 'excluded_sources': len(removed)})
    return records, inventory, set(queries), {normalized(papers[q]['title']) for q in queries}


def learn(rows):
    # Unjudged changes always remain uncertain. Only learn from identified hard-label differences.
    identified = [r for r in rows if r['comparisons']['hard']['difference_lower']
                  == r['comparisons']['hard']['difference_upper']]
    if len(identified) <= len(FEATURES):
        raise ValueError('Too few identified outcomes for the fixed gate')
    x = np.array([r['features'] for r in identified])
    y = np.array([r['comparisons']['hard']['known_difference'] for r in identified])
    center, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-8)
    aligned = np.column_stack((np.ones(len(x)), (x - center) / scale))
    inverse = np.linalg.inv(aligned.T @ aligned + np.diag([0.] + [10.] * x.shape[1]))
    weights = inverse @ (aligned.T @ y)
    prediction = aligned @ weights
    leverage = ((aligned @ inverse) * aligned).sum(axis=1)
    held = (prediction - leverage * y) / (1 - leverage)
    # Apply the fixed threshold to unseen venues. Use all raw margins for the simple gate.
    gap_threshold = float(np.quantile([r['features'][4] for r in rows], RATE))
    return {'center': center, 'scale': scale, 'weights': weights,
            'learned_threshold': float(np.quantile(held, 1 - RATE)), 'gap_threshold': gap_threshold,
            'identified_count': len(identified), 'identified_positive': int((y > 0).sum()),
            'identified_negative': int((y < 0).sum())}


def evaluate(rows, policy):
    x = np.asarray([r['features'] for r in rows])
    score_started = perf_counter()
    signs = policy['weights'][0] + ((x - policy['center']) / policy['scale']) @ policy['weights'][1:]
    score_ms_per_query = (perf_counter() - score_started) * 1000 / len(rows)
    learned = signs > policy['learned_threshold']
    margin = x[:, 4] <= policy['gap_threshold']
    count = int(learned.sum())
    matched = np.zeros(len(rows), dtype=bool)
    matched[np.argsort(x[:, 4], kind='stable')[:count]] = True
    methods = {'flat': np.zeros(len(rows), dtype=bool), 'always_fragment': np.ones(len(rows), dtype=bool),
               'frozen_margin': margin, 'frozen_learned': learned, 'matched_rate_margin': matched}
    outputs = {}
    for method, selected in methods.items():
        delta = {}
        for grade in ('soft', 'hard'):
            observed = [r['comparisons'][grade] for r in rows]
            delta[grade] = {
                k: mean(value[k] * chosen for value, chosen in zip(observed, selected, strict=True))
                for k in ('known_difference', 'difference_lower', 'difference_upper')
            }
            delta[grade]['paired_known_delta'] = paired_interval(
                [v['known_difference'] * chosen for v, chosen in zip(observed, selected, strict=True)], repeats=4000)
        ms = [r['base_ms'] + r['arms']['pair_bottleneck' if chosen else 'dense']['branch_ms']
              + (r['feature_ms'] if method not in ('flat', 'always_fragment') else 0.)
              + (score_ms_per_query if method == 'frozen_learned' else 0.)
              for r, chosen in zip(rows, selected, strict=True)]
        outputs[method] = {'fragment_calls': int(selected.sum()), 'mean_ms': mean(ms),
                           'p95_ms': float(np.quantile(ms, .95)), 'delta_vs_flat': delta}
    rng = np.random.default_rng(SEED)
    hard = np.array([r['comparisons']['hard']['known_difference'] for r in rows])
    samples = [hard[rng.choice(len(rows), count, replace=False)].sum() / len(rows) for _ in range(2000)]
    switch = learned.astype(int) - matched.astype(int)
    signed = np.asarray([[r['comparisons']['hard'][k] for k in ('difference_lower', 'difference_upper')]
                         for r in rows]) * switch[:, None]
    direct = {
        'difference_lower': float(signed.min(axis=1).mean()),
        'difference_upper': float(signed.max(axis=1).mean()),
        'paired_known_delta': paired_interval(hard * switch, repeats=4000),
    }
    return {'arms': outputs, 'learned_minus_matched_margin_hard': direct, 'matched_random_hard_known_delta': {
        'mean': mean(samples), 'central_95': np.quantile(samples, [.025, .975]).tolist(),
    }, 'optimistic_known_oracle_up_to_matched_rate': float(np.maximum(np.sort(hard)[len(hard) - count:], 0).sum()
                                                      / len(rows))}


def run(out):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Requires Slurm CUDA')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    checkpoint = EMBEDDING / 'checkpoint-provenance.json'
    for name, expected in json.loads(checkpoint.read_text())['files'].items():
        if sha(EMBEDDING / name) != expected:
            raise ValueError('Changed embedding checkpoint')
    out.mkdir(parents=True, exist_ok=False)
    model = QwenTorchEmbedding(batch_size=4)
    model.model.config.use_cache = False
    model.encode(['warm up', 'fixed candidate ranking', 'evidence decision'], query=True)
    train, train_info = [], {}
    query_ids, query_titles = set(), set()
    for d in CALIBRATION:
        rows, index, ids, titles = corpus(d, out, checkpoint, model, query_ids, query_titles)
        train.extend(rows)
        train_info[d] = index
        query_ids.update(ids)
        query_titles.update(titles)
    policy = learn(train)
    full = {}
    for d in TEST:
        rows, index, _, _ = corpus(d, out, checkpoint, model, query_ids, query_titles)
        path = out / f'{d}-rows.json'
        path.write_text(json.dumps(rows, indent=2, allow_nan=False) + '\n')
        full[d] = {'index': index, 'rows_sha256': sha(path), **evaluate(rows, policy)}
    train_path = out / 'calibration-rows.json'
    train_path.write_text(json.dumps(train, indent=2, allow_nan=False) + '\n')
    result = {
        'job_id': os.environ['SLURM_JOB_ID'], 'gpu': torch.cuda.get_device_name(),
        'source_sha256': sha(Path(__file__)), 'model_sha256': sha(checkpoint), 'features': FEATURES,
        'calibration': train_info, 'calibration_sha256': sha(train_path), 'target_rate': RATE,
        'policy': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in policy.items()},
        'evaluation': full,
        'scope': 'Exploratory replay on reused CoF content. Calibration and evaluation exclude matching '
                 'query IDs and normalized titles. Profiles and source articles can still overlap. '
                 'Ridge and latency are evaluated from recorded counterfactual branches. Training omits '
                 'hard-label differences that are not identified. Matched-rate comparison uses only '
                 'test score margins and the learned policy count. Timing adds measured costs from the '
                 'chosen branch; it is not a separate serving run. No citation support or enterprise gold.',
    }
    (out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
