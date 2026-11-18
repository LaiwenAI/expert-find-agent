"""Compare unsupervised source and owner score calibration on CoF retrieval."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time

import numpy as np
import torch

from expert_search.cof_benchmark import load
from expert_search.statistics import paired_interval
from experiments.retrieval.cof_heldout_encoder_comparison import DATASET, exclusions, normalize_text
from experiments.retrieval.cof_online_ownership import EMBEDDING, TOP_K, prepare, serialize_hits, sha, text
from experiments.retrieval.cof_query_encoding_budget import QueryEncoderVariants, compare_labels

METHODS = ('dense', 'source_mean', 'source_top10', 'owner_mean')
TOP_ANCHORS = 10
COEFFICIENT = .5
REPEATS = 3


def anchors(encoder):
    root, papers, _, labels = load('KDD')
    inventory = json.loads((root / 'manifest.json').read_text())
    omit = set(inventory['query_ids_in_profiles']) | set(inventory['exact_query_texts_in_profiles'])
    ids = sorted(set(labels['soft']) - omit)
    values = [text(papers[key]) for key in ids]
    titles = {normalize_text(papers[key]['title']) for key in ids}
    encoder.batch_size = 16
    vectors = encoder.encode(values, query=True)
    encoder.batch_size = 1
    return ids, titles, vectors, hashlib.sha256(json.dumps(values).encode()).hexdigest()


def build_calibration(ranker, anchor_vectors):
    started = time.perf_counter()
    references = torch.as_tensor(anchor_vectors, device='cuda')
    with torch.inference_mode():
        all_scores = references @ ranker.vectors.T
        top = torch.topk(all_scores, TOP_ANCHORS, dim=0).values.mean(dim=0)
        source_mean = all_scores.mean(dim=0)
    scores = all_scores.cpu().numpy()
    index = ranker.index
    owner_mean = np.maximum.reduceat(scores[:, index.owner_rows], index.owner_starts, axis=1).mean(axis=0)
    result = {'source_mean': source_mean * COEFFICIENT, 'source_top10': top * COEFFICIENT,
              'owner_mean': owner_mean * COEFFICIENT}
    return result, time.perf_counter() - started


def rank_owned(ranker, scores, owner_penalty):
    if owner_penalty is None:
        return ranker.rank_host(scores)
    scores = scores.cpu().numpy()
    index = ranker.index
    maxima = np.maximum.reduceat(scores[index.owner_rows], index.owner_starts) - owner_penalty
    selected = np.argsort(-maxima, kind='stable')[:TOP_K]
    hits = []
    for owner in selected:
        rows = index.owners[index.owner_ids[owner]]
        witness = rows[int(np.argmax(scores[rows]))]
        hits.append((int(owner), int(witness), float(maxima[owner])))
    return hits


def request(ranker, connection, encoder, query, bias, method):
    started = time.perf_counter()
    vector = encoder.encode([query], query=True)[0]
    scores = ranker.score(vector)
    if method.startswith('source_'):
        scores = scores - bias
    hits = rank_owned(ranker, scores, bias if method == 'owner_mean' else None)
    body, experts = serialize_hits(ranker, connection, hits)
    elapsed = (time.perf_counter() - started) * 1000
    citations = [(r['expert'], r['source']) for r in json.loads(body)]
    return experts, citations, elapsed


def experiment(ranker, db, encoder, queries, labels, penalties):
    rows = []
    with sqlite3.connect(db) as connection:
        for method in METHODS:
            request(ranker, connection, encoder, next(iter(queries.values())), penalties.get(method), method)
        for i, (qid, query) in enumerate(queries.items()):
            trials = {method: [] for method in METHODS}
            for j in range(REPEATS):
                offset = (i + j) % len(METHODS)
                for method in METHODS[offset:] + METHODS[:offset]:
                    trials[method].append(request(ranker, connection, encoder, query, penalties.get(method), method))
            baseline = trials['dense'][0][0]
            row = {}
            for method, values in trials.items():
                experts, citations, _ = values[0]
                if any((e, c) != (experts, citations) for e, c, _ in values):
                    raise ValueError('Single-query ranking changed across repeats')
                row[method] = {
                    'experts': experts, 'citations': citations, 'ms': [v[2] for v in values],
                    'labels': {key: compare_labels(baseline, experts, grade[qid]) for key, grade in labels.items()},
                }
            rows.append({'id': qid, 'methods': row})
    return rows


def summarize(rows):
    result = {}
    for method in METHODS:
        values = [r['methods'][method] for r in rows]
        timings = [ms for arm in values for ms in arm['ms']]
        p50, p95 = np.quantile(timings, [.5, .95])
        result[method] = {'p50_ms': float(p50), 'p95_ms': float(p95),
                          'same_rank_and_citations': sum(
                              r['methods'][method]['citations'] == r['methods']['dense']['citations'] for r in rows),
                          'unique_returned_sources': len({s for arm in values for _, s in arm['citations']}),
                          'most_frequent_source_count': max(np.unique(
                              [s for arm in values for _, s in arm['citations']], return_counts=True)[1]).item(),
                          'labels': {}}
        for key in ('soft', 'hard'):
            grades = [v['labels'][key] for v in values]
            report = {k: float(np.mean([g[k] for g in grades])) for k in grades[0]}
            report['bootstrap_known_delta'] = paired_interval([g['known_difference'] for g in grades], repeats=4000)
            result[method]['labels'][key] = report
    return result


def run(output):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Requires Slurm CUDA')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    checkpoint = EMBEDDING / 'checkpoint-provenance.json'
    for name, expected in json.loads(checkpoint.read_text())['files'].items():
        if sha(EMBEDDING / name) != expected:
            raise ValueError('Changed encoder')
    variant = QueryEncoderVariants()
    variant.configure('no_cache')
    encoder = variant.encoder
    started = time.perf_counter()
    anchor_ids, anchor_titles, vectors, anchor_sha = anchors(encoder)
    anchor_seconds = time.perf_counter() - started
    anchor_ids = set(anchor_ids)
    omitted_queries, omitted_sources, _ = exclusions()
    output.mkdir(parents=True, exist_ok=False)
    report = {'job_id': os.environ['SLURM_JOB_ID'], 'gpu': torch.cuda.get_device_name(),
              'model_sha256': sha(checkpoint), 'code_sha256': sha(Path(__file__)),
              'calibration': {'dataset': 'KDD', 'queries': len(anchor_ids), 'text_sha256': anchor_sha,
                              'encoding_seconds': anchor_seconds, 'top_anchors': TOP_ANCHORS,
                              'coefficient': COEFFICIENT}, 'datasets': {}}
    for dataset in ('SIGIR', DATASET):
        _, papers, _, qrels = load(dataset)
        remove_q = set(omitted_queries) if dataset == DATASET else set()
        remove_s = set(omitted_sources) if dataset == DATASET else set()
        overlap = {pid for pid, paper in papers.items()
                   if pid in anchor_ids or normalize_text(paper['title']) in anchor_titles}
        remove_q.update(overlap & qrels['soft'].keys())
        remove_s.update(overlap)
        ranker, db, queries, labels, inventory = prepare(
            dataset, output, checkpoint, exclude_queries=remove_q, exclude_sources=remove_s)
        try:
            penalties, seconds = build_calibration(ranker, vectors)
            rows = experiment(ranker, db, encoder, queries, labels, penalties)
            path = output / f'{dataset}-rows.json'
            path.write_text(json.dumps(rows, indent=2, allow_nan=False) + '\n')
            report['datasets'][dataset] = {'index': inventory, 'rows_sha256': sha(path),
                                           'offline_calibration_seconds': seconds,
                                           'penalty_bytes': {k: v.nbytes if isinstance(v, np.ndarray)
                                                             else v.element_size() * v.numel()
                                                             for k, v in penalties.items()},
                                           'methods': summarize(rows)}
        finally:
            db.unlink()
        del ranker
    report['scope'] = 'No calibration relevance grades. Source means and top-ten means use unrelated KDD query text. Owner mean averages maximum per-owner source scores over the same queries. Original postfilter benchmark relevance may be incomplete. Timings include encoding, ranking and citation lookup; no semantic verification or service.'
    (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
