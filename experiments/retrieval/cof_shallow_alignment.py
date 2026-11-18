"""Align shallow query vectors to a frozen full-depth document index."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sqlite3
from statistics import mean
import time
import unicodedata
from zipfile import ZipFile

import numpy as np
import torch

from experiments.retrieval.cof_online_ownership import EMBEDDING, prepare, request, sha, text
from experiments.retrieval.cof_query_encoding_budget import QueryEncoderVariants, compare_labels
from expert_search.cof_benchmark import load
from expert_search.cof_data import SHA256
from expert_search.statistics import paired_interval

ARMS = {
    'full': ('no_cache', False),
    'raw_21': ('layers_21', False),
    'aligned_21': ('layers_21', True),
    'raw_14': ('layers_14', False),
    'aligned_14': ('layers_14', True),
}
ALPHAS = (.001, .01, .1)
SEED = 20261118
REPEATS = 3
FIT_ROWS = 3072
CALIBRATION_ROWS = 512


def normalize_text(value):
    return re.sub(r'\W+', '', unicodedata.normalize('NFKC', value).casefold())


def public_training_texts():
    archive = Path('artifacts/public/cof/data.zip')
    if sha(archive) != SHA256:
        raise ValueError('Unexpected CoF archive')
    excluded_keys = set()
    excluded_titles = set()
    for dataset in ('SIGIR', 'NIPS'):
        _, papers, _, _ = load(dataset)
        excluded_keys.update(papers)
        excluded_titles.update(normalize_text(p['title']) for p in papers.values() if p.get('title'))
    candidates = {}
    with ZipFile(archive) as source, source.open('data/KDD_papers_test.json') as stream:
        for line in stream:
            item = json.loads(line)
            title = normalize_text(item.get('title', ''))
            if not title or not item.get('abstract') or item['paper'] in excluded_keys or title in excluded_titles:
                continue
            value = text(item)
            old = candidates.get(title)
            if old is None or value < old:
                candidates[title] = value
    ordered = sorted(candidates, key=lambda title: hashlib.sha256(f'{SEED}:{title}'.encode()).digest())
    if len(ordered) < FIT_ROWS + CALIBRATION_ROWS:
        raise ValueError('Insufficient disjoint training papers')
    return [candidates[t] for t in ordered[:FIT_ROWS + CALIBRATION_ROWS]]


def unit(vectors):
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if not np.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError('Invalid projection norm')
    return vectors / norms


class AlignedEncoder:
    def __init__(self, raw, weights, bias):
        self.raw = raw
        self.weights = weights
        self.bias = bias

    def encode(self, texts, query=False):
        return unit(self.raw.encode(texts, query=query) @ self.weights + self.bias)


def fit(training, variants, output):
    encoder = variants.encoder
    encoder.batch_size = 16
    started = time.perf_counter()
    variants.configure('no_cache')
    teacher = encoder.encode(training, query=True)
    embedding_elapsed = time.perf_counter() - started
    target_fit = teacher[:FIT_ROWS].astype(np.float64)
    target_calibration = teacher[FIT_ROWS:]
    y_mean = target_fit.mean(axis=0)
    projections = {}
    weights_for_save = {}
    details = {}
    for mode in ('layers_21', 'layers_14'):
        variants.configure(mode)
        start = time.perf_counter()
        shallow = encoder.encode(training, query=True)
        embedding_elapsed += time.perf_counter() - start
        x_fit = shallow[:FIT_ROWS].astype(np.float64)
        x_mean = x_fit.mean(axis=0)
        x_fit -= x_mean
        gram = x_fit.T @ x_fit
        cross = x_fit.T @ (target_fit - y_mean)
        best = None
        scores = {}
        for alpha in ALPHAS:
            weights = np.linalg.solve(gram + alpha * np.eye(gram.shape[0]), cross)
            bias = y_mean - x_mean @ weights
            weights = weights.astype(np.float32)
            bias = bias.astype(np.float32)
            value = float(np.sum(unit(shallow[FIT_ROWS:] @ weights + bias) * target_calibration, axis=1).mean())
            scores[str(alpha)] = value
            if best is None or value > best[0]:
                best = value, alpha, weights, bias
        _, alpha, weights, bias = best
        projections[mode] = AlignedEncoder(encoder, weights, bias)
        weights_for_save[f'{mode}_weights'] = weights
        weights_for_save[f'{mode}_bias'] = bias
        details[mode] = {
            'chosen_alpha': alpha, 'calibration_cosine_by_alpha': scores,
            'raw_calibration_cosine': float((shallow[FIT_ROWS:] * target_calibration).sum(axis=1).mean()),
        }
    encoder.batch_size = 1
    path = output / 'projections.npz'
    np.savez(path, **weights_for_save)
    return projections, {
        'fit_papers': FIT_ROWS, 'calibration_papers': CALIBRATION_ROWS,
        'embedding_seconds': embedding_elapsed, 'total_seconds': time.perf_counter() - started,
        'training_queries_sha256': hashlib.sha256(json.dumps(training).encode()).hexdigest(),
        'projector_sha256': sha(path), 'details': details,
    }


def benchmark(variants, projections, ranker, connection, queries, labels):
    payloads = {qid: json.dumps({'query': query}, ensure_ascii=False) for qid, query in queries.items()}

    def run(arm, qid):
        mode, aligned = ARMS[arm]
        variants.configure(mode)
        encoder = projections[mode] if aligned else variants.encoder
        return request(ranker, connection, encoder, payloads[qid], 'host_reduction')

    for arm in ARMS:
        run(arm, next(iter(queries)))
    rng = random.Random(SEED)
    order = list(ARMS)
    results = {q: {arm: {'timings': []} for arm in ARMS} for q in queries}
    for repeat in range(REPEATS):
        cohort = list(queries)
        rng.shuffle(cohort)
        for i, qid in enumerate(cohort):
            offset = (i + repeat) % len(order)
            for arm in order[offset:] + order[:offset]:
                payload, rank, timing = run(arm, qid)
                row = results[qid][arm]
                fingerprint = hashlib.sha256(payload).hexdigest()
                if row.get('response_sha256', fingerprint) != fingerprint:
                    raise ValueError(f'Unstable response: {qid}/{arm}')
                row.update(rank=rank, response_sha256=fingerprint)
                row['timings'].append(timing)
    for qid, arms in results.items():
        ref = arms['full']['rank']
        for row in arms.values():
            row['top5_overlap'] = len(set(ref) & set(row['rank'])) / len(ref)
            row['quality'] = {task: compare_labels(ref, row['rank'], by_query[qid])
                              for task, by_query in labels.items()}
    return results


def summarize(rows):
    reports = {}
    for arm in ARMS:
        data = [value[arm] for value in rows.values()]
        baseline = [value['full'] for value in rows.values()]
        timings = [1000 * timing['total'] for r in data for timing in r['timings']]
        quality = {}
        for task in data[0]['quality']:
            records = [r['quality'][task] for r in data]
            quality[task] = {
                **{key: mean(r[key] for r in records) for key in records[0]},
                'known_paired_interval': paired_interval([record['known_difference'] for record in records]),
            }
        reports[arm] = {
            'mean_total_ms': mean(timings), 'p95_total_ms': float(np.percentile(timings, 95)),
            'paired_ms_minus_full': paired_interval([
                1000 * (mean(t['total'] for t in a['timings']) - mean(t['total'] for t in b['timings']))
                for a, b in zip(data, baseline, strict=True)
            ]),
            'mean_top5_overlap': mean(r['top5_overlap'] for r in data), 'quality': quality,
        }
    return reports


def main(output):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Requires Slurm CUDA allocation')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    manifest = EMBEDDING / 'checkpoint-provenance.json'
    for name, expected in json.loads(manifest.read_text())['files'].items():
        if sha(EMBEDDING / name) != expected:
            raise ValueError(f'Changed embedding checkpoint: {name}')
    output.mkdir(parents=True, exist_ok=False)
    variants = QueryEncoderVariants()
    projections, training = fit(public_training_texts(), variants, output)
    report = {
        'job_id': os.environ['SLURM_JOB_ID'], 'gpu': torch.cuda.get_device_name(),
        'script_sha256': sha(Path(__file__)), 'checkpoint_manifest_sha256': sha(manifest),
        'arms': ARMS, 'training': training, 'datasets': {},
        'scope': (
            'KDD paper title+abstract as pseudoqueries; no reviewer labels used in fitting. '
            'Excludes exact normalized titles and IDs found in SIGIR or NIPS paper inventories. '
            'Calibration selects regularization against the full-depth teacher; both test indices '
            'retain frozen document vectors. Query loops include warm online encoding, projection, '
            'retrieval, SQLite evidence read and response construction. Excludes network, model '
            'startup, and semantic verification. NIPS sparse judgments use paired label bounds. '
            'Evaluation cohorts have been reused, and no fresh industrial or serving result is implied.'
        ),
    }
    for dataset in ('SIGIR', 'NIPS'):
        ranker, db, queries, labels, metadata = prepare(dataset, output, manifest)
        try:
            with sqlite3.connect(db) as connection:
                rows = benchmark(variants, projections, ranker, connection, queries, labels)
        finally:
            db.unlink()
        path = output / f'{dataset}-rows.json'
        path.write_text(json.dumps(rows, indent=2) + '\n')
        report['datasets'][dataset] = {**metadata, 'rows_sha256': sha(path), 'results': summarize(rows)}
        del ranker
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    main(parser.parse_args().output)
