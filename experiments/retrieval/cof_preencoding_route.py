"""Compare cheap query-only routing with post-encoding continuation."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import sqlite3
from statistics import mean
import time

import numpy as np
import torch

from experiments.retrieval.cof_heldout_encoder_comparison import DATASET, exclusions
from experiments.retrieval.cof_online_ownership import EMBEDDING, prepare, sha
from experiments.retrieval.cof_query_backbone_finetune import ROOT
from experiments.retrieval.cof_query_encoding_budget import compare_labels
from experiments.retrieval.cof_resumable_query import ResumableQuery, payload, request
from experiments.retrieval.cof_query_encoding_budget import QueryEncoderVariants
from expert_search.statistics import paired_interval

METHODS = ('full', 'shallow', 'resume', 'preencoding', 'resume_matched')
GRADES = ('soft', 'hard')
REPEATS = 3
FOLDS = 5
SEED = 20261118
TOLERANCE = .02
WORDS = re.compile(r'\w+', re.UNICODE)


def features(query):
    title, _, body = query.partition('\n')
    words = WORDS.findall(query)
    size = max(1, len(words))
    return np.array([
        math.log1p(len(words)),
        len({w.casefold() for w in words}) / size,
        len(WORDS.findall(title)) / size,
        sum(any(c.isdigit() for c in w) for w in words) / size,
        sum(len(w) > 1 and w.isupper() for w in words) / size,
        sum(len(w) for w in words) / size,
        sum(body.count(c) for c in '.;:?') / size,
    ], dtype=np.float64)


def fit(x, y):
    center = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-8)
    normalized = (x - center) / scale
    bias = y.mean(axis=0)
    weights = np.linalg.solve(normalized.T @ normalized + 5 * np.eye(x.shape[1]), normalized.T @ (y - bias))
    return center, scale, bias, weights


def score(model, x):
    center, scale, bias, weights = model
    return np.min((x - center) / scale @ weights + bias, axis=-1)


def calibrate(ranker, db, queries, labels, encoder):
    observations = []
    with sqlite3.connect(db) as connection:
        for qid, query in queries.items():
            original = request(payload(query), encoder, ranker, connection, 'full')
            early = request(payload(query), encoder, ranker, connection, 'shallow')
            changes = [compare_labels(original[1], early[1], labels[g][qid])['known_difference'] for g in GRADES]
            observations.append((features(query), early[2], changes))
    matrix = np.stack([row[0] for row in observations])
    deltas = np.asarray([row[2] for row in observations])
    groups = np.array_split(np.random.default_rng(SEED).permutation(len(matrix)), FOLDS)
    oof = np.zeros(len(matrix))
    for heldout in groups:
        training = np.setdiff1d(np.arange(len(matrix)), heldout)
        oof[heldout] = score(fit(matrix[training], deltas[training]), matrix[heldout])
    cutoff = float('inf')
    accepted = np.zeros(len(matrix), dtype=bool)
    for candidate in sorted(set(oof), reverse=True):
        above = oof >= candidate
        if above.mean() <= .5 and min(deltas[above].sum(axis=0) / len(matrix)) >= -TOLERANCE:
            cutoff, accepted = float(candidate), above
    return fit(matrix, deltas), cutoff, {
        'queries': len(matrix),
        'max_calibration_early_fraction': .5,
        'oof_early': int(accepted.sum()),
        'oof_quality_delta': (deltas[accepted].sum(axis=0) / len(matrix)).tolist(),
        'cutoff': cutoff if np.isfinite(cutoff) else None,
        'resume_margin': float(np.median([row[1] for row in observations])),
    }


def run(queries, labels, ranker, connection, encoder, router, cutoff, margin, matched_margin, pool):
    def execute(query, body, method, eligible):
        began = time.perf_counter()
        if method == 'preencoding':
            risk = score(router, features(query))
            path = 'shallow' if risk >= cutoff else 'full'
        else:
            path = method
        limit = matched_margin if method == 'resume_matched' else margin
        digest, experts, _, exited, _ = request(body, encoder, ranker, connection, path, limit, eligible)
        return digest, experts, exited, time.perf_counter() - began

    bodies = {q: payload(text) for q, text in queries.items()}
    results = {q: {arm: {'seconds': []} for arm in METHODS} for q in queries}
    first = next(iter(queries))
    for arm in METHODS:
        execute(queries[first], bodies[first], arm, None if pool is None else pool[first])
    for repeat in range(REPEATS):
        for i, (qid, text) in enumerate(queries.items()):
            offset = (i + repeat) % len(METHODS)
            for arm in METHODS[offset:] + METHODS[:offset]:
                digest, experts, exited, elapsed = execute(
                    text, bodies[qid], arm, None if pool is None else pool[qid]
                )
                record = results[qid][arm]
                if record.get('digest', digest) != digest:
                    raise ValueError(f'Unstable response: {qid}/{arm}')
                record.update(digest=digest, experts=experts, exited=exited)
                record['seconds'].append(elapsed)
            for arm in ('resume', 'preencoding', 'resume_matched'):
                item = results[qid][arm]
                expected = 'shallow' if item['exited'] else 'full'
                if item['digest'] != results[qid][expected]['digest']:
                    raise ValueError(f'Incorrect routed response: {qid}/{arm}')

    summary = {}
    for arm in METHODS:
        records = [r[arm] for r in results.values()]
        durations = [1000 * t for r in records for t in r['seconds']]
        quality = {}
        for grade, judgments in labels.items():
            differences = [
                compare_labels(row['full']['experts'], row[arm]['experts'], judgments[q])
                for q, row in results.items()
            ]
            quality[grade] = {
                **{key: mean(r[key] for r in differences) for key in differences[0]},
                'paired': paired_interval([d['known_difference'] for d in differences]),
            }
        summary[arm] = {
            'early_exits': sum(r['exited'] for r in records),
            'mean_ms': mean(durations), 'p95_ms': float(np.quantile(durations, .95)),
            'quality': quality,
        }
    rate = summary['preencoding']['early_exits'] / len(results)
    summary['random_same_acceptance_expected_known_delta'] = {
        g: rate * summary['shallow']['quality'][g]['known_difference'] for g in GRADES
    }
    return results, summary


def main(output):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Requires Slurm and CUDA')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    manifest = EMBEDDING / 'checkpoint-provenance.json'
    for name, expected in json.loads(manifest.read_text())['files'].items():
        if sha(EMBEDDING / name) != expected:
            raise ValueError(f'Changed embedding file: {name}')
    baseline = json.loads((ROOT / 'summary.json').read_text())
    matrix = ROOT / 'projection.npz'
    if sha(matrix) != baseline['projector_sha256'] or sha(manifest) != baseline['checkpoint_manifest_sha256']:
        raise ValueError('Changed model artifact')
    with np.load(matrix, allow_pickle=False) as saved:
        encoder = ResumableQuery(QueryEncoderVariants(), (saved['weights'], saved['bias']))
    output.mkdir(parents=True, exist_ok=False)
    ranker, db, queries, labels, _ = prepare('SIGIR', output, manifest)
    try:
        router, cutoff, calibration = calibrate(ranker, db, queries, labels, encoder)
    finally:
        db.unlink()
    del ranker
    dump = output / 'router.npz'
    np.savez(dump, center=router[0], scale=router[1], bias=router[2], weights=router[3], cutoff=cutoff)
    excluded_q, excluded_docs, exclusion = exclusions()
    ranker, db, queries, labels, metadata = prepare(
        DATASET, output, manifest, exclude_queries=excluded_q, exclude_sources=excluded_docs
    )
    judged = {q: labels['soft'][q].keys() & ranker.index.owners.keys() for q in queries}
    if any(not pool or labels['soft'][q].keys() != labels['hard'][q].keys() for q, pool in judged.items()):
        raise ValueError('Invalid judged pool')
    results = {}
    try:
        with sqlite3.connect(db) as connection:
            count = sum(int(score(router, features(q)) >= cutoff) for q in queries.values())
            margins = sorted(
                (request(payload(q), encoder, ranker, connection, 'shallow')[2] for q in queries.values()),
                reverse=True,
            )
            matched_margin = margins[count - 1] if count else float('inf')
            if sum(m >= matched_margin for m in margins) != count:
                raise ValueError('Tied margins at the label-blind acceptance-matching cutoff')
            for protocol, eligibility in (('global', None), ('judged_pool', judged)):
                rows, summary = run(queries, labels, ranker, connection, encoder, router, cutoff,
                                    calibration['resume_margin'], matched_margin, eligibility)
                path = output / f'{protocol}-rows.json'
                path.write_text(json.dumps(rows, indent=2) + '\n')
                results[protocol] = {'summary': summary, 'rows_sha256': sha(path)}
    finally:
        db.unlink()
    report = {
        'job_id': os.environ['SLURM_JOB_ID'], 'gpu': torch.cuda.get_device_name(),
        'script_sha256': sha(Path(__file__)), 'router_sha256': sha(dump),
        'calibration': calibration, 'matched_acceptance_diagnostic': count,
        'exclusions': exclusion, 'index_metadata': metadata,
        'results': results,
        'scope': (
            'Ridge predicts changes in soft/hard P@5 from seven query-only features. '
            'Five-fold out-of-fold SIGIR predictions set the cutoff using a point loss '
            'tolerance, followed by fitting on all SIGIR. Previously explored filtered SciRepEval '
            'is disjoint from router fitting. The median continuation gate uses SIGIR margins. '
            'A separate diagnostic matches exit coverage with unlabeled target margins. '
            'Router overhead, any second search and SQLite citation delivery are timed. '
            'No network, concurrency or evidence semantics. Global labels are sparse.'
        ),
    }
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
