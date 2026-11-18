"""Calibrate a single-query recheck for unstable batched expert and citation rankings."""

import argparse
import json
import os
from pathlib import Path
import sqlite3

import numpy as np
import torch

from experiments.retrieval.cof_heldout_encoder_comparison import DATASET, exclusions
from experiments.retrieval.cof_online_ownership import EMBEDDING, TOP_K, prepare, sha
from experiments.retrieval.cof_query_encoding_budget import QueryEncoderVariants
from experiments.retrieval.cof_request_batching import load_plan, pipeline, report

CALIBRATION_RATE = 120
TEST_RATES = (120, 240)
THRESHOLDS = (.0001, .0003, .001, .003, .01, .03)
REQUIRED_CAPTURE = .9


def rank_with_margins(ranker, scores):
    scores = scores.cpu().numpy()
    index = ranker.index
    edges = scores[index.owner_rows]
    maxima = np.maximum.reduceat(edges, index.owner_starts)
    ordered = np.argsort(-maxima, kind='stable')
    results, witness_gaps = [], []
    for owner in ordered[:TOP_K]:
        sources = index.owners[index.owner_ids[owner]]
        source_scores = scores[sources]
        best = sources[int(np.argmax(source_scores))]
        results.append((int(owner), int(best), float(maxima[owner])))
        if len(sources) > 1:
            second = np.partition(source_scores, -2)[-2]
            witness_gaps.append(float(maxima[owner] - second))
    # If two selected experts share a witness, their scores move together as long
    # as both witnesses hold. The witness margins cover that condition.
    order_gaps = [high[2] - low[2] for high, low in zip(results, results[1:])
                  if high[1] != low[1]]
    boundary_source = results[-1][1]
    outside = ordered[TOP_K:]
    if outside.size:
        # Ignore shared boundary scores only. A different paper from any
        # excluded expert could still change membership or tie order.
        excluding_witness = np.maximum.reduceat(
            np.where(index.owner_rows == boundary_source, -np.inf, edges),
            index.owner_starts,
        )
        competitor = float(excluding_witness[outside].max())
        if np.isfinite(competitor):
            order_gaps.append(results[-1][2] - competitor)
    return results, {
        'expert_margin': min(order_gaps, default=None),
        'witness_margin': min(witness_gaps, default=None),
    }


def smallest_margin(row):
    margins = (row.get('expert_margin'), row.get('witness_margin'))
    return min((value for value in margins if value is not None), default=float('inf'))


def review(encoder, ranker, threshold=None):
    def rank(scores, query):
        results, details = rank_with_margins(ranker, scores)
        should_recheck = threshold is not None and smallest_margin(details) <= threshold
        details['rechecked'] = should_recheck
        if should_recheck:
            single_query = encoder.encode([query], query=True)[0]
            results = ranker.rank_host(ranker.score(single_query))
        return results, details
    return rank


def calibrate(rows):
    unstable = sum(not r['same_experts_and_citations'] for r in rows)
    if unstable == 0:
        raise RuntimeError('No ranking drift in calibration trace')
    tradeoff = []
    for threshold in THRESHOLDS:
        selected = [r for r in rows if smallest_margin(r) <= threshold]
        caught = sum(not r['same_experts_and_citations'] for r in selected)
        tradeoff.append({'threshold': threshold, 'selected': len(selected), 'caught': caught})
    chosen = next((row for row in tradeoff if row['caught'] / unstable >= REQUIRED_CAPTURE), tradeoff[-1])
    return {'requests': len(rows), 'unstable': unstable, 'grid': tradeoff, 'chosen': chosen,
            'reached_target': chosen['caught'] / unstable >= REQUIRED_CAPTURE}


def run_arm(encoder, ranker, conn, plan, queries, method, threshold=None):
    if method in ('serial', 'batch8_backlog'):
        return pipeline(encoder, ranker, conn, plan, queries, method)
    return pipeline(encoder, ranker, conn, plan, queries, 'batch8_backlog',
                    review(encoder, ranker, threshold))


def save_arm(output, name, records, responses, original, labels, runtime):
    summary = report(records, responses, original, labels, runtime)
    filename = output / f'{name}.json'
    filename.write_text(json.dumps(records, indent=2, allow_nan=False) + '\n')
    summary['rows_sha256'] = sha(filename)
    summary['fallbacks'] = sum(r.get('rechecked', False) for r in records)
    return summary


def main(output):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Requires Slurm and CUDA')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    checkpoint = EMBEDDING / 'checkpoint-provenance.json'
    for name, expected in json.loads(checkpoint.read_text())['files'].items():
        if sha(EMBEDDING / name) != expected:
            raise ValueError(f'Changed model file: {name}')
    variant = QueryEncoderVariants()
    variant.configure('no_cache')
    encoder = variant.encoder
    output.mkdir(parents=True, exist_ok=False)
    result = {'job_id': os.environ['SLURM_JOB_ID'], 'gpu': torch.cuda.get_device_name(),
              'script_sha256': sha(Path(__file__)), 'checkpoint_sha256': sha(checkpoint)}
    omitted_queries, omitted_sources, _ = exclusions()
    for dataset in ('SIGIR', DATASET):
        options = {'exclude_queries': omitted_queries, 'exclude_sources': omitted_sources} if dataset == DATASET else {}
        ranker, db, queries, labels, metadata = prepare(dataset, output, checkpoint, **options)
        try:
            with sqlite3.connect(db) as conn:
                encoder.batch_size = 8
                for vec in encoder.encode(list(queries.values())[:8], query=True):
                    scores = ranker.score(vec)
                    assert rank_with_margins(ranker, scores)[0] == ranker.rank_host(scores)
                if dataset == 'SIGIR':
                    plan = load_plan(queries, CALIBRATION_RATE)
                    serial = run_arm(encoder, ranker, conn, plan, queries, 'serial')
                    audited = run_arm(encoder, ranker, conn, plan, queries, 'margin_audit')
                    rows, responses, runtime = audited
                    metrics = save_arm(output, 'SIGIR-calibration', rows, responses, serial[1], labels, runtime)
                    cal = calibrate(rows)
                    threshold = cal['chosen']['threshold']
                    result['calibration'] = {'dataset': dataset, 'target_rate': CALIBRATION_RATE,
                                             'threshold_selection': cal, 'metrics': metrics, 'index': metadata}
                    print('Calibration', cal, flush=True)
                else:
                    result['evaluation'] = {'dataset': dataset, 'index': metadata, 'loads': {}}
                    methods = ['serial', 'batch8_backlog', 'margin_recheck']
                    for i, rate in enumerate(TEST_RATES):
                        plan = load_plan(queries, rate)
                        arms = {method: run_arm(encoder, ranker, conn, plan, queries, method, threshold)
                                for method in methods[i:] + methods[:i]}
                        result['evaluation']['loads'][rate] = {
                            name: save_arm(output, f'{dataset}-{rate}-{name}', rows, response,
                                           arms['serial'][1], labels, runtime)
                            for name, (rows, response, runtime) in arms.items()
                        }
                        print(dataset, rate, 'complete', flush=True)
        finally:
            db.unlink()
        del ranker
    (output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    main(parser.parse_args().output)
