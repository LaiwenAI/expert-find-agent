"""Compare exact batched ownership reduction under reproducible HTTP arrivals."""

import argparse
from functools import partial
import json
import os
from pathlib import Path
from statistics import median

import numpy as np
import torch

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from experiments.retrieval.cof_fixed_shape_http import CAP, FixedShape
from experiments.retrieval.cof_length_pool_http import bucket_worker
from experiments.retrieval.cof_loopback_http import CONNECTIONS, compare_method, stats
from experiments.retrieval.cof_online_ownership import EMBEDDING, sha
from experiments.retrieval.cof_segmented_gpu_ownership import SegmentedOwnerRanking, prepare_cohort

RATES = (100, 240)
MODES = ('host', 'device', 'host_dispatch', 'device_dispatch')
REPEATS = 3
REQUESTS = 384
SEED = 20261120
PATIENCE = .008


def plan(queries, rate, repeat):
    rng = np.random.default_rng([SEED, rate, repeat])
    gaps = rng.exponential(1 / rate, REQUESTS - 1)
    query_ids = rng.choice(sorted(queries), size=REQUESTS).tolist()
    return np.concatenate(([0], np.cumsum(gaps))).tolist(), query_ids


def assemble_rankers(ranker, device):
    def device_one(vectors):
        return (device.rank(ranker.score(v)) for v in vectors)

    def score_batch(vectors):
        # Preserve the exact torch.mv scoring kernel for each query.
        return torch.stack([ranker.score(v) for v in vectors])

    def host_dispatch(vectors):
        return (ranker.rank_host(scores) for scores in score_batch(vectors).cpu())

    def device_dispatch(vectors):
        return device.rank_batch(score_batch(vectors))

    return {'host': None, 'device': device_one, 'host_dispatch': host_dispatch,
            'device_dispatch': device_dispatch}


def summarize(reports):
    fields = {
        'request_p95_ms': lambda r: r['latency_ms']['total_ms']['p95'],
        'queue_p95_ms': lambda r: r['latency_ms']['queue_ms']['p95'],
        'service_p95_ms': lambda r: r['latency_ms']['service_ms']['p95'],
        'post_encoding_p95_ms': lambda r: r['post_encoding_p95_ms'],
        'drained_per_second': lambda r: r['drained_requests_per_second'],
        'mean_batch_size_per_request': lambda r: r['mean_batch_size_per_request'],
    }
    summary = {}
    for field, select in fields.items():
        summary[field] = {}
        for arm in MODES:
            values = [select(r[arm]) for r in reports]
            summary[field][arm] = {'traces': values, 'median': median(values)}
    return summary


def run(output):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Requires Slurm CUDA')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    checkpoint = EMBEDDING / 'checkpoint-provenance.json'
    for name, digest in json.loads(checkpoint.read_text())['files'].items():
        if sha(EMBEDDING / name) != digest:
            raise ValueError('Changed embedding checkpoint')

    output.mkdir(parents=True, exist_ok=False)
    ranker, db, queries, labels, inventory, _ = prepare_cohort('SciRepEval', output, checkpoint)
    original = QwenTorchEmbedding(batch_size=CAP)
    original.model.config.use_cache = False
    device = SegmentedOwnerRanking(ranker, batch_capacity=CAP)
    callbacks = assemble_rankers(ranker, device)
    # Exercise batch-axis offsets with varying ties and distinct maxima on every row.
    scores = torch.arange(CAP * device.source_count, device='cuda').reshape(CAP, -1).remainder(13).float() - 5
    assert device.rank_batch(scores) == [ranker.rank_host(row) for row in scores]
    loads = {}
    try:
        fixture = FixedShape(original)
        known = list(queries)
        for start in range(0, len(known), CAP):
            vectors = fixture.encode([queries[q] for q in known[start:start + CAP]], query=True)
            reference = [ranker.rank_host(ranker.score(vec)) for vec in vectors]
            if any(list(callback(vectors)) != reference for callback in callbacks.values() if callback):
                raise ValueError('Exact reduction disagrees on identical query vectors')
        print('Exact aggregation controls passed', len(known), flush=True)
        for rate in RATES:
            traces, observed = [], {arm: {} for arm in MODES}
            for repeat in range(REPEATS):
                arrivals = plan(queries, rate, repeat)
                offset = (repeat + RATES.index(rate)) % len(MODES)
                arms = {}
                for arm in MODES[offset:] + MODES[:offset]:
                    encoder = FixedShape(original)
                    worker = partial(bucket_worker, patience=PATIENCE, rank_batch=callbacks[arm])
                    arms[arm] = compare_method(ranker, db, encoder, 'microbatch8', arrivals, queries,
                                               worker_impl=worker)
                metrics = {}
                for arm, (rows, responses, runtime) in arms.items():
                    result = stats(rows, responses, arms['host'][1], labels, runtime)
                    differences = []
                    for row, (answer, _), (base, _) in zip(rows, responses, arms['host'][1], strict=True):
                        observed[arm].setdefault(row['query'], set()).add(row['response_sha256'])
                        if not row['same_bytes'] and row['same_experts_and_citations']:
                            differences.extend(abs(a['score'] - b['score']) for a, b in
                                               zip(json.loads(answer), json.loads(base), strict=True))
                    result['max_score_difference_when_order_and_citations_equal'] = max(differences, default=0.0)
                    result['post_encoding_p95_ms'] = float(np.quantile(
                        [r['service_ms'] - r['batch_encode_ms'] for r in rows], .95))
                    p = output / f'{rate}-{repeat}-{arm}.json'
                    p.write_text(json.dumps(rows, indent=2, allow_nan=False) + '\n')
                    result['rows_sha256'] = sha(p)
                    metrics[arm] = result
                traces.append(metrics)
                print(rate, repeat, {m: round(metrics[m]['latency_ms']['total_ms']['p95'], 3) for m in MODES},
                      flush=True)
            loads[rate] = {
                'traces': traces, 'summary': summarize(traces),
                'unique_queries_with_more_than_one_response': {
                    arm: sum(len(values) > 1 for values in variants.values())
                    for arm, variants in observed.items()
                },
                'observed_unique_queries': len(observed['host']),
            }
    finally:
        db.unlink()
    scripts = ('cof_segmented_ownership_http.py', 'cof_segmented_gpu_ownership.py', 'cof_loopback_http.py',
               'cof_fixed_shape_http.py', 'cof_length_pool_http.py', 'cof_online_ownership.py')
    report = {
        'job_id': os.environ['SLURM_JOB_ID'], 'gpu': torch.cuda.get_device_name(),
        'checkpoint_sha256': sha(checkpoint), 'code_sha256': {s: sha(Path(__file__).with_name(s)) for s in scripts},
        'index': inventory, 'connections': CONNECTIONS, 'patience_ms': PATIENCE * 1000,
        'modes': MODES, 'loads': loads, 'queries_with_identical_exact_reduction_at_fixed_vectors': len(known),
        'scope': 'Warm loopback HTTP with one GPU worker and no query cache. Fixed eight-slot encoding, '
                 'per-bucket age limit, same exact per-query dot products, and SQLite source delivery. '
                 'Dispatch arms wait for one coalesced result before sending the first response. '
                 'Reported values describe repeated traces from a limited query pool. No WAN or semantic verifier.',
    }
    (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
