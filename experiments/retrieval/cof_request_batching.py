"""Measure throughput, queue latency and ranking parity under fixed arrivals."""
import argparse
from bisect import bisect_right
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from statistics import mean
import time

import numpy as np
import torch

from experiments.retrieval.cof_heldout_encoder_comparison import DATASET, exclusions
from experiments.retrieval.cof_online_ownership import EMBEDDING, prepare, serialize_hits, sha
from experiments.retrieval.cof_query_encoding_budget import QueryEncoderVariants, compare_labels
from experiments.retrieval.cof_resumable_query import payload

MODES = {
    'serial': (1, 0.0),
    'batch4_backlog': (4, 0.0),
    'batch8_backlog': (8, 0.0),
    'batch4_wait1ms': (4, .001),
}
RATES = (50, 120, 240)
REQUESTS = 384
SEED = 20261118


def sleep_until(deadline):
    remaining = deadline - time.perf_counter()
    while remaining > 0:
        time.sleep(remaining)
        remaining = deadline - time.perf_counter()


def next_batch(arrivals, cursor, maximum, window, now):
    last = min(cursor + maximum - 1, len(arrivals) - 1)
    deadline = max(arrivals[cursor], min(arrivals[last], arrivals[cursor] + window))
    return max(now, deadline)


def load_plan(query_ids, rate):
    rng = np.random.default_rng(SEED + rate)
    gaps = rng.exponential(1 / rate, REQUESTS - 1)
    times = np.concatenate(([0], np.cumsum(gaps))).tolist()
    qids = rng.choice(list(query_ids), size=REQUESTS).tolist()
    return times, qids


def pipeline(encoder, ranker, connection, plan, queries, mode, review=None):
    arrivals, qids = plan
    maximum, wait = MODES[mode]
    encoder.batch_size = maximum
    serialized = {q: payload(text) for q, text in queries.items()}
    records = []
    responses = []
    cursor = 0
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    while cursor < len(qids):
        clock = time.perf_counter()
        deadline = next_batch(arrivals, cursor, maximum, wait, clock - start)
        sleep_until(start + deadline)
        begun = time.perf_counter()
        stop = min(bisect_right(arrivals, begun - start, lo=cursor), cursor + maximum)
        if stop <= cursor:
            continue
        batch = qids[cursor:stop]
        query_strings = [json.loads(serialized[q])['query'] for q in batch]
        vectors = encoder.encode(query_strings, query=True)
        encoded = time.perf_counter()
        for i, vec in enumerate(vectors):
            scores = ranker.score(vec)
            details = {}
            if review is not None and len(batch) > 1:
                rank, details = review(scores, query_strings[i])
            else:
                rank = ranker.rank_host(scores)
            body, experts = serialize_hits(ranker, connection, rank)
            finished = time.perf_counter()
            arrived = start + arrivals[cursor + i]
            responses.append((body, experts))
            records.append({
                'query': batch[i], 'batch': len(batch), 'queue_ms': 1000 * (begun - arrived),
                'service_ms': 1000 * (finished - begun), 'total_ms': 1000 * (finished - arrived),
                'batch_encode_ms': 1000 * (encoded - begun),
                **details,
            })
        cursor = stop
    end = time.perf_counter()
    return records, responses, {
        'elapsed_seconds': end - start,
        'drained_requests_per_second': len(records) / (end - start),
        'issued_requests_per_second': (len(records) - 1) / arrivals[-1],
        'torch_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
    }


def shape(response):
    return [(hit['expert'], hit['source'], hit['title'], hit['excerpt']) for hit in json.loads(response)]


def report(records, responses, reference, labels, runtime):
    for row, (body, experts), (base, original) in zip(records, responses, reference, strict=True):
        row['response_sha256'] = hashlib.sha256(body).hexdigest()
        row['same_bytes'] = body == base
        row['same_experts'] = experts == original
        row['same_experts_and_citations'] = body == base or shape(body) == shape(base)
        row['quality'] = {g: compare_labels(original, experts, qrels[row['query']]) for g, qrels in labels.items()}
    result = {
        **runtime, 'requests': len(records), 'mean_batch_size_per_request': mean(row['batch'] for row in records),
        'same_bytes': sum(r['same_bytes'] for r in records), 'same_ordered_experts': sum(r['same_experts'] for r in records),
        'same_ordered_experts_and_citations': sum(r['same_experts_and_citations'] for r in records),
        'latency_ms': {}, 'quality': {},
    }
    for component in ('queue_ms', 'service_ms', 'total_ms'):
        data = [row[component] for row in records]
        p50, p95, p99 = np.quantile(data, [.5, .95, .99])
        result['latency_ms'][component] = {
            'mean': mean(data), 'p50': float(p50), 'p95': float(p95), 'p99': float(p99),
        }
    for grade in labels:
        metrics = [row['quality'][grade] for row in records]
        result['quality'][grade] = {m: mean(d[m] for d in metrics) for m in metrics[0]}
    return result


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
    result = {
        'job_id': os.environ['SLURM_JOB_ID'], 'gpu': torch.cuda.get_device_name(),
        'script_sha256': sha(Path(__file__)), 'checkpoint_sha256': sha(checkpoint),
        'requests_per_trace': REQUESTS, 'target_rates': RATES, 'batch_policies': MODES,
        'datasets': {},
        'scope': (
            'One worker, one GPU with deterministic open-loop simulated arrival times. '
            'Real timers include queueing, JSON parsing, batch encoder inference, exact ranking, '
            'SQLite lookup and serialization. Queries are encoded only when their batch dispatches. '
            'The same generated request IDs and arrivals are used in every arm; raw text is '
            'not exported. No network, concurrent worker replication, or semantic evidence verification.'
        ),
    }
    omitted_queries, omitted_sources, _ = exclusions()
    for i, dataset in enumerate(('SIGIR', DATASET)):
        kwargs = {} if i == 0 else {'exclude_queries': omitted_queries, 'exclude_sources': omitted_sources}
        ranker, db, queries, labels, metadata = prepare(dataset, output, checkpoint, **kwargs)
        outputs = {}
        try:
            with sqlite3.connect(db) as connection:
                warm_query = list(queries.values())[:8]
                encoder.batch_size = 8
                for q in encoder.encode(warm_query, query=True):
                    serialize_hits(ranker, connection, ranker.rank_host(ranker.score(q)))
                for j, rate in enumerate(RATES):
                    plan = load_plan(queries, rate)
                    modes = list(MODES)
                    offset = (i + j) % len(modes)
                    measurements = {
                        mode: pipeline(encoder, ranker, connection, plan, queries, mode)
                        for mode in modes[offset:] + modes[:offset]
                    }
                    reference = measurements['serial'][1]
                    summary = {}
                    for method, (records, responses, runtime) in measurements.items():
                        summary[method] = report(records, responses, reference, labels, runtime)
                        data = output / f'{dataset}-{rate}-{method}.json'
                        data.write_text(json.dumps(records, indent=2) + '\n')
                        summary[method]['rows_sha256'] = sha(data)
                    outputs[rate] = summary
        finally:
            db.unlink()
        result['datasets'][dataset] = {'index': metadata, 'loads': outputs}
        del ranker
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    main(parser.parse_args().output)
