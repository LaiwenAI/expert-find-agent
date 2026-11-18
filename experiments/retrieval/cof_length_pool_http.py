"""Measure age-bounded, per-length batching with a fixed-shape CoF encoder."""

import argparse
from collections import defaultdict, deque
from concurrent.futures import Future
from functools import partial
import json
import os
from pathlib import Path
from queue import Empty
import sqlite3
import time

import torch

from experiments.retrieval.cof_fixed_shape_http import CAP, FixedShape, matching
from experiments.retrieval.cof_heldout_encoder_comparison import DATASET, exclusions
from experiments.retrieval.cof_loopback_http import compare_method, deliver_batch, stats
from experiments.retrieval.cof_online_ownership import EMBEDDING, prepare, serialize_hits, sha
from experiments.retrieval.cof_query_encoding_budget import QueryEncoderVariants
from experiments.retrieval.cof_request_batching import load_plan

RATES = (120, 240)
WAITS_MS = {'bucket0': 0, 'bucket8': 8, 'bucket20': 20}
ARMS = ('serial', 'dynamic8', 'fixed_serial', 'fixed8', *WAITS_MS)


def choose_bucket(queues, now, patience, cap, closing):
    if not queues:
        return None, None
    oldest = min(queues, key=lambda width: queues[width][0][1])
    deadline = queues[oldest][0][1] + patience
    if closing or deadline <= now:
        return oldest, None
    ready = [width for width in queues if len(queues[width]) >= cap]
    if ready:
        return min(ready, key=lambda width: queues[width][0][1]), None
    return None, deadline - now


def bucket_worker(pending, ranker, database, encoder, cap, warm_query, ready, *, patience, rank_batch=None):
    queues = defaultdict(deque)
    closing = False

    def accept(item):
        nonlocal closing
        if item is None:
            closing = True
            return
        query, arrived, future = item
        try:
            ids, = encoder.tokenize([query], query=True)
            queues[encoder.width(ids)].append((ids, arrived, future))
        except Exception as error:
            future.set_exception(error)

    try:
        with sqlite3.connect(database) as connection:
            for size in (1, cap):
                begun = time.perf_counter()
                vectors = encoder.encode([warm_query] * size, query=True)
                if rank_batch is None:
                    for vector in vectors:
                        serialize_hits(ranker, connection, ranker.rank_host(ranker.score(vector)))
                else:
                    batch = [(warm_query, begun, Future()) for _ in range(size)]
                    deliver_batch(ranker, connection, batch, vectors, begun, time.perf_counter(),
                                  rank_batch=rank_batch)
            ready.set_result(True)
            while queues or not closing:
                while not closing:
                    try:
                        accept(pending.get_nowait())
                    except Empty:
                        break
                width, remaining = choose_bucket(queues, time.perf_counter(), patience, cap, closing)
                if width is None:
                    if closing:
                        break
                    try:
                        accept(pending.get(timeout=remaining))
                    except Empty:
                        pass
                    continue
                queue = queues[width]
                batch = [queue.popleft() for _ in range(min(len(queue), cap))]
                if not queue:
                    del queues[width]
                begun = time.perf_counter()
                try:
                    vectors = encoder.forward([ids for ids, _, _ in batch], width)
                    deliver_batch(ranker, connection, batch, vectors, begun, time.perf_counter(),
                                  rank_batch=rank_batch)
                except Exception as error:
                    for _, _, future in batch:
                        if not future.done():
                            future.set_exception(error)
    except Exception as error:
        if not ready.done():
            ready.set_exception(error)
        else:
            raise


def run(output):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Requires Slurm CUDA')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    checkpoint = EMBEDDING / 'checkpoint-provenance.json'
    for name, expected in json.loads(checkpoint.read_text())['files'].items():
        if sha(EMBEDDING / name) != expected:
            raise ValueError('Changed encoder')
    variants = QueryEncoderVariants()
    variants.configure('no_cache')
    standard = variants.encoder
    omitted_queries, omitted_sources, _ = exclusions()
    output.mkdir(parents=True, exist_ok=False)
    datasets = {}
    for i, dataset in enumerate(('SIGIR', DATASET)):
        options = {} if not i else {'exclude_queries': omitted_queries, 'exclude_sources': omitted_sources}
        ranker, db, queries, labels, index = prepare(dataset, output, checkpoint, **options)
        loads = {}
        try:
            for j, rate in enumerate(RATES):
                plan = load_plan(queries, rate)
                arms, geometry = {}, {}
                offset = (i + j) % len(ARMS)
                for method in ARMS[offset:] + ARMS[:offset]:
                    encoder = standard if method in ('serial', 'dynamic8') else FixedShape(standard)
                    mode = 'serial' if method.endswith('serial') else 'microbatch8'
                    kwargs = {}
                    if method in WAITS_MS:
                        kwargs = {'worker_impl': partial(bucket_worker, patience=WAITS_MS[method] / 1000)}
                    arms[method] = compare_method(ranker, db, encoder, mode, plan, queries, **kwargs)
                    if isinstance(encoder, FixedShape):
                        geometry[method] = {
                            'forward_by_width': dict(encoder.forward_shapes),
                            'physical_slots': CAP * sum(encoder.forward_shapes.values()),
                            'used_slots': sum(encoder.real_rows.values()),
                            'includes_warmup': True,
                        }
                metrics = {}
                for method, (rows, responses, runtime) in arms.items():
                    result = stats(rows, responses, arms['serial'][1], labels, runtime)
                    result['matching_fixed_serial'] = matching(responses, arms['fixed_serial'][1])
                    if method in geometry:
                        result['geometry'] = geometry[method]
                    path = output / f'{dataset}-{rate}-{method}.json'
                    path.write_text(json.dumps(rows, indent=2, allow_nan=False) + '\n')
                    result['rows_sha256'] = sha(path)
                    metrics[method] = result
                loads[rate] = metrics
                print(dataset, rate, 'complete', flush=True)
        finally:
            db.unlink()
        datasets[dataset] = {'index': index, 'loads': loads}
        del ranker
    files = ('cof_length_pool_http.py', 'cof_loopback_http.py', 'cof_fixed_shape_http.py')
    report = {'job_id': os.environ['SLURM_JOB_ID'], 'gpu': torch.cuda.get_device_name(),
              'encoder_sha256': sha(checkpoint),
              'code_sha256': {name: sha(Path(__file__).with_name(name)) for name in files},
              'waits_ms': WAITS_MS, 'datasets': datasets,
              'scope': 'Real loopback HTTP, fixed reproducible synthetic arrivals, one GPU worker. '
                       'Each padded shape has a local queue. A full queue can run early. When the '
                       'oldest queue has expired, it takes precedence; compute already in flight '
                       'can exceed the collection timer. No query embedding cache or semantic verifier.'}
    (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args().output)
