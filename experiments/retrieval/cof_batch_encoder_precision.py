"""Isolate padding position and inference precision effects in microbatched search."""

import argparse
import json
import os
from pathlib import Path
import sqlite3

import numpy as np
import torch

from experiments.retrieval.cof_heldout_encoder_comparison import DATASET, exclusions
from experiments.retrieval.cof_online_ownership import EMBEDDING, prepare, serialize_hits, sha
from experiments.retrieval.cof_query_encoding_budget import QueryEncoderVariants
from experiments.retrieval.cof_request_batching import load_plan, pipeline, report, shape

RATE = 240
MATCHED_BATCH_SIZE = 8
CONFIGS = (
    ('bf16_default', torch.bfloat16, False),
    ('bf16_query_positions', torch.bfloat16, True),
    ('fp32_query_positions', torch.float32, True),
)


def query_positions(_model, args, kwargs):
    mask = kwargs['attention_mask']
    kwargs['position_ids'] = (mask.long().cumsum(-1) - 1).clamp_min_(0)
    return args, kwargs


def responses(encoder, ranker, connection, queries, batch_size):
    encoder.batch_size = batch_size
    all_vectors, all_responses = [], []
    for begin in range(0, len(queries), batch_size):
        vectors = encoder.encode(queries[begin:begin + batch_size], query=True)
        all_vectors.append(vectors)
        for vector in vectors:
            rank = ranker.rank_host(ranker.score(vector))
            all_responses.append(serialize_hits(ranker, connection, rank))
    return np.concatenate(all_vectors), all_responses


def parity(bodies, reference, indices=None):
    if indices is None:
        indices = range(len(bodies))
    comparisons = [(bodies[i], reference[i]) for i in indices]
    return {
        'requests': len(comparisons),
        'same_bytes': sum(a[0] == b[0] for a, b in comparisons),
        'same_ordered_experts': sum(a[1] == b[1] for a, b in comparisons),
        'same_ordered_experts_and_citations': sum(
            a[0] == b[0] or shape(a[0]) == shape(b[0]) for a, b in comparisons
        ),
    }


def set_parameter_precision(model, dtype):
    buffers = [(module, name, value) for module in model.modules()
               for name, value in module.named_buffers(recurse=False)]
    model.to(dtype=dtype)
    # Preserve the original precision of rotary frequencies and other buffers.
    for module, name, value in buffers:
        setattr(module, name, value)


def each_configuration(encoder, ranker, connection, plan, queries, order):
    records = {}
    _, ids = plan
    strings = [queries[qid] for qid in ids]
    for name, dtype, normalized_positions in order:
        set_parameter_precision(encoder.model, dtype)
        handle = encoder.model.register_forward_pre_hook(
            query_positions, with_kwargs=True
        ) if normalized_positions else None
        try:
            encoder.batch_size = MATCHED_BATCH_SIZE
            encoder.encode(strings[:MATCHED_BATCH_SIZE], query=True)
            encoder.encode(strings[:1], query=True)
            singles = responses(encoder, ranker, connection, strings, 1)
            fixed = responses(encoder, ranker, connection, strings, MATCHED_BATCH_SIZE)
            records[name] = (singles, fixed, pipeline(
                encoder, ranker, connection, plan, queries, 'batch8_backlog'
            ))
        finally:
            if handle is not None:
                handle.remove()
    return records


def summarize(records, output, dataset, labels):
    (reference_vectors, reference), _, (_, _, _) = records['bf16_default']
    qrels = {}
    for method, ((serial_vectors, serial), (fixed_vectors, fixed), (rows, live, runtime)) in records.items():
        distances = np.linalg.norm(serial_vectors - fixed_vectors, axis=1)
        against_original = np.linalg.norm(reference_vectors - fixed_vectors, axis=1)
        live_metrics = report(rows, live, reference, labels, runtime)
        path = output / f'{dataset}-{method}.json'
        path.write_text(json.dumps(rows, indent=2, allow_nan=False) + '\n')
        live_metrics['rows_sha256'] = sha(path)
        qrels[method] = {
            'fixed8_against_own_serial': parity(fixed, serial),
            'fixed8_against_bf16_serial': parity(fixed, reference),
            'own_serial_against_bf16_serial': parity(serial, reference),
            'fixed8_embedding_l2': {
                'own_serial_p95': float(np.quantile(distances, .95)),
                'own_serial_max': float(distances.max()),
                'bf16_serial_p95': float(np.quantile(against_original, .95)),
            },
            'open_loop_against_bf16_serial': live_metrics,
            'open_loop_against_own_serial': parity(live, serial),
        }
    return qrels


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
        'target_rate': RATE, 'fixed_batch_size': MATCHED_BATCH_SIZE, 'datasets': {},
        'scope': 'FP32 uses upcast BF16 checkpoint values and the same unchanged index.',
    }
    omitted_queries, omitted_sources, _ = exclusions()
    for i, dataset in enumerate(('SIGIR', DATASET)):
        options = {} if i == 0 else {'exclude_queries': omitted_queries, 'exclude_sources': omitted_sources}
        ranker, db, queries, labels, metadata = prepare(dataset, output, checkpoint, **options)
        plan = load_plan(queries, RATE)
        try:
            with sqlite3.connect(db) as connection:
                order = CONFIGS if i == 0 else CONFIGS[::-1]
                measured = each_configuration(encoder, ranker, connection, plan, queries, order)
                result['datasets'][dataset] = {
                    'index': metadata,
                    'arms': summarize(measured, output, dataset, labels),
                }
                print(dataset, 'complete', flush=True)
        finally:
            db.unlink()
        del ranker
    (output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    main(parser.parse_args().output)
