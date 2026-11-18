"""Measure latency and ranking changes under smaller online query encoders."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sqlite3
from statistics import mean
import time

import numpy as np
import torch

from experiments.retrieval.cof_online_ownership import EMBEDDING, TOP_K, prepare, request, sha
from expert_search.qwen_embedding_torch import QwenTorchEmbedding, TASK
from expert_search.statistics import paired_interval

# Decoder layers, maximum tokens, and whether to retain the generation KV cache.
ARMS = {
    'reference': (28, 512, True),
    'no_cache': (28, 512, False),
    'cap_128': (28, 128, False),
    'cap_64': (28, 64, False),
    'layers_21': (21, 512, False),
    'layers_14': (14, 512, False),
}
REPEATS = 5
SEED = 20261118


class QueryEncoderVariants:
    def __init__(self):
        self.encoder = QwenTorchEmbedding(max_length=512, batch_size=1)
        model = self.encoder.model
        if len(model.layers) != 28 or not model.config.use_cache:
            raise ValueError('Unexpected reference decoder or cache setting')
        self.depths = {n: model.layers[:n] for n in (14, 21, 28)}

    def configure(self, arm):
        depth, token_cap, cache = ARMS[arm]
        self.encoder.model.layers = self.depths[depth]
        self.encoder.model.config.use_cache = cache
        self.encoder.max_length = token_cap


def compare_labels(reference, alternative, labels):
    removed = set(reference) - set(alternative)
    added = set(alternative) - set(reference)
    delta = sum(labels.get(owner) == 1 for owner in added) - sum(
        labels.get(owner) == 1 for owner in removed
    )
    return {
        'known_positive_at_5': sum(labels.get(owner) == 1 for owner in alternative) / TOP_K,
        'judged_at_5': sum(owner in labels for owner in alternative) / TOP_K,
        'difference_lower': (delta - sum(owner not in labels for owner in removed)) / TOP_K,
        'difference_upper': (delta + sum(owner not in labels for owner in added)) / TOP_K,
        'known_difference': delta / TOP_K,
    }


def benchmark(variants, ranker, connection, queries, labels):
    payloads = {qid: json.dumps({'query': text}, ensure_ascii=False) for qid, text in queries.items()}
    tokenizer = variants.encoder.tokenizer
    rows = {
        qid: {
            'raw_tokens': len(tokenizer.encode('Instruct: ' + TASK + '\nQuery:' + text)),
            'arms': {arm: {'timings': []} for arm in ARMS},
        } for qid, text in queries.items()
    }
    for arm in ARMS:
        variants.configure(arm)
        request(ranker, connection, variants.encoder, next(iter(payloads.values())), 'host_reduction')
    rng = random.Random(SEED)
    order = list(ARMS)
    for repeat in range(REPEATS):
        cohort = list(queries)
        rng.shuffle(cohort)
        for i, qid in enumerate(cohort):
            offset = (i + repeat) % len(order)
            for arm in order[offset:] + order[:offset]:
                variants.configure(arm)
                payload, ranking, timing = request(
                    ranker, connection, variants.encoder, payloads[qid], 'host_reduction'
                )
                observation = rows[qid]['arms'][arm]
                checksum = hashlib.sha256(payload).hexdigest()
                if observation.get('response_sha256', checksum) != checksum:
                    raise ValueError(f'Response changed across repeats: {qid}/{arm}')
                observation.update(ranking=ranking, response_sha256=checksum)
                observation['timings'].append(timing)
    for qid, row in rows.items():
        ref = row['arms']['reference']
        for observation in row['arms'].values():
            observation['ranking_overlap_at_5'] = len(set(ref['ranking']) & set(observation['ranking'])) / TOP_K
            observation['same_response'] = observation['response_sha256'] == ref['response_sha256']
            observation['quality'] = {
                task: compare_labels(ref['ranking'], observation['ranking'], by_query[qid])
                for task, by_query in labels.items()
            }
    return rows


def milliseconds(rows, key):
    return [1000 * t[key] for row in rows for t in row['timings']]


def summarize(rows):
    reports = {}
    for arm, (_, cap, _) in ARMS.items():
        data = [row['arms'][arm] for row in rows.values()]
        baseline = [row['arms']['reference'] for row in rows.values()]
        latencies = milliseconds(data, 'total')
        reports[arm] = {
            'mean_encode_ms': mean(milliseconds(data, 'encode')),
            'mean_total_ms': mean(latencies),
            'p95_total_ms': float(np.percentile(latencies, 95)),
            'paired_total_ms_minus_reference': paired_interval([
                mean(milliseconds([candidate], 'total')) - mean(milliseconds([reference], 'total'))
                for candidate, reference in zip(data, baseline, strict=True)
            ]),
            'same_response': sum(r['same_response'] for r in data),
            'mean_top5_overlap': mean(r['ranking_overlap_at_5'] for r in data),
            'clipped_queries': sum(row['raw_tokens'] > cap for row in rows.values()),
            'quality': {},
        }
        for task in data[0]['quality']:
            values = [r['quality'][task] for r in data]
            reports[arm]['quality'][task] = {
                **{key: mean(r[key] for r in values) for key in values[0]},
                'paired_known_difference': paired_interval([r['known_difference'] for r in values]),
            }
    return reports


def main(output):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Requires Slurm CUDA allocation')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    manifest = EMBEDDING / 'checkpoint-provenance.json'
    for file, expected in json.loads(manifest.read_text())['files'].items():
        if sha(EMBEDDING / file) != expected:
            raise ValueError(f'Changed embedding checkpoint: {file}')
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    variants = QueryEncoderVariants()
    torch.cuda.synchronize()
    report = {
        'job_id': os.environ['SLURM_JOB_ID'], 'gpu': torch.cuda.get_device_name(),
        'model_manifest_sha256': sha(manifest), 'script_sha256': sha(Path(__file__)),
        'harness_sha256': sha(Path('experiments/retrieval/cof_online_ownership.py')),
        'model_init_seconds': time.perf_counter() - started,
        'repeats_per_query': REPEATS, 'arms': ARMS, 'datasets': {},
        'scope': (
            'Frozen documents and fixed exact CPU ownership reduction after CUDA source scoring. '
            'Query decoder truncation retains the final normalization and does not align or train '
            'early layers. Caps include the fixed task prefix. Timing covers warm sequential JSON '
            'request processing, query encoding, retrieval, SQLite hydration and serialization. '
            'Excludes initialization, network, and semantic citation validation. Sparse NIPS '
            'judgments use attainable paired bounds from unjudged changes; known positives are '
            'not full P@5. Bootstrap intervals resample queries, not service workloads.'
        ),
    }
    for dataset in ('SIGIR', 'NIPS'):
        ranker, db, queries, labels, metadata = prepare(dataset, output, manifest)
        try:
            with sqlite3.connect(db) as connection:
                rows = benchmark(variants, ranker, connection, queries, labels)
        finally:
            db.unlink()
        raw = output / f'{dataset}-rows.json'
        raw.write_text(json.dumps(rows, indent=2) + '\n')
        report['datasets'][dataset] = {**metadata, 'rows_sha256': sha(raw), 'results': summarize(rows)}
        del ranker
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    main(parser.parse_args().output)
