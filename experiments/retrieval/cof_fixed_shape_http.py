"""Measure fixed-shape embedding stability under real loopback microbatching."""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path

import numpy as np
import torch

from expert_search.qwen_embedding import TASK
from experiments.retrieval.cof_heldout_encoder_comparison import DATASET, exclusions
from experiments.retrieval.cof_loopback_http import compare_method, stats
from experiments.retrieval.cof_online_ownership import EMBEDDING, prepare, sha
from experiments.retrieval.cof_query_encoding_budget import QueryEncoderVariants
from experiments.retrieval.cof_request_batching import load_plan, shape

CAP = 8
BUCKET = 64
RATES = (120, 240)
ARMS = ('serial', 'dynamic8', 'fixed_serial', 'fixed8')


class FixedShape:
    def __init__(self, original):
        self.tokenizer = original.tokenizer
        self.model = original.model
        self.max_length = original.max_length
        self.batch_size = CAP
        self.forward_shapes = Counter()
        self.real_rows = Counter()

    def encode(self, texts, query=False):
        if len(texts) > CAP:
            raise ValueError('Dispatch exceeds the fixed slot count')
        ids = self.tokenize(texts, query)
        groups = defaultdict(list)
        for i, tokens in enumerate(ids):
            groups[self.width(tokens)].append((i, tokens))
        vectors = None
        for width, members in sorted(groups.items()):
            encoded = self.forward([tokens for _, tokens in members], width)
            if vectors is None:
                vectors = np.empty((len(texts), encoded.shape[1]), dtype=np.float32)
            vectors[[position for position, _ in members]] = encoded
        if vectors is None:
            raise ValueError('Empty fixed-shape encoder input')
        return vectors

    def tokenize(self, texts, query=False):
        inputs = [('Instruct: ' + TASK + '\nQuery:' + text) if query else text for text in texts]
        return self.tokenizer(inputs, padding=False, truncation=True, max_length=self.max_length)['input_ids']

    def width(self, tokens):
        return min(self.max_length, ((len(tokens) + BUCKET - 1) // BUCKET) * BUCKET)

    def forward(self, rows, width):
        if not 0 < len(rows) <= CAP or any(self.width(ids) != width for ids in rows):
            raise ValueError('Rows must share a fixed length bucket')
        padded = rows + [rows[0]] * (CAP - len(rows))
        with torch.inference_mode():
            batch = self.tokenizer.pad({'input_ids': padded}, padding='max_length',
                                       max_length=width, return_tensors='pt').to('cuda')
            hidden = self.model(**batch).last_hidden_state[:len(rows), -1, :].float()
            vectors = torch.nn.functional.normalize(hidden, p=2, dim=1).cpu().numpy()
        if not np.isfinite(vectors).all():
            raise ValueError('Invalid fixed-shape encoder output')
        self.forward_shapes[width] += 1
        self.real_rows[width] += len(rows)
        return vectors


def matching(a, b):
    return {
        'same_bytes': sum(x[0] == y[0] for x, y in zip(a, b, strict=True)),
        'same_experts_and_citations': sum(
            x[0] == y[0] or shape(x[0]) == shape(y[0]) for x, y in zip(a, b, strict=True)
        ),
    }


def run(output):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Requires Slurm CUDA')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    checkpoint = EMBEDDING / 'checkpoint-provenance.json'
    for name, expected in json.loads(checkpoint.read_text())['files'].items():
        if sha(EMBEDDING / name) != expected:
            raise ValueError('Changed encoder')
    model = QueryEncoderVariants()
    model.configure('no_cache')
    standard = model.encoder
    omitted_queries, omitted_sources, _ = exclusions()
    output.mkdir(parents=True, exist_ok=False)
    report = {'job_id': os.environ['SLURM_JOB_ID'], 'gpu': torch.cuda.get_device_name(),
              'script_sha256': sha(Path(__file__)), 'encoder_sha256': sha(checkpoint),
              'slots': CAP, 'length_bucket': BUCKET, 'datasets': {}}
    for i, dataset in enumerate(('SIGIR', DATASET)):
        options = {} if not i else {'exclude_queries': omitted_queries, 'exclude_sources': omitted_sources}
        ranker, db, queries, labels, inventory = prepare(dataset, output, checkpoint, **options)
        by_rate = {}
        try:
            for j, rate in enumerate(RATES):
                plan = load_plan(queries, rate)
                results, geometries = {}, {}
                offset = (i + j) % len(ARMS)
                for arm in ARMS[offset:] + ARMS[:offset]:
                    encoder = FixedShape(standard) if arm.startswith('fixed') else standard
                    mode = 'serial' if arm.endswith('serial') else 'microbatch8'
                    results[arm] = compare_method(ranker, db, encoder, mode, plan, queries)
                    if isinstance(encoder, FixedShape):
                        geometries[arm] = {
                            'forward_by_width': dict(encoder.forward_shapes),
                            'physical_slots': CAP * sum(encoder.forward_shapes.values()),
                            'used_slots': sum(encoder.real_rows.values()),
                            'includes_warmup': True,
                        }
                baseline = results['serial'][1]
                fixed = results['fixed_serial'][1]
                summarized = {}
                for arm, (rows, responses, runtime) in results.items():
                    data = stats(rows, responses, baseline, labels, runtime)
                    data['matching_fixed_serial'] = matching(responses, fixed)
                    if arm in geometries:
                        data['geometry'] = geometries[arm]
                    path = output / f'{dataset}-{rate}-{arm}.json'
                    path.write_text(json.dumps(rows, indent=2, allow_nan=False) + '\n')
                    data['rows_sha256'] = sha(path)
                    summarized[arm] = data
                by_rate[rate] = summarized
                print(dataset, rate, 'complete', flush=True)
        finally:
            db.unlink()
        report['datasets'][dataset] = {'index': inventory, 'loads': by_rate}
        del ranker
    report['scope'] = (
        'Two CoF loops with real persistent HTTP/TCP on a single host and frozen BF16 Qwen3. '
        'Canonical arms execute one fixed eight-slot tensor per 64-token width bucket in a dispatch. '
        'Dummy slots repeat the first real query and are discarded. No saved query representations. '
        'Byte parity is measured against canonical singleton scheduling and against ordinary '
        'unpadded serial scheduling. Timed HTTP scope excludes semantic verification.'
    )
    (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
