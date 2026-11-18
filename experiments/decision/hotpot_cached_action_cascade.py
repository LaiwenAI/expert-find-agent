"""Measure pre-indexed dense retrieval and validation-calibrated evidence routing."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import paired_interval
from experiments.decision.hotpot_evidence_action import CANDIDATE_LIMIT, DATA_SHA256, action_features, candidate_actions, sha
from experiments.decision.hotpot_evidence_action_timing import EvidenceHead, SEED
from experiments.hotpotqa.semantic_complement import INSTRUCTION, verify_model

RATES = (.10, .25)
METHODS = ('mlp', 'cached_dense', 'gate10', 'gate25', 'random25')


def calibrate(pack, expected_sha, head):
    if sha(pack) != expected_sha:
        raise ValueError('Calibration pack differs from the original training run')
    with np.load(pack, allow_pickle=False) as data:
        features, counts = data['validation_features'], data['validation_count']
    gaps = [margin(head.scores(x[:n])) for x, n in zip(features, counts, strict=True) if n > 1]
    thresholds = {f'gate{int(rate * 100)}': float(np.quantile(gaps, rate)) for rate in RATES}
    return thresholds, {'validation_questions': len(counts), 'eligible': len(gaps),
                        'thresholds': thresholds,
                        'selected': {k: sum(x <= v for x in gaps) for k, v in thresholds.items()}}


def margin(scores):
    top = np.partition(scores, -2)[-2:]
    return float(top[1] - top[0])


def build_index(records, encoder):
    started = time.perf_counter()
    passages = sorted({'\n'.join([title, *parts])
                       for row in records for title, parts in row['context']})
    batch_size = 32
    encoder.batch_size = batch_size
    matrices = []
    for offset in range(0, len(passages), 4 * batch_size):
        matrices.append(encoder.encode(passages[offset:offset + 4 * batch_size]))
    vectors = np.concatenate(matrices)
    content_hash = hashlib.sha256()
    for passage in passages:
        content_hash.update(json.dumps(passage, ensure_ascii=False).encode() + b'\n')
    index = {passage: i for i, passage in enumerate(passages)}
    encoder.batch_size = CANDIDATE_LIMIT
    return index, vectors, {'documents': len(passages), 'vector_bytes': vectors.nbytes,
                            'build_seconds': time.perf_counter() - started,
                            'ordered_passage_sha256': content_hash.hexdigest(),
                            'vectors_sha256': hashlib.sha256(vectors).hexdigest()}


def request(row, method, encoder, head, index, vectors, thresholds):
    started = time.perf_counter()
    question = row['question']
    titles, text, first, actions, rank = candidate_actions(question, row['context'])
    retrieved = time.perf_counter()
    choice, use_dense = 0, False
    if len(actions) > 1:
        if method == 'cached_dense':
            use_dense = True
        else:
            scores = head.scores(action_features(question, titles, text, first, actions, rank))
            choice = int(scores.argmax())
            if method in thresholds:
                use_dense = margin(scores) <= thresholds[method]
            elif method == 'random25':
                digest = hashlib.sha256(f'{SEED}:{row["_id"]}'.encode()).digest()
                use_dense = digest[0] < 64
        if use_dense:
            query_vector = encoder.encode([INSTRUCTION + question])[0]
            candidate_vectors = vectors[[index[text[i]] for i in actions]]
            choice = int((candidate_vectors @ query_vector).argmax())
    decided = time.perf_counter()
    second = actions[choice]
    payload = json.dumps([(titles[i], text[i]) for i in (first, second)], ensure_ascii=False).encode()
    delivered = time.perf_counter()
    return {'first': first, 'second': second, 'choice': choice, 'had_choice': len(actions) > 1,
            'used_gpu': use_dense, 'chars': len(text[first]) + 1 + len(text[second]),
            'bytes': len(payload), 'decision_ms': 1000 * (decided - retrieved),
            'total_ms': 1000 * (delivered - started)}


def summarize(rows):
    summary = {}
    for method in METHODS:
        arm = [row[method] for row in rows]
        summary[method] = {
            'complete': sum(r['complete'] for r in arm), 'gpu_calls': sum(r['used_gpu'] for r in arm),
            'mean_chars': float(np.mean([r['chars'] for r in arm])),
        }
        for field in ('decision_ms', 'total_ms'):
            summary[method][field] = dict(zip(('p50', 'p95', 'p99'),
                map(float, np.quantile([r[field] for r in arm], [.50, .95, .99])), strict=True))
    compared = {}
    for name in METHODS[1:]:
        diff = [int(row[name]['complete']) - int(row['mlp']['complete']) for row in rows]
        compared[f'{name}_minus_mlp'] = paired_interval(diff, repeats=2000)
    diff = [int(r['gate25']['complete']) - int(r['random25']['complete']) for r in rows]
    compared['gate25_minus_random25'] = paired_interval(diff, repeats=2000)
    return {'questions': len(rows), 'arms': summary, 'compared': compared,
            'oracle_either_complete': sum(r['mlp']['complete'] or r['cached_dense']['complete'] for r in rows)}


def run(dataset, trained, pack, output):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Requires a Slurm CUDA allocation')
    if sha(dataset) != DATA_SHA256:
        raise ValueError('Changed benchmark data')
    spec = json.loads((trained / 'summary.json').read_text())
    checkpoint, replay = trained / 'mlp.npz', trained / 'replay.npz'
    if sha(checkpoint) != spec['models']['mlp']['weights_sha256'] or sha(replay) != spec['replay_sha256']:
        raise ValueError('Changed model or replay')
    with np.load(replay, allow_pickle=False) as saved:
        expected = dict(zip(saved['ids'], saved['mlp'], strict=True))
    head = EvidenceHead(checkpoint)
    thresholds, calibration = calibrate(pack, spec['input_sha256'], head)
    model_hash = verify_model()
    encoder = QwenTorchEmbedding(max_length=512, batch_size=32)
    encoder.model.config.use_cache = False
    pool = json.loads(dataset.read_text())
    index, vectors, cache = build_index(pool, encoder)
    print('Built index', cache['documents'], 'documents', flush=True)
    records = [r for r in pool if r['_id'] in expected]
    if len(records) != len(expected):
        raise ValueError('Changed evaluation cohort')
    np.random.default_rng(SEED).shuffle(records)
    for row in records[:12]:
        for method in METHODS:
            request(row, method, encoder, head, index, vectors, thresholds)

    rows = []
    for i, row in enumerate(records):
        offset = i % len(METHODS)
        order = METHODS[offset:] + METHODS[:offset]
        arms = {m: request(row, m, encoder, head, index, vectors, thresholds) for m in order}
        if arms['mlp']['choice'] != expected[row['_id']]:
            raise ValueError('Frozen decision replay mismatch')
        gold = {title for title, _ in row['supporting_facts']}
        for arm in arms.values():
            if arm['first'] != arms['mlp']['first']:
                raise ValueError('First passage changed across arms')
            arm['complete'] = gold <= {row['context'][j][0] for j in (arm['first'], arm['second'])}
        rows.append({'id': row['_id'], **arms})
        if len(rows) % 1500 == 0:
            print(f'{len(rows)} live requests per arm', flush=True)
    output.mkdir(parents=True, exist_ok=False)
    path = output / 'rows.json'
    path.write_text(json.dumps(rows, indent=2, allow_nan=False) + '\n')
    result = {'job_id': os.environ['SLURM_JOB_ID'], 'gpu': encoder.torch.cuda.get_device_name(),
              'checkpoint_sha256': sha(checkpoint), 'index_model_sha256': model_hash,
              'dataset_sha256': DATA_SHA256, 'calibration_pack_sha256': sha(pack), 'calibration': calibration,
              'index': cache, 'script_sha256': sha(Path(__file__)), 'rows_sha256': sha(path),
              'all': summarize(rows),
              'with_alternative': summarize([r for r in rows if r['mlp']['had_choice']]),
              'scope': 'Warm, single requests with an in-memory passage-vector index built without '
                       'labels. Timing includes per-request BM25, action scoring and JSON. The cache '
                       'build is measured separately. No global search, network, or evidence verification.'}
    (output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--trained', type=Path, required=True)
    parser.add_argument('--pack', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.dataset, args.trained, args.pack, args.output)
