"""Measure single-request evidence selection with a frozen CPU decision head."""

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import paired_interval
from experiments.decision.hotpot_evidence_action import (
    CANDIDATE_LIMIT, DATA_SHA256, action_features, candidate_actions, sha, words,
)
from experiments.hotpotqa.semantic_complement import INSTRUCTION, verify_model

METHODS = ('bm25', 'missing_terms', 'mlp', 'dense')
SEED = 20261119


class EvidenceHead:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as weights:
            self.center, self.scale = weights['center'], weights['scale']
            self.input_weight, self.input_bias = weights['0.weight'], weights['0.bias']
            self.output_weight, self.output_bias = weights['2.weight'][0], weights['2.bias'][0]

    def scores(self, features):
        hidden = np.maximum((features - self.center) / self.scale @ self.input_weight.T + self.input_bias, 0)
        return hidden @ self.output_weight + self.output_bias

    def choose(self, features):
        return int(self.scores(features).argmax())


def request(question, context, method, encoder, head):
    started = time.perf_counter()
    titles, text, first, actions, rank = candidate_actions(question, context)
    retrieved = time.perf_counter()
    choice = 0
    if len(actions) > 1:
        if method == 'missing_terms':
            missing = words(question) - words(text[first])
            choice = max(range(len(actions)), key=lambda slot: len(missing & words(text[actions[slot]])))
        elif method == 'mlp':
            choice = head.choose(action_features(question, titles, text, first, actions, rank))
        elif method == 'dense':
            query, *candidates = encoder.encode([INSTRUCTION + question, *(text[d] for d in actions)])
            choice = int((np.array(candidates) @ query).argmax())
    decided = time.perf_counter()
    second = actions[choice]
    payload = json.dumps([(titles[i], text[i]) for i in (first, second)], ensure_ascii=False).encode()
    delivered = time.perf_counter()
    return {
        'first': first, 'second': second, 'action': choice, 'had_choice': len(actions) > 1,
        'chars': len(text[first]) + 1 + len(text[second]), 'bytes': len(payload),
        'selection_ms': (decided - retrieved) * 1000, 'total_ms': (delivered - started) * 1000,
    }


def aggregate(rows):
    summary = {}
    for method in METHODS:
        arm = [row[method] for row in rows]
        metric = {'complete': sum(a['complete'] for a in arm),
                  'mean_chars': float(np.mean([a['chars'] for a in arm]))}
        for field in ('total_ms', 'selection_ms'):
            metric[field] = dict(zip(('p50', 'p95', 'p99'),
                                     map(float, np.quantile([r[field] for r in arm], [.5, .95, .99])),
                                     strict=True))
        summary[method] = metric
    dense_diff = [int(r['mlp']['complete']) - int(r['dense']['complete']) for r in rows]
    return {'queries': len(rows), 'arms': summary, 'mlp_minus_dense': paired_interval(dense_diff, repeats=2000)}


def main(dataset, trained, output):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Requires a Slurm CUDA allocation')
    if sha(dataset) != DATA_SHA256:
        raise ValueError('Changed evaluation dataset')
    summary_path = trained / 'summary.json'
    original = json.loads(summary_path.read_text())
    checkpoint, replay = trained / 'mlp.npz', trained / 'replay.npz'
    if sha(checkpoint) != original['models']['mlp']['weights_sha256'] or sha(replay) != original['replay_sha256']:
        raise ValueError('Changed model or replay')
    with np.load(replay, allow_pickle=False) as archive:
        expected = dict(zip(archive['ids'], archive['mlp'], strict=True))

    model_hash = verify_model()
    encoder = QwenTorchEmbedding(max_length=512, batch_size=CANDIDATE_LIMIT)
    head = EvidenceHead(checkpoint)
    records = [r for r in json.loads(dataset.read_text()) if r['_id'] in expected]
    if len(records) != len(expected):
        raise ValueError('Changed evaluation cohort')
    rng = np.random.default_rng(SEED)
    rng.shuffle(records)
    for record in records[:12]:
        for method in METHODS:
            request(record['question'], record['context'], method, encoder, head)

    rows = []
    for i, record in enumerate(records):
        order = METHODS[i % len(METHODS):] + METHODS[:i % len(METHODS)]
        arms = {m: request(record['question'], record['context'], m, encoder, head) for m in order}
        if arms['mlp']['action'] != expected[record['_id']]:
            raise ValueError('CPU action differs from the frozen GPU replay')
        gold = {title for title, _ in record['supporting_facts']}
        for arm in arms.values():
            if arm['first'] != arms['bm25']['first'] or arm['chars'] > arms['bm25']['chars']:
                raise ValueError('Mismatched first choice or exceeded evidence budget')
            arm['complete'] = gold <= {record['context'][j][0] for j in (arm['first'], arm['second'])}
        rows.append({'id': record['_id'], **arms})
        if len(rows) % 1500 == 0:
            print(f'{len(rows)} measured', flush=True)
    if any(sum(row[m]['complete'] for row in rows) != original['outcomes'][m]['complete']
           for m in ('bm25', 'missing_terms', 'mlp')):
        raise ValueError('Candidate action quality does not reproduce the frozen replay')

    output.mkdir(parents=True, exist_ok=False)
    path = output / 'rows.json'
    path.write_text(json.dumps(rows, indent=2, allow_nan=False) + '\n')
    summary = {
        'job_id': os.environ['SLURM_JOB_ID'], 'gpu': encoder.torch.cuda.get_device_name(),
        'dataset_sha256': DATA_SHA256, 'head_sha256': sha(checkpoint), 'embedding_manifest_sha256': model_hash,
        'training_summary_sha256': sha(summary_path), 'rows_sha256': sha(path), 'script_sha256': sha(Path(__file__)),
        'all': aggregate(rows), 'with_alternative': aggregate([r for r in rows if r['bm25']['had_choice']]),
        'scope': 'Warm single requests. Measured from provided passage pool through BM25, action '
                 'selection and two-passage JSON serialization. Dense encodes query and eligible '
                 'second passages only. Candidate availability, response labels and startup are outside '
                 'the timer; no network, global retrieval, answer generation or expert search.',
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--trained', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.dataset, args.trained, args.output)
