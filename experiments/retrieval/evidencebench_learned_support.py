"""Train lightweight support and redundancy estimates on EvidenceBench development papers."""
import argparse
import json
import os
from pathlib import Path
from statistics import mean

import numpy as np

from experiments.retrieval.evidencebench_character_budget import ROOT, sha
from expert_search.evidencebench_data import prepare, aspect_coverage
from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import clustered_paired_interval


def probability(logits):
    return 1 / (1 + np.exp(-np.clip(logits, -35, 35)))


def fit(features, labels):
    x = np.asarray(features)
    y = np.asarray(labels)
    center = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-6)
    x = np.column_stack((np.ones(len(x)), (x - center) / scale))
    weights = np.zeros(x.shape[1])
    penalty = np.diag([0.01] + [1.] * (len(weights) - 1))
    for _ in range(20):
        p = probability(x @ weights)
        curvature = x.T @ (x * (p * (1-p))[:, None]) + penalty
        weights -= np.linalg.solve(curvature, x.T @ (p-y) + penalty @ weights)
    return lambda features: probability(weights[0] + (np.asarray(features)-center) / scale @ weights[1:])


def select(support, overlap, costs, cap, adaptive):
    selected = []
    remaining = list(range(len(costs)))
    spent = 0
    while affordable := [i for i in remaining if spent + costs[i] <= cap + 1]:
        def value(i):
            redundancy = max((support[j] * overlap[i, j] for j in selected), default=0) if adaptive else 0
            return support[i] * (1 - redundancy) / costs[i]
        i = max(affordable, key=value)
        selected.append(i)
        remaining.remove(i)
        spent += costs[i]
    return selected, spent - bool(selected)


def main(output):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Requires Slurm allocation')
    checkpoint = Path('artifacts/models/qwen3-embedding-0.6b')
    manifest = checkpoint/'checkpoint-provenance.json'
    for name, expected in json.loads(manifest.read_text())['files'].items():
        if sha(checkpoint/name) != expected:
            raise ValueError(f'Changed model file: {name}')
    dev_eval = Path('artifacts/evidencebench-role-index-v1/evaluation.json')
    test_eval = ROOT/'evaluation.json'
    if sha(test_eval) != json.loads((ROOT/'verification.json').read_text())['artifacts'][str(test_eval)]:
        raise ValueError('Changed test ranking')
    dev, test = prepare('dev'), prepare('test')
    if {q['paper_id'] for q in dev.values()} & {q['paper_id'] for q in test.values()}:
        raise ValueError('Paper overlap')
    model = QwenTorchEmbedding(max_length=256, batch_size=32)

    def encoded(data, evaluation):
        records = json.loads(evaluation.read_text())
        if len(records) != len(data) or {r['id'] for r in records} != set(data):
            raise ValueError('Changed ranking cohort')
        for record in records:
            q = data[record['id']]
            ranking = record['selected']
            if aspect_coverage(q, ranking) != record['metrics']:
                raise ValueError('Changed saved ranking metrics')
            texts = [q['paper_as_candidate_pool'][i].strip() for i in ranking]
            v = model.encode([t or '[empty sentence]' for t in texts] + [
                'Instruct: Retrieve a sentence containing concrete evidence for this statement.\nQuery: ' + q['hypothesis']])
            relevance = v[:-1] @ v[-1]
            similarity = v[:-1] @ v[:-1].T
            features = np.column_stack((relevance, np.arange(len(texts))/10))
            pairs = np.stack(np.broadcast_arrays(similarity, relevance[:,None], relevance[None,:]), axis=-1)
            yield record['id'], q, ranking, features, pairs, np.array([len(t)+1 for t in texts])

    point_x, point_y, pair_x, pair_y = [], [], [], []
    for _, q, ranking, features, pairs, _ in encoded(dev, dev_eval):
        aspects = [set(q['sentence_index2aspects'].get(str(i), ())) for i in ranking]
        point_x.extend(features)
        point_y.extend(bool(a) for a in aspects)
        for i, left in enumerate(aspects):
            for j, right in enumerate(aspects):
                if i != j and left and right:
                    pair_x.append(pairs[i,j])
                    pair_y.append(len(left & right) / len(left))
    support_model = fit(point_x, point_y)
    overlap_model = fit(pair_x, pair_y)

    rows = []
    for key, q, ranking, features, pairs, costs in encoded(test, test_eval):
        support = support_model(features)
        overlap = overlap_model(pairs)
        row = {'id': key, 'paper_id': q['paper_id'], 'caps': {}}
        for cap in (500, 1000, 2000):
            arms = {}
            for method, adaptive in (('static_support', False), ('redundancy_adjusted', True)):
                selected, used = select(support, overlap, costs, cap, adaptive)
                arms[method] = {**aspect_coverage(q, [ranking[i] for i in selected]), 'characters': int(used)}
            row['caps'][str(cap)] = arms
        rows.append(row)
    report = {'dev_queries': len(dev), 'test_queries': len(test), 'dev_sentence_samples': len(point_y),
              'dev_supporting_sentence_pairs': len(pair_y), 'job_id': os.environ['SLURM_JOB_ID'],
              'model_sha256': sha(manifest), 'script_sha256': sha(Path(__file__)),
              'inputs_sha256': {p.name if p.parent == Path('artifacts/public/evidencebench') else str(p): sha(p)
                                for p in (dev_eval, test_eval, *(Path('artifacts/public/evidencebench')/f'{s}.json' for s in ('dev','test')))},
              'caps': {}, 'scope': 'Development-only fitting. Fixed previously explored test pools. Frozen Qwen cosine and rank predict sentence support, and pair cosine plus query similarities predict directional aspect overlap conditional on both sentences supporting at least one aspect. Uses only predicted values for test selection. Character budgets omit serving and verification. Does not evaluate expert fit, semantic entailment, or an independent confirmatory test.'}
    for cap in ('500','1000','2000'):
        batch = [r['caps'][cap] for r in rows]
        report['caps'][cap] = {arm: {m: mean(r[arm][m] for r in batch)
                                   for m in ('aspect_recall','characters')} for arm in batch[0]}
        report['caps'][cap]['adjusted_minus_static'] = clustered_paired_interval(
            [r['redundancy_adjusted']['aspect_recall']-r['static_support']['aspect_recall'] for r in batch],
            [r['paper_id'] for r in rows])
    output.mkdir(parents=True, exist_ok=False)
    (output/'rows.json').write_text(json.dumps(rows, indent=2)+'\n')
    report['rows_sha256'] = sha(output/'rows.json')
    (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
