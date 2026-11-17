"""Measure effects of random per-reviewer evidence caps on frozen dense retrieval."""
import argparse
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

import numpy as np
from expert_search.cof_benchmark import load
from expert_search.statistics import paired_interval


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, _, reviewers, labels = load('SIGIR')
    vector = root / 'qwen-embedding-torch-v1'
    matrix_path = vector / 'vectors.npy'
    if sha(matrix_path) != json.loads((vector / 'summary.json').read_text())['vectors_sha256']:
        raise ValueError('Vector hash mismatch')
    pids = sorted({pid for r in reviewers.values() for pid in r['papers']})
    qids = sorted(labels['soft'])
    eids = sorted(reviewers)
    if set(qids) != set(labels['hard']) or any(set(labels[t][q]) != set(eids) for t in labels for q in qids):
        raise ValueError('Incomplete label matrix')
    matrix = np.load(matrix_path, allow_pickle=False)
    if matrix.shape != (len(pids) + len(qids), 1024):
        raise ValueError('Vector dimensions changed')
    position = {pid: i for i, pid in enumerate(pids)}
    sources = {eid: [position[pid] for pid in sorted(set(reviewers[eid]['papers']))] for eid in eids}
    if any(not x for x in sources.values()):
        raise ValueError('Empty reviewer profile')
    scores = matrix[len(pids):] @ matrix[:len(pids)].T
    median = statistics.median(len(x) for x in sources.values())
    full = np.stack([scores[:, sources[eid]].max(axis=1) for eid in eids], axis=1)
    frozen = {r['query_id']: r['rankings']['cpu'] for r in json.loads(Path('artifacts/cof-gpu-quality-v1/rows.json').read_text())}
    def ranking(row):
        return sorted(range(len(eids)), key=lambda i: (-float(row[i]), eids[i]))[:5]
    if any([eids[i] for i in ranking(full[j])] != frozen[q][:5] for j, q in enumerate(qids)):
        raise ValueError('Baseline does not reproduce frozen ranking')
    started = time.perf_counter()
    selected = {}
    for eid in eids:
        ids = sources[eid]
        docs = matrix[ids]
        if len(ids) <= 7:
            selected[eid] = ids
            continue
        norms = np.linalg.norm(docs, axis=1, keepdims=True)
        unit = docs / np.maximum(norms, 1e-30)
        centroid = unit.mean(axis=0)
        first = int(np.argmax(unit @ centroid))
        chosen = [first]
        similarities = unit @ unit[first]
        while len(chosen) < 7:
            distances = 1.0 - similarities
            distances[chosen] = -np.inf
            nxt = int(np.argmax(distances))
            chosen.append(nxt)
            similarities = np.maximum(similarities, unit @ unit[nxt])
        selected[eid] = [ids[i] for i in chosen]
    construction_seconds = time.perf_counter() - started
    diverse = np.stack([scores[:, selected[eid]].max(axis=1) for eid in eids], axis=1)
    measurements = {q: [] for q in qids}
    baseline = []
    for j, q in enumerate(qids):
        ids = [eids[i] for i in ranking(full[j])]
        baseline.append({'above_median': sum(len(sources[e]) > median for e in ids)/5,
                         **{t: sum(labels[t][q][e] > 0 for e in ids)/5 for t in labels}})
    for seed in range(20):
        rng = random.Random(20261121 + seed)
        capped = {eid: rng.sample(sources[eid], min(7, len(sources[eid]))) for eid in eids}
        capped_scores = np.stack([scores[:, capped[eid]].max(axis=1) for eid in eids], axis=1)
        for j, q in enumerate(qids):
            ids = [eids[i] for i in ranking(capped_scores[j])]
            measurements[q].append({'above_median': sum(len(sources[e]) > median for e in ids)/5,
                                    **{t: sum(labels[t][q][e] > 0 for e in ids)/5 for t in labels}})
    fields = ('above_median', 'soft', 'hard')
    rows = [{'query_id': q, 'baseline': baseline[j],
             'capped_mean': {f: statistics.mean(x[f] for x in measurements[q]) for f in fields}}
            for j, q in enumerate(qids)]
    for j, q in enumerate(qids):
        ids = [eids[i] for i in ranking(diverse[j])]
        rows[j]['diverse'] = {'above_median': sum(len(sources[e]) > median for e in ids)/5,
                              **{t: sum(labels[t][q][e] > 0 for e in ids)/5 for t in labels}}
    report = {'construction_seconds': construction_seconds,
              'diverse': {f: statistics.mean(r['diverse'][f] for r in rows) for f in fields},
              'diverse_minus_random': {f: paired_interval([r['diverse'][f]-r['capped_mean'][f] for r in rows]) for f in fields},
              'diverse_minus_full': {f: paired_interval([r['diverse'][f]-r['baseline'][f] for r in rows]) for f in fields},
              'queries': len(qids), 'reviewers': len(eids), 'cap': 7, 'samples': 20, 'seed_start': 20261121,
              'baseline': {f: statistics.mean(r['baseline'][f] for r in rows) for f in fields},
              'capped': {f: statistics.mean(r['capped_mean'][f] for r in rows) for f in fields},
              'paired_cap_minus_full': {f: paired_interval([r['capped_mean'][f] - r['baseline'][f] for r in rows]) for f in fields},
              'input_sha256': {'vectors': sha(matrix_path), 'frozen_rows': sha(Path('artifacts/cof-gpu-quality-v1/rows.json')),
                **{n: sha(root/n) for n in ('SIGIR_reviewers_test.json','SIGIR_queries_test_soft.json','SIGIR_queries_test_hard.json')}, 'script': sha(Path(__file__))},
              'scope': 'Post-hoc diversity-based capped source selection and random intervention on reused public reviewer task. Diversity selects the source nearest the centroid then greedily the farthest source by cosine distance; no labels or query used. Selection may retain outliers. Construction time excludes encoder. Full scoring was computed first and is not a retrieval speed measurement. Randomly sample at most seven papers per reviewer, fixed across queries for each seed; scores are computed for the full corpus first. Reviewer-level labels need not describe retained papers. Query bootstrap on seed-averaged outcomes does not account for reviewer, query dependence or seed sampling uncertainty. No evidence-support, enterprise fit, contribution, causal fairness or end-to-end speed conclusion; no index speedup is measured.'}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'rows.json').write_text(json.dumps(rows, indent=2)+'\n')
    (args.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
