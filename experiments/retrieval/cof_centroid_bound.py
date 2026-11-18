"""Measure whether per-expert vector bounds prune exact CoF candidate search."""
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np

from expert_search.cof_benchmark import load


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root, _, reviewers, labels = load('SIGIR')
    manifest = Path('artifacts/cof-gpu-public-input-v1/input-provenance.json')
    provenance = json.loads(manifest.read_text())
    for file, expected in provenance['input_sha256'].items():
        if digest(file) != expected:
            raise ValueError(f'Changed input: {file}')

    queries = provenance['query_ids']
    experts = sorted(reviewers)
    papers = sorted({p for reviewer in reviewers.values() for p in reviewer['papers']})
    positions = {paper: i for i, paper in enumerate(papers)}
    owners = [np.array(sorted({positions[p] for p in reviewers[e]['papers']})) for e in experts]
    vectors = np.load(root/'qwen-embedding-torch-v1/vectors.npy', allow_pickle=False).astype(np.float64)
    if len(vectors) != len(papers) + len(queries):
        raise ValueError('Unexpected vector count')
    source_vectors = vectors[:len(papers)]
    def split(rows, count):
        local = source_vectors[rows]
        selected = [0]
        while len(selected) < min(count, len(rows)):
            distances = np.linalg.norm(local[:, None] - local[selected], axis=2).min(axis=1)
            distances[selected] = -1
            selected.append(int(np.argmax(distances)))
        assignment = np.linalg.norm(local[:, None] - local[selected], axis=2).argmin(axis=1)
        return [rows[assignment == j] for j in range(len(selected)) if np.any(assignment == j)]

    layouts = {}
    for count in (1, 2, 4):
        parts = [(e, part) for e, rows in enumerate(owners) for part in split(rows, count)]
        parent, groups = zip(*parts)
        centers = np.stack([source_vectors[rows].mean(axis=0) for rows in groups])
        radii = np.array([np.linalg.norm(source_vectors[rows] - center, axis=1).max()
                          for rows, center in zip(groups, centers)])
        layouts[count] = groups, parent, centers, radii

    results = []
    for i, query in enumerate(queries):
        vector = vectors[len(papers) + i]
        exact = np.einsum('ij,j->i', source_vectors, vector, optimize=False)
        expert_scores = [exact[rows].max() for rows in owners]
        exact_order = sorted(range(len(experts)), key=lambda e: (-expert_scores[e], e))
        row = {'query_id': query, 'layouts': {}}
        for count, (groups, parent, centers, radii) in layouts.items():
            center_scores = centers @ vector
            bounds = center_scores + radii * np.linalg.norm(vector) + 1e-10
            order = np.argsort(-bounds, kind='stable')
            if any(exact[rows].max() > bound for rows, bound in zip(groups, bounds)):
                raise ValueError('Computed upper bound violated')
            best_centers = np.full(len(experts), -np.inf)
            np.maximum.at(best_centers, list(parent), center_scores)
            row['layouts'][str(count)] = {}
            for budget in (5, 10, 20):
                best = {}
                cache = {}
                ranked = []
                for group in order:
                    if len(ranked) >= budget and bounds[group] < best[ranked[-1]]:
                        break
                    missing = [p for p in groups[group] if p not in cache]
                    if missing:
                        cache.update(zip(missing, np.einsum(
                            'ij,j->i', source_vectors[missing], vector, optimize=False)))
                    score = max(cache[p] for p in groups[group])
                    expert = parent[group]
                    best[expert] = max(best.get(expert, -np.inf), score)
                    ranked = sorted(best, key=lambda e: (-best[e], e))[:budget]
                centroid_only = np.argsort(-best_centers, kind='stable')[:budget]
                row['layouts'][str(count)][str(budget)] = {
                    'exact_match': ranked == exact_order[:budget],
                    'source_fraction': len(cache) / len(papers),
                    'total_dot_products_fraction': (len(cache) + len(groups)) / len(papers),
                    'centroid_only_topk_overlap': len(set(centroid_only) & set(exact_order[:budget])) / budget,
                    'centroid_only_soft_precision': sum(labels['soft'][query][experts[e]] > 0 for e in centroid_only) / budget,
                    'exact_soft_precision': sum(labels['soft'][query][experts[e]] > 0 for e in exact_order[:budget]) / budget,
                }
        results.append(row)

    report = {'queries': len(queries), 'experts': len(experts), 'sources': len(papers),
              'provenance_sha256': digest(manifest), 'script_sha256': digest(__file__), 'layouts': {},
              'scope': 'Reused public queries and cached vectors. The bounds follow Cauchy-Schwarz in real arithmetic with an FP64 guard and empirical checks. Dot products are counted separately from actual latency. Exact-top-k overlap is not relevance. Soft labels are reviewer self-ratings.'}
    for count, (groups, _, _, _) in layouts.items():
        aggregate = {'groups': len(groups), 'budgets': {}}
        for budget in ('5', '10', '20'):
            entries = [r['layouts'][str(count)][budget] for r in results]
            aggregate['budgets'][budget] = {k: statistics.mean(e[k] for e in entries) for k in entries[0]}
        report['layouts'][str(count)] = aggregate
    out = Path('artifacts/cof-centroid-bound-v2')
    out.mkdir(exist_ok=False)
    (out/'rows.json').write_text(json.dumps(results, indent=2)+'\n')
    report['rows_sha256'] = digest(out/'rows.json')
    (out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
