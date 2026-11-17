"""Measure association between bilingual query cosine and ranking overlap."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('artifacts/bilingual-retrieval-v1')


def analyze(rows):
    if len(rows) < 4 or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Insufficient or duplicate pairs')
    cosine = np.array([r['query_cosine'] for r in rows], dtype=float)
    overlap = np.array([r['language_overlap']['exact'] for r in rows], dtype=float)
    if not np.isfinite(cosine).all() or not np.isfinite(overlap).all() or np.any((overlap < 0) | (overlap > 1)):
        raise ValueError('Invalid observations')
    # Sort using ID as an explicit deterministic tie-breaker; bins have equal size.
    order = sorted(range(len(rows)), key=lambda i: (cosine[i], rows[i]['id']))
    groups = np.array([r['group_id'] for r in rows])
    bins = []
    for index in np.array_split(order, 4):
        bins.append(dict(count=len(index), cosine_range=[float(cosine[index].min()), float(cosine[index].max())],
                         mean_cosine=float(cosine[index].mean()), mean_top10_overlap=float(overlap[index].mean()),
                         fraction_overlap_at_most_0_2=float(np.mean(overlap[index] <= .2))))
    unique = np.unique(groups)
    group_indices = [np.flatnonzero(groups == group) for group in unique]
    rng = np.random.default_rng(20261118)
    correlations = []
    for _ in range(2000):
        indices = np.concatenate([group_indices[i] for i in rng.integers(0, len(unique), len(unique))])
        if np.std(cosine[indices]) > 0 and np.std(overlap[indices]) > 0:
            correlations.append(float(np.corrcoef(cosine[indices], overlap[indices])[0, 1]))
    if not correlations or np.std(cosine) == 0 or np.std(overlap) == 0:
        raise ValueError('Correlation undefined')
    return dict(pairs=len(rows), groups=len(unique), pearson_r=float(np.corrcoef(cosine, overlap)[0, 1]),
                cluster_bootstrap_percentile_interval=list(map(float, np.quantile(correlations, [.025, .975]))),
                bootstrap_replicates=len(correlations), seed=20261118, quartiles=bins,
                scope='Post-hoc descriptive association on construction pairs; company-group bootstrap reflects sampling variation under a group-resampling assumption. Cosine requires both language views. Neither overlap nor cosine measures expert relevance or verifies translation equivalence; no causal, held-out or deployment claim.')


def main():
    summary = json.loads((ROOT / 'summary.json').read_text())
    path = ROOT / 'evaluation.jsonl'
    digest = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
    if digest != summary['evaluation_sha256']:
        raise ValueError('Changed diagnostic data')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    result = analyze(rows)
    result['evaluation_sha256'] = digest
    result['code_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output = ROOT / 'alignment-dependence.json'
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
