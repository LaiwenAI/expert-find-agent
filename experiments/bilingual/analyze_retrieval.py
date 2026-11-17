"""Parent/company-group bootstrap and transfer checks for bilingual diagnostics."""
import hashlib
import json
from pathlib import Path
import numpy as np


def run():
    root = Path('artifacts/bilingual-retrieval-v1')
    summary = json.loads((root / 'summary.json').read_text())
    rows = [json.loads(s) for s in (root / 'evaluation.jsonl').read_text().splitlines()]
    queries = [json.loads(s) for s in (root / 'queries.jsonl').read_text().splitlines()]
    if len(rows) != 1000 or [r['id'] for r in rows] != [r['id'] for r in queries] or len({r['parent_id'] for r in rows}) != 1000:
        raise ValueError('Incomplete or changed paired cohort')
    for name, expected in [('evaluation.jsonl', summary['evaluation_sha256']), ('protocol.json', summary['protocol_sha256']), ('query-vectors.npz', summary['encoding']['query_vectors_sha256'])]:
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Changed diagnostic file')
    with np.load(root / 'query-vectors.npz', allow_pickle=False) as cache:
        vectors = cache['vectors']
        norms = np.linalg.norm(vectors, axis=1)
        if vectors.shape != (2000, 1024) or not np.isfinite(vectors).all() or not np.allclose(norms, 1, atol=2e-5):
            raise ValueError('Invalid query vectors')
    groups = sorted({r['group_id'] for r in rows})
    grouped = [[r for r in rows if r['group_id'] == group] for group in groups]
    rng = np.random.default_rng(20261117)
    columns = ['exact_overlap', 'ef64_overlap', 'ef512_overlap', 'ef64_minus_exact', 'ef512_minus_exact']
    values = lambda r: [r['language_overlap']['exact'], r['language_overlap']['ef64'], r['language_overlap']['ef512'], r['language_overlap']['ef64'] - r['language_overlap']['exact'], r['language_overlap']['ef512'] - r['language_overlap']['exact']]
    sums = np.asarray([np.sum([values(r) for r in g], axis=0) for g in grouped])
    sizes = np.asarray([len(g) for g in grouped])
    samples = []
    for _ in range(2000):
        draw = rng.integers(0, len(groups), size=len(groups))
        samples.append(sums[draw].sum(axis=0) / sizes[draw].sum())
    intervals = np.quantile(samples, [.025, .975], axis=0)
    means = np.mean([values(r) for r in rows], axis=0)
    result = dict(pairs=len(rows), independent_parent_ids=1000, connected_company_groups=len(groups), bootstrap_replicates=2000, seed=20261117,
                  estimates={name: dict(mean=float(means[i]), percentile_interval=[float(intervals[0, i]), float(intervals[1, i])]) for i, name in enumerate(columns)},
                  input_sha256={str(root / name): hashlib.sha256((root / name).read_bytes()).hexdigest() for name in ['evaluation.jsonl', 'protocol.json', 'query-vectors.npz', 'queries.jsonl']}, code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='Sampling intervals for observed ranking stability in this construction cohort, not expert relevance, semantic correctness, equivalence, or held-out generalization. Parent/source translations remain model-screened only; exact ranking reference is not relevance gold.')
    (root / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('input_sha256', 'code_sha256')}, indent=2))


if __name__ == '__main__':
    run()
