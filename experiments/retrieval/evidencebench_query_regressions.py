"""Describe per-query gains and regressions in frozen EvidenceBench selections."""
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from expert_search.evidencebench_data import aspect_coverage


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    root = Path('artifacts/evidencebench-role-test-v1')
    data_path = Path('artifacts/public/evidencebench/test.json')
    protocol = json.loads((root / 'protocol.json').read_text())
    if sha(data_path) != protocol['dataset_sha256']:
        raise ValueError('Dataset hash mismatch')
    data = json.loads(data_path.read_text())
    rows = json.loads((root / 'evaluation.json').read_text())
    if len(rows) != len(data) or {r['id'] for r in rows} != set(data):
        raise ValueError('Incomplete cohort')
    changes = []
    for row in rows:
        source = data[row['id']]
        for name, field in (('metrics', 'selected'), ('baseline', 'baseline_selected')):
            if row[name] != aspect_coverage(source, row[field]):
                raise ValueError('Frozen coverage mismatch')
            if len(row[field]) != 10 or len(set(row[field])) != 10:
                raise ValueError('Invalid sentence selection')
        old, new = row['baseline']['aspect_recall'], row['metrics']['aspect_recall']
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in (old, new)):
            raise ValueError('Invalid recall')
        changes.append({'delta': new-old, 'baseline': old, 'new': new,
                        'same_set': set(row['selected']) == set(row['baseline_selected'])})
    counts = Counter('improved' if x['delta'] > 1e-12 else 'worsened' if x['delta'] < -1e-12 else 'unchanged' for x in changes)
    def group(fn):
        chosen = [x for x in changes if fn(x)]
        return {'queries': len(chosen), 'mean_difference': statistics.mean(x['delta'] for x in chosen) if chosen else None,
                'worsened': sum(x['delta'] < -1e-12 for x in chosen)}
    result = {'queries': len(changes), 'comparison_counts': dict(counts),
              'mean_difference': statistics.mean(x['delta'] for x in changes),
              'worst_change': min(x['delta'] for x in changes), 'best_change': max(x['delta'] for x in changes),
              'same_selected_set': sum(x['same_set'] for x in changes),
              'baseline_zero': group(lambda x: x['baseline'] == 0),
              'baseline_complete': group(lambda x: x['baseline'] == 1),
              'baseline_intermediate': group(lambda x: 0 < x['baseline'] < 1),
              'input_sha256': {'dataset': sha(data_path), 'protocol': sha(root/'protocol.json'),
                               'evaluation': sha(root/'evaluation.json'), 'script': sha(Path(__file__))},
              'scope': 'Post-hoc descriptive analysis of 293 reused EvidenceBench test queries. The original labels are model-assisted, not independent industrial expert-fit judgments. Baseline-defined subgroups and extremes are selected after observing the outcome; unequal ceiling and floor make these subgroup comparisons noncausal. Counts ignore shared source-paper dependence and do not estimate generalization, attribution correctness, quality preservation, or a deployable fallback policy.'}
    output = Path('artifacts/evidencebench-query-regressions-v1')
    output.mkdir(parents=True, exist_ok=False)
    (output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
