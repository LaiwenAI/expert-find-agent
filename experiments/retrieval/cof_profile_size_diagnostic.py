"""Post-hoc profile-size exposure diagnostics on frozen CoF SIGIR rankings."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

from expert_search.cof_benchmark import load
from expert_search.statistics import paired_interval


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=Path, default=Path('artifacts/cof-gpu-quality-v1/rows.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, _, reviewers, labels = load('SIGIR')
    rows = json.loads(args.rows.read_text())
    sizes = {eid: len(set(r['papers'])) for eid, r in reviewers.items()}
    ordered = sorted(sizes.values())
    median = statistics.median(ordered)
    arms = ('lexical', 'gpu')
    results = []
    for row in rows:
        qid = row['query_id']
        if qid not in labels['soft'] or set(labels['soft'][qid]) != set(reviewers):
            raise ValueError('Incomplete or incompatible reviewer labels')
        arm_data = {}
        for arm in arms:
            ids = row['rankings'][arm][:5]
            if len(ids) != 5 or len(set(ids)) != 5 or not set(ids) <= set(reviewers):
                raise ValueError('Invalid frozen ranking')
            arm_data[arm] = {
                'mean_profile_size': statistics.mean(sizes[eid] for eid in ids),
                'above_median_share': sum(sizes[eid] > median for eid in ids) / 5,
                'soft_positive_share': sum(labels['soft'][qid][eid] > 0 for eid in ids) / 5,
                'hard_positive_share': sum(labels['hard'][qid][eid] > 0 for eid in ids) / 5,
            }
        results.append({'query_id': qid, 'arms': arm_data})
    if len(results) != len(labels['soft']) or len({r['query_id'] for r in results}) != len(results):
        raise ValueError('Ranking and label cohorts differ')
    fields = tuple(results[0]['arms']['gpu'])
    report = {'queries': len(results), 'reviewers': len(reviewers), 'profile_size_median': median,
              'population_above_median_share': sum(n > median for n in sizes.values()) / len(sizes),
              'population_mean_profile_size': statistics.mean(sizes.values()),
              'population_label_rates_by_profile_group': {task: {group: statistics.mean(labels[task][q][eid] > 0 for q in labels[task] for eid in reviewers if (sizes[eid] > median) == above) for group, above in (('at_or_below_median', False), ('above_median', True))} for task in labels},
              'means': {arm: {field: statistics.mean(r['arms'][arm][field] for r in results) for field in fields} for arm in arms},
              'paired_gpu_minus_lexical': {field: paired_interval([r['arms']['gpu'][field] - r['arms']['lexical'][field] for r in results]) for field in fields},
              'input_sha256': {'rows': sha(args.rows), **{name: sha(root / name) for name in ('SIGIR_reviewers_test.json','SIGIR_queries_test_soft.json','SIGIR_queries_test_hard.json')}, 'script': sha(Path(__file__))},
              'scope': 'Post-hoc exploratory exposure analysis of reused CoF SIGIR reviewer data. Profile size counts distinct indexed papers, not career output or personal contribution. GPU is the dense max-source ranking; lexical is BM25 max-source. Query bootstrap assumes independent queries. Associations do not establish volume bias causally; reviewer relevance and corpus coverage may vary with profile size. No enterprise fit, individual contribution, fairness, or training-free correction is established.'}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'rows.json').write_text(json.dumps(results, indent=2) + '\n')
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
