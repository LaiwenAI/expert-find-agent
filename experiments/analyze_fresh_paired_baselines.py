"""Compare fixed-view enterprise baselines with shared unknown-label bounds."""
import argparse
from collections import defaultdict
import hashlib
import json
import statistics
from pathlib import Path

from expert_search.evaluation_gain import actionable_gain_interval, paired_actionable_gain_interval


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(output):
    dataset = Path('artifacts/dataset-release/fresh-enterprise-eval-v1')
    baseline = Path('artifacts/fresh-evaluation-baselines-v1')
    manifest = json.loads((dataset / 'manifest.json').read_text())
    summary = json.loads((baseline / 'summary.json').read_text())
    if sha(dataset / 'manifest.json') != summary['dataset_manifest_sha256']:
        raise ValueError('Changed dataset manifest')
    for name in ('queries.jsonl', 'pools.jsonl', 'snapshots.jsonl', 'judgments.jsonl', 'dense-baseline-runs.jsonl'):
        if sha(dataset / name) != manifest['files_sha256'][name]: raise ValueError('Changed dataset member')
    queries = {r['id']: r for r in map(json.loads, (dataset / 'queries.jsonl').read_text().splitlines()) if r['split'] == 'development'}
    pools = {r['query_id']: r['candidates'] for r in map(json.loads, (dataset / 'pools.jsonl').read_text().splitlines()) if r['query_id'] in queries}
    grades = defaultdict(dict)
    for row in map(json.loads, (dataset / 'judgments.jsonl').read_text().splitlines()):
        qid, expert = row['query_id'], row['expert_id']
        if qid in queries:
            if expert in grades[qid]: raise ValueError('Duplicate judgment')
            grades[qid][expert] = row['grade']
    results = json.loads((baseline / 'evaluation.json').read_text()); rankings = defaultdict(dict)
    for row in results:
        qid, method, ranking = row['query_id'], row['method'], row['ranking']
        if qid not in queries or method in rankings[qid] or len(ranking) != len(set(ranking)) or set(ranking) != set(pools[qid]):
            raise ValueError('Invalid frozen ranking')
        if actionable_gain_interval(ranking, grades[qid], 5) != row['actionable_gain']:
            raise ValueError('Changed ranking or judgment')
        rankings[qid][method] = ranking
    if set(rankings) != set(queries) or set(pools) != set(queries) or any(set(v) != set(summary['methods']) for v in rankings.values()):
        raise ValueError('Incomplete baseline comparison')
    comparisons = (('qwen-paper-max', 'bm25-max'), ('qwen-view-max', 'bm25-max'), ('tfidf-max', 'bm25-max'))
    reports = {}
    for first, second in comparisons:
        bounds = [paired_actionable_gain_interval(rankings[qid][first], rankings[qid][second], grades[qid], 5) for qid in sorted(queries)]
        reports[first + '_minus_' + second] = {'mean_lower': statistics.mean(r['lower'] for r in bounds),
                                                'mean_upper': statistics.mean(r['upper'] for r in bounds),
                                                'mean_independent_lower': statistics.mean(r['independent_bounds'][0] for r in bounds),
                                                'mean_independent_upper': statistics.mean(r['independent_bounds'][1] for r in bounds),
                                                'cancelled_shared_unjudged': sum(r['cancelled_shared_unjudged'] for r in bounds),
                                                'definitely_positive_queries': sum(r['lower'] > 0 for r in bounds),
                                                'definitely_negative_queries': sum(r['upper'] < 0 for r in bounds)}
    report = {'queries': len(queries), 'comparisons': reports,
              'input_sha256': {'dataset_manifest': sha(dataset / 'manifest.json'), 'baseline_summary': sha(baseline / 'summary.json'),
                               'baseline_evaluation': sha(baseline / 'evaluation.json')},
              'code_sha256': {str(p): sha(p) for p in (Path(__file__), Path('src/expert_search/evaluation_gain.py'))},
              'scope': 'Post-hoc bounds on 80 fixed seven-candidate development needs with incomplete same-model silver judgments, using identical frozen expert/source views across methods. Shared unjudged view grades cancel when rank coefficients match; bounds reflect missing-label uncertainty, not confidence intervals or semantic correctness. Candidate pool is selected by dense and random retrieval, with no global recall, independent human labels, online latency or held-out claim.'}
    output.mkdir(parents=True, exist_ok=False); (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n'); print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
