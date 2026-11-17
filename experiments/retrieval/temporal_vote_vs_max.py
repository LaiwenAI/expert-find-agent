"""Compare candidate-first max and vote on a local temporal author proxy."""
import argparse
from collections import defaultdict
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

from expert_search.candidate_evidence_pipeline import CandidateEvidencePipeline


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main(output):
    parent = Path('artifacts/local-temporal-author-proxy-v1'); history = Path('artifacts/corpus-asof-2020-v1')
    metadata = json.loads((parent / 'summary.json').read_text()); payload = (parent / 'results.jsonl').read_bytes()
    if hashlib.sha256(payload).hexdigest() != metadata['results_sha256'] or sha(history / 'evidence.jsonl') != metadata['historical_sha256']:
        raise ValueError('Changed proxy inputs')
    cases = [json.loads(line) for line in payload.splitlines()]
    if len(cases) != 200 or len({r['id'] for r in cases}) != len(cases) or any(not r['known_author_ids'] for r in cases):
        raise ValueError('Invalid cohort')
    pipeline = CandidateEvidencePipeline(history, 'artifacts/temporal-candidate-lm-2020-v1', 'artifacts/temporal-evidence-2020-v1.sqlite')
    pipeline.evidence.prepare_adaptive_scoring(); rows = []
    for i, case in enumerate(cases):
        arms = {}
        for arm in (('max','vote') if i % 2 == 0 else ('vote','max')):
            start = time.perf_counter()
            result = pipeline.search_reranked(case['title'], k=10, candidates=50, evidence_per_expert=3, aggregation=arm)
            elapsed = time.perf_counter() - start; ranking = [r['expert']['id'] for r in result['results']]
            if len(ranking) != len(set(ranking)) or any(e['expert_id'] != r['expert']['id'] for r in result['results'] for e in r['evidence']):
                raise ValueError('Invalid evidence ownership')
            truth = set(case['known_author_ids']); ranks = [i for i, eid in enumerate(ranking, 1) if eid in truth]
            arms[arm] = {'ranking': ranking, 'seconds': elapsed, 'hit': int(bool(ranks)),
                         'mrr': 1 / min(ranks) if ranks else 0., 'recall': len(set(ranking) & truth) / len(truth)}
        rows.append({'id': case['id'], 'arms': arms})
    pipeline.evidence.db.close()
    parent_ids = list(range(len(cases)))
    def find(i):
        while parent_ids[i] != i:
            parent_ids[i] = parent_ids[parent_ids[i]]; i = parent_ids[i]
        return i
    owners = {}
    for i, case in enumerate(cases):
        for author in case['known_author_ids']:
            if author in owners: parent_ids[find(i)] = find(owners[author])
            else: owners[author] = i
    grouped = defaultdict(list)
    for i in range(len(cases)): grouped[find(i)].append(i)
    groups = list(grouped.values()); rng = random.Random(20261120); replicates = 10000
    differences = {metric: [r['arms']['vote'][metric] - r['arms']['max'][metric] for r in rows] for metric in ('hit','mrr','recall')}
    draws = {metric: [] for metric in differences}
    for _ in range(replicates):
        chosen = [index for _ in groups for index in rng.choice(groups)]
        for metric in differences: draws[metric].append(sum(differences[metric][i] for i in chosen) / len(chosen))
    intervals = {}
    for metric, values in draws.items():
        values.sort(); intervals[metric] = {'mean': statistics.mean(differences[metric]), 'percentile_95': [values[249], values[9749]]}
    report = {'queries': len(rows), 'author_connected_groups': len(groups), 'largest_group': max(map(len, groups)),
              'methods': {arm: {metric: statistics.mean(r['arms'][arm][metric] for r in rows) for metric in ('hit','mrr','recall','seconds')} for arm in ('max','vote')},
              'paired_vote_minus_max': intervals, 'bootstrap_replicates': replicates, 'seed': 20261120,
              'input_sha256': {'parent_results': metadata['results_sha256'], 'history': metadata['historical_sha256']},
              'scope': 'Post-hoc 200-query local historical-author proxy, fixed 50 candidates and up to three displayed source records. Known author associations are incomplete and unverified; other candidates are unjudged. One alternating timing per query excludes index construction, initialization and model reasoning. Author-connected bootstrap does not address topic dependence or leakage; reused query cohort and intervals do not establish enterprise relevance or general latency.'}
    output.mkdir(parents=True, exist_ok=False); (output / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
    report['results_sha256'] = sha(output / 'results.json'); report['code_sha256'] = {str(p): sha(p) for p in (Path(__file__), Path('src/expert_search/candidate_evidence_pipeline.py'), Path('src/expert_search/fast_retrieval.py'))}
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n'); print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
