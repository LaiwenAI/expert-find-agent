"""Explore matched-source mean versus max and vote in one candidate pool."""
import argparse
import hashlib
import json
import random
import statistics
from collections import defaultdict
import time
from pathlib import Path

from expert_search.candidate_evidence_pipeline import CandidateEvidencePipeline


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main(output):
    parent = Path('artifacts/local-temporal-author-proxy-v1'); history = Path('artifacts/corpus-asof-2020-v1'); previous = Path('artifacts/temporal-vote-vs-max-v1')
    meta = json.loads((parent / 'summary.json').read_text()); prior = json.loads((previous / 'summary.json').read_text())
    if sha(parent / 'results.jsonl') != meta['results_sha256'] or sha(history / 'evidence.jsonl') != meta['historical_sha256'] or sha(previous / 'results.json') != prior['results_sha256']:
        raise ValueError('Changed frozen inputs')
    cases = [json.loads(line) for line in (parent / 'results.jsonl').read_text().splitlines()]; prior_rows = json.loads((previous / 'results.json').read_text())
    if len(cases) != 200 or [r['id'] for r in cases] != [r['id'] for r in prior_rows]: raise ValueError('Changed cohort')
    pipeline = CandidateEvidencePipeline(history, 'artifacts/temporal-candidate-lm-2020-v1', 'artifacts/temporal-evidence-2020-v1.sqlite')
    pipeline.evidence.prepare_adaptive_scoring(); rows = []
    for case, old in zip(cases, prior_rows):
        start = time.perf_counter(); pool = [r['expert']['id'] for r in pipeline.ranker.search(case['title'], 50)['results']]
        source_ids = [sid for eid in pool for sid in pipeline.evidence_by_expert[eid]]
        scores = pipeline.evidence.adaptive_document_scores(case['title'], source_ids)['scores']
        values = {}; arms = {}; truth = set(case['known_author_ids'])
        for eid in pool:
            hits = sorted((scores[sid] for sid in pipeline.evidence_by_expert[eid] if scores[sid] > 0), reverse=True)[:5]
            values[eid] = {'max': hits[0] if hits else 0., 'vote': sum(hits), 'mean': statistics.mean(hits) if hits else 0.}
        for arm in ('max', 'vote', 'mean'):
            eligible = pool if arm == 'max' else [eid for eid in pool if values[eid][arm] > 0]
            ranking = sorted(eligible, key=lambda eid: (-values[eid][arm], eid))[:10]
            if arm in ('max', 'vote') and ranking != old['arms'][arm]['ranking']: raise ValueError('Changed reference ranking')
            evidence = pipeline.evidence.evidence_from_scores(ranking, pipeline.evidence_by_expert, scores, 3)
            if any(e['expert_id'] != eid for eid in ranking for e in evidence[eid]): raise ValueError('Foreign source')
            ranks = [i for i, eid in enumerate(ranking, 1) if eid in truth]
            arms[arm] = {'ranking': ranking, 'hit': int(bool(ranks)), 'mrr': 1 / min(ranks) if ranks else 0.,
                         'known_recall': len(set(ranking) & truth) / len(truth),
                         'mean_profile_records': statistics.mean(len(pipeline.evidence_by_expert[eid]) for eid in ranking)}
        rows.append({'id': case['id'], 'arms': arms, 'shared_all_arms_seconds': time.perf_counter() - start})
    pipeline.evidence.db.close()
    links = list(range(len(cases)))
    def find(i):
        while links[i] != i:
            links[i] = links[links[i]]; i = links[i]
        return i
    owners = {}
    for i, case in enumerate(cases):
        for eid in case['known_author_ids']:
            if eid in owners: links[find(i)] = find(owners[eid])
            else: owners[eid] = i
    grouped = defaultdict(list)
    for i in range(len(cases)): grouped[find(i)].append(i)
    groups = list(grouped.values()); rng = random.Random(20261120)
    diffs = {metric: [r['arms']['mean'][metric] - r['arms']['max'][metric] for r in rows] for metric in ('hit','mean_profile_records')}
    draws = {metric: [] for metric in diffs}
    for _ in range(10000):
        sample = [i for _ in groups for i in rng.choice(groups)]
        for metric in diffs: draws[metric].append(sum(diffs[metric][i] for i in sample) / len(sample))
    paired = {}
    for metric, values in draws.items():
        values.sort(); paired[metric] = {'mean': statistics.mean(diffs[metric]), 'percentile_95': [values[249], values[9749]]}
    report = {'queries': len(rows), 'matches_previous_max_and_vote': len(rows),
              'methods': {arm: {metric: statistics.mean(r['arms'][arm][metric] for r in rows) for metric in ('hit','mrr','known_recall','mean_profile_records')} for arm in ('max','vote','mean')},
              'mean_shared_all_arms_seconds': statistics.mean(r['shared_all_arms_seconds'] for r in rows),
              'paired_mean_minus_max': paired, 'author_connected_groups': len(groups), 'bootstrap_replicates': 10000, 'seed': 20261120,
              'input_sha256': {'parent': meta['results_sha256'], 'history': meta['historical_sha256'], 'prior_results': prior['results_sha256']},
              'scope': 'Post-hoc matched-positive top-five mean, max and sum within the same candidate-50 pool on 200 repeatedly studied temporal author queries. Mean divides by the count of positive matched records, not all available records; zero-match experts are omitted for mean and vote. All arms share retrieval and source scoring and are measured together; no per-arm or end-to-end latency estimate. Unverified known-author associations, incomplete judgments, semantic leakage and repeated experimentation preclude industrial accuracy or causal claims. Author-cluster bootstrap does not address topic dependence; comparisons are post-hoc and unadjusted. Mean is a heuristic, not attribution evidence.'}
    output.mkdir(parents=True, exist_ok=False); (output / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
    report['results_sha256'] = sha(output / 'results.json'); report['code_sha256'] = {str(p): sha(p) for p in (Path(__file__), Path('src/expert_search/candidate_evidence_pipeline.py'), Path('src/expert_search/fast_retrieval.py'))}
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n'); print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
