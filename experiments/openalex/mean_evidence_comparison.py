"""Compare matched-source mean with max and vote on the public proxy."""
import argparse
import hashlib
import json
import random
import statistics
from collections import defaultdict
import time
from pathlib import Path

from expert_search.candidate_evidence_pipeline import CandidateEvidencePipeline
from expert_search.retrieval import index
from experiments.retrieval.build_candidate_pipeline import build
from .verify_author_recovery import verify


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main(output):
    data = Path('benchmarks/openalex-author-recovery-v1'); verify(data)
    output.mkdir(parents=True, exist_ok=False); start = time.perf_counter()
    index(data, output / 'source.sqlite'); build(data, output / 'source.sqlite', output / 'lm', output / 'fast.sqlite')
    build_seconds = time.perf_counter() - start
    engine = CandidateEvidencePipeline(data, output / 'lm', output / 'fast.sqlite'); engine.evidence.prepare_adaptive_scoring()
    queries = [json.loads(line) for line in (data / 'queries.jsonl').read_text().splitlines()]
    labels = {r['query_id']: set(r['known_author_ids']) for r in map(json.loads, (data / 'qrels.jsonl').read_text().splitlines())}
    if len(queries) != 99 or any(not labels.get(r['id']) for r in queries): raise ValueError('Changed labeled cohort')
    rows = []
    for query in queries:
        start = time.perf_counter(); pool = [r['expert']['id'] for r in engine.ranker.search(query['text'], 50)['results']]
        ids = [sid for eid in pool for sid in engine.evidence_by_expert[eid]]
        scores = engine.evidence.adaptive_document_scores(query['text'], ids)['scores']; values = {}
        for eid in pool:
            hits = sorted((scores[sid] for sid in engine.evidence_by_expert[eid] if scores[sid] > 0), reverse=True)[:5]
            values[eid] = {'max': hits[0] if hits else 0., 'vote': sum(hits), 'mean': statistics.mean(hits) if hits else 0.}
        arms = {}; truth = labels[query['id']]
        for arm in ('max','vote','mean'):
            eligible = pool if arm == 'max' else [eid for eid in pool if values[eid][arm] > 0]
            ranking = sorted(eligible, key=lambda eid: (-values[eid][arm], eid))[:10]
            evidence = engine.evidence.evidence_from_scores(ranking, engine.evidence_by_expert, scores, 3)
            if any(e['expert_id'] != eid for eid in ranking for e in evidence[eid]): raise ValueError('Wrong evidence owner')
            ranks = [i for i,eid in enumerate(ranking, 1) if eid in truth]
            arms[arm] = {'ranking': ranking, 'hit': int(bool(ranks)), 'mrr': 1 / min(ranks) if ranks else 0.,
                         'recall': len(set(ranking) & truth) / len(truth), 'mean_profile_records': statistics.mean(len(engine.evidence_by_expert[eid]) for eid in ranking)}
        rows.append({'id': query['id'], 'arms': arms, 'shared_seconds': time.perf_counter() - start})
    engine.evidence.db.close()
    links = list(range(len(queries)))
    def find(i):
        while links[i] != i:
            links[i] = links[links[i]]; i = links[i]
        return i
    owners = {}
    for i, query in enumerate(queries):
        for eid in labels[query['id']]:
            if eid in owners: links[find(i)] = find(owners[eid])
            else: owners[eid] = i
    grouped = defaultdict(list)
    for i in range(len(queries)): grouped[find(i)].append(i)
    groups = list(grouped.values()); rng = random.Random(20261120)
    diffs = {metric: [r['arms']['mean'][metric]-r['arms']['max'][metric] for r in rows] for metric in ('hit','mean_profile_records')}
    draws = {metric: [] for metric in diffs}
    for _ in range(10000):
        sample = [i for _ in groups for i in rng.choice(groups)]
        for metric in diffs: draws[metric].append(sum(diffs[metric][i] for i in sample) / len(sample))
    paired = {}
    for metric, values in draws.items():
        values.sort(); paired[metric] = {'mean': statistics.mean(diffs[metric]), 'percentile_95': [values[249], values[9749]]}
    report = {'queries': len(rows), 'index_build_seconds': build_seconds,
              'methods': {arm: {metric: statistics.mean(r['arms'][arm][metric] for r in rows) for metric in ('hit','mrr','recall','mean_profile_records')} for arm in ('max','vote','mean')},
              'paired_mean_minus_max': paired, 'author_connected_groups': len(groups), 'bootstrap_replicates': 10000, 'seed': 20261120,
              'dataset_manifest_sha256': sha(data / 'manifest.json'),
              'scope': 'Post-hoc reused OpenAlex author-recovery proxy, fixed 50-candidate pool. Mean averages up to five positive matched source scores, and vote sums them; max retains zero-score fallback. Three arms share retrieval and scoring; timing does not measure individual latency. Known author associations are incomplete and machine-disambiguated; other authors are unjudged. Bootstrap groups shared known authors; topic and other dependencies remain. This is not individual contribution or enterprise-fit evaluation.'}
    (output / 'results.json').write_text(json.dumps(rows, indent=2) + '\n'); report['results_sha256'] = sha(output / 'results.json')
    report['code_sha256'] = {str(p): sha(p) for p in (Path(__file__),Path('src/expert_search/candidate_evidence_pipeline.py'),Path('src/expert_search/fast_retrieval.py'))}
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n'); print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
