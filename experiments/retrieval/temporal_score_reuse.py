"""Paired replay of evidence-score reuse on a local temporal proxy."""
import hashlib
import json
import statistics
from pathlib import Path

from expert_search.candidate_evidence_pipeline import CandidateEvidencePipeline


def main(output=Path('artifacts/local-temporal-score-reuse-v1')):
    parent = Path('artifacts/local-temporal-author-proxy-v1')
    raw = (parent / 'results.jsonl').read_bytes()
    summary = json.loads((parent / 'summary.json').read_text())
    if hashlib.sha256(raw).hexdigest() != summary['results_sha256']:
        raise ValueError('Changed query set')
    queries = [json.loads(line) for line in raw.splitlines()]
    if len(queries) != 200 or len({q['id'] for q in queries}) != 200:
        raise ValueError('Incomplete query set')
    pipeline = CandidateEvidencePipeline('artifacts/corpus-asof-2020-v1', 'artifacts/temporal-candidate-lm-2020-v1', 'artifacts/temporal-evidence-2020-v1.sqlite')
    pipeline.evidence.prepare_adaptive_scoring()
    reuse = pipeline.evidence.evidence_from_scores
    query = ''
    def prior(experts, sources, scores, n):
        return pipeline.evidence.evidence_for_selective(query, experts, n)
    timings = {'prior': [], 'reuse': []}
    evidence_times = {'prior': [], 'reuse': []}
    for i, item in enumerate(queries):
        query = item['title']
        outputs = {}
        order = ('prior', 'reuse') if i % 2 == 0 else ('reuse', 'prior')
        for arm in order:
            pipeline.evidence.evidence_from_scores = prior if arm == 'prior' else reuse
            result = pipeline.search_reranked(query, k=10, candidates=50, evidence_per_expert=3)
            outputs[arm] = result['results']
            timings[arm].append(result['timing']['total_seconds'])
            evidence_times[arm].append(result['timing']['evidence_seconds'])
        if outputs['prior'] != outputs['reuse']:
            raise ValueError('Ranking, scores or evidence differ')
    pipeline.evidence.db.close()
    report = dict(queries=len(queries), identical_full_results=len(queries),
                  mean_total_seconds={arm: statistics.mean(timings[arm]) for arm in timings},
                  mean_evidence_seconds={arm: statistics.mean(evidence_times[arm]) for arm in timings},
                  parent_sha256=summary['results_sha256'],
                  code_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), Path('src/expert_search/candidate_evidence_pipeline.py'), Path('src/expert_search/fast_retrieval.py'))},
                  scope='Post-hoc alternating paired timings on 200 previously studied historical author-recovery title queries. Both paths use identical rankings and three lexical evidence items per expert. One measurement per query; excludes initialization, index construction and model reasoning. Local corpus association and identical evidence do not establish individual contribution, semantic support or industrial relevance.')
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts/local-temporal-score-reuse-v1'))
    main(parser.parse_args().output)
