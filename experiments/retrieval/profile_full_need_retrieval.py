"""Profile candidate-first retrieval components on enterprise development needs."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

from expert_search.candidate_evidence_pipeline import CandidateEvidencePipeline
from expert_search.retrieval import tokens


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def quantile(values, probability):
    values = sorted(values); return values[round((len(values)-1)*probability)]


def main(output):
    data = Path('artifacts/dataset-release/fresh-enterprise-eval-v1'); manifest = json.loads((data / 'manifest.json').read_text())
    if sha(data / 'queries.jsonl') != manifest['files_sha256']['queries.jsonl']: raise ValueError('Changed queries')
    queries = [r for r in map(json.loads,(data / 'queries.jsonl').read_text().splitlines()) if r['split'] == 'development']
    if len(queries) != 80 or len({r['id'] for r in queries}) != 80: raise ValueError('Changed cohort')
    pipeline = CandidateEvidencePipeline(); pipeline.evidence.prepare_adaptive_scoring(); rows = []
    for q in queries:
        result = pipeline.search_reranked(q['text'], k=20, candidates=50, evidence_per_expert=1)
        rows.append({'id': q['id'], 'query_token_count': len(tokens(q['text'])), 'distinct_query_tokens': len(set(tokens(q['text']))),
                     'route': result['route'], 'estimated_global_postings': result['estimated_global_postings'],
                     'estimated_selective_probes': result['estimated_selective_probes'],
                     'timing': {k: result['timing'][k] for k in ('ranking_seconds','scoring_seconds','evidence_seconds','total_seconds')}})
    pipeline.evidence.db.close()
    def stats(values): return {'mean': statistics.mean(values), 'p50': quantile(values,.5), 'p95': quantile(values,.95), 'max': max(values)}
    report = {'queries': len(rows), 'candidate_pool': 50, 'returned': 20, 'evidence_per_expert': 1,
              'routes': {name: sum(r['route']==name for r in rows) for name in ('global','selective')},
              'seconds': {name: stats([r['timing'][name] for r in rows]) for name in ('ranking_seconds','scoring_seconds','evidence_seconds','total_seconds')},
              'tokens': stats([r['query_token_count'] for r in rows]), 'distinct_tokens': stats([r['distinct_query_tokens'] for r in rows]),
              'input_sha256': {'queries': manifest['files_sha256']['queries.jsonl'], 'corpus_snapshot': pipeline.ranker.snapshot},
              'scope': 'Post-hoc local fixed 80 development needs, one sequential measurement each with warmed posting-count cache. Quantiles are empirical order statistics, not production latency percentiles. Component times are nested and do not include initialization, posting-count preparation, index build, model reasoning or queueing. Route estimates are heuristics, not measured operations; query and expert identities remain private.'}
    output.mkdir(parents=True, exist_ok=False); (output / 'private-results.json').write_text(json.dumps(rows, indent=2)+'\n')
    report['results_sha256'] = sha(output / 'private-results.json'); report['code_sha256'] = {str(p): sha(p) for p in (Path(__file__),Path('src/expert_search/candidate_evidence_pipeline.py'),Path('src/expert_search/fast_retrieval.py'),Path('src/expert_search/certified_topk.py'))}
    (output / 'summary.json').write_text(json.dumps(report, indent=2)+'\n'); print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True); main(p.parse_args().output)
