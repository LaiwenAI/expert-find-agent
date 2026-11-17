"""Compare candidate-filtered and full global posting accumulation."""
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

from expert_search.candidate_evidence_pipeline import CandidateEvidencePipeline


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main(output):
    data = Path('artifacts/dataset-release/fresh-enterprise-eval-v1'); manifest = json.loads((data / 'manifest.json').read_text())
    if sha(data / 'queries.jsonl') != manifest['files_sha256']['queries.jsonl']: raise ValueError('Changed query set')
    queries = [r for r in map(json.loads, (data / 'queries.jsonl').read_text().splitlines()) if r['split'] == 'development']
    if len(queries) != 80 or len({r['id'] for r in queries}) != 80: raise ValueError('Invalid development cohort')
    engine = CandidateEvidencePipeline(); engine.evidence.prepare_adaptive_scoring(); optimized = engine.evidence.filtered_global_document_scores
    def prior(query, ids):
        full = engine.evidence.document_scores(query); return {eid: full.get(eid,0.) for eid in ids}
    rows = []
    for i,q in enumerate(queries):
        arms = {}; results = {}
        for arm in (('prior','filtered') if i % 2 == 0 else ('filtered','prior')):
            engine.evidence.filtered_global_document_scores = prior if arm == 'prior' else optimized
            start = time.perf_counter(); result = engine.search_reranked(q['text'], k=20, candidates=50, evidence_per_expert=1)
            arms[arm] = {'seconds': time.perf_counter()-start, 'route': result['route']}; results[arm] = result['results']
        if results['prior'] != results['filtered'] or arms['prior']['route'] != arms['filtered']['route']:
            raise ValueError('Changed rankings, scores, evidence or route')
        rows.append({'id': q['id'], 'arms': arms})
    engine.evidence.db.close()
    global_rows = [r for r in rows if r['arms']['prior']['route'] == 'global']
    def aggregate(values):
        return {'queries': len(values), 'mean_seconds': {a: statistics.mean(r['arms'][a]['seconds'] for r in values) for a in ('prior','filtered')},
                'filtered_faster': sum(r['arms']['filtered']['seconds'] < r['arms']['prior']['seconds'] for r in values)} if values else None
    report = {'queries': len(rows), 'identical_complete_results': len(rows), 'all': aggregate(rows), 'global_route': aggregate(global_rows),
              'input_sha256': {'queries': manifest['files_sha256']['queries.jsonl'], 'corpus_snapshot': engine.ranker.snapshot},
              'scope': 'Post-hoc local 80 full enterprise development needs; candidate50, return20 and one evidence item per candidate. Fixed route heuristic chooses global or selective; only the global branch changes. One alternating paired timing per query includes retrieval and evidence materialization but excludes initialization, posting-count preparation, indexing and model reasoning. No independent relevance labels or production latency distribution; source ownership is not semantic support or personal contribution.'}
    output.mkdir(parents=True, exist_ok=False); (output / 'private-results.json').write_text(json.dumps(rows, indent=2)+'\n')
    report['results_sha256'] = sha(output / 'private-results.json'); report['code_sha256'] = {str(p): sha(p) for p in (Path(__file__), Path('src/expert_search/fast_retrieval.py'), Path('src/expert_search/candidate_evidence_pipeline.py'))}
    (output / 'summary.json').write_text(json.dumps(report, indent=2)+'\n'); print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True); main(p.parse_args().output)
