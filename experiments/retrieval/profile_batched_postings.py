"""Compare exact global posting iteration with batched SQLite access."""
import argparse
from collections import defaultdict
import hashlib
import json
import statistics
import time
from pathlib import Path

from expert_search.candidate_evidence_pipeline import CandidateEvidencePipeline
from expert_search.retrieval import tokens


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def batched(search, query, chunk_size=400):
    terms = sorted(set(tokens(query))); scores = defaultdict(float)
    for offset in range(0, len(terms), chunk_size):
        batch = terms[offset:offset+chunk_size]; placeholders = ','.join('?' for _ in batch)
        sql = f'SELECT doc,bm25 FROM scores WHERE term IN ({placeholders}) ORDER BY term,doc'
        for doc, score in search.db.execute(sql, batch): scores[doc] += score
    return {search.metadata[doc][0]: score for doc, score in scores.items()}


def main(output):
    data = Path('artifacts/dataset-release/fresh-enterprise-eval-v1'); manifest = json.loads((data / 'manifest.json').read_text())
    if sha(data / 'queries.jsonl') != manifest['files_sha256']['queries.jsonl']: raise ValueError('Changed input')
    queries = [r for r in map(json.loads,(data / 'queries.jsonl').read_text().splitlines()) if r['split'] == 'development']
    if len(queries) != 80: raise ValueError('Changed development set')
    pipeline = CandidateEvidencePipeline(); search = pipeline.evidence; rows = []
    if search.ambiguous_evidence: raise ValueError('Ambiguous evidence IDs')
    for i, query in enumerate(queries):
        arms = {}; scores = {}
        for arm in (('one_term','batch') if i % 2 == 0 else ('batch','one_term')):
            start = time.perf_counter(); scores[arm] = search.document_scores(query['text']) if arm == 'one_term' else batched(search, query['text'])
            arms[arm] = time.perf_counter() - start
        if scores['one_term'] != scores['batch']: raise ValueError('Batched scores differ')
        rows.append({'id': query['id'], 'seconds': arms, 'scored_sources': len(scores['batch'])})
    pipeline.evidence.db.close()
    report = {'queries': len(rows), 'exact_score_maps_equal': len(rows),
              'mean_seconds': {arm: statistics.mean(r['seconds'][arm] for r in rows) for arm in ('one_term','batch')},
              'batch_faster_queries': sum(r['seconds']['batch'] < r['seconds']['one_term'] for r in rows),
              'input_sha256': {'queries': manifest['files_sha256']['queries.jsonl'], 'corpus_snapshot': pipeline.ranker.snapshot},
              'scope': 'Post-hoc 80 full enterprise development needs. Global score-map comparison only, not candidate routing, evidence loading or model reasoning. Fixed 400-term SQLite batches with ordered term accumulation; no change to ranking semantics. One alternating measurement per query; initialization, index construction and queueing excluded. Private texts and per-query IDs not published. Exact score agreement is not relevance or end-to-end accuracy.'}
    output.mkdir(parents=True, exist_ok=False); (output / 'private-results.json').write_text(json.dumps(rows, indent=2)+'\n')
    report['results_sha256'] = sha(output / 'private-results.json'); report['code_sha256'] = {str(p): sha(p) for p in (Path(__file__),Path('src/expert_search/fast_retrieval.py'),Path('src/expert_search/retrieval.py'))}
    (output / 'summary.json').write_text(json.dumps(report, indent=2)+'\n'); print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True); main(p.parse_args().output)
