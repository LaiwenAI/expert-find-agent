"""Audit prompt-prefix truncation on enterprise development retrievals."""
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

from expert_search.candidate_evidence_pipeline import CandidateEvidencePipeline
from expert_search.retrieval import tokens


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main(output):
    data = Path('artifacts/dataset-release/fresh-enterprise-eval-v1'); manifest = json.loads((data / 'manifest.json').read_text())
    if sha(data / 'queries.jsonl') != manifest['files_sha256']['queries.jsonl']: raise ValueError('Changed queries')
    queries = [r for r in map(json.loads, (data / 'queries.jsonl').read_text().splitlines()) if r['split'] == 'development']
    if len(queries) != 80 or len({r['id'] for r in queries}) != 80: raise ValueError('Changed development cohort')
    pipeline = CandidateEvidencePipeline(); pipeline.evidence.prepare_adaptive_scoring(); measures = []
    for i, query in enumerate(queries):
        wanted = set(tokens(query['text'])); arms = {}
        for arm in (('lm','adaptive_bm25') if i % 2 == 0 else ('adaptive_bm25','lm')):
            start = time.perf_counter()
            if arm == 'lm': result = pipeline.search(query['text'], k=20, evidence_per_expert=1)
            else: result = pipeline.search_reranked(query['text'], k=20, candidates=50, evidence_per_expert=1)
            elapsed = time.perf_counter() - start; candidates = result['results']; source = [e for r in candidates for e in r['evidence']]
            if any(e['expert_id'] != r['expert']['id'] for r in candidates for e in r['evidence']): raise ValueError('Foreign evidence')
            truncated = 0; affected = 0; lost_total = 0; full_chars = visible_chars = 0
            for e in source:
                text = e['text']; prefix = text[:6000]; full_terms = wanted & set(tokens(text)); prefix_terms = wanted & set(tokens(prefix))
                lost = full_terms - prefix_terms
                truncated += len(text) > 6000; affected += bool(lost); lost_total += len(lost)
                full_chars += len(text); visible_chars += len(prefix)
            arms[arm] = {'sources': len(source), 'truncated_sources': truncated, 'sources_with_lost_query_terms': affected,
                         'lost_distinct_source_term_occurrences': lost_total, 'full_characters': full_chars,
                         'visible_characters': visible_chars, 'retrieval_seconds': elapsed}
        measures.append(arms)
    pipeline.evidence.db.close()
    report = {'queries': len(queries), 'limit_characters': 6000, 'arms': {}}
    for arm in ('lm','adaptive_bm25'):
        values = [r[arm] for r in measures]
        report['arms'][arm] = {key: sum(r[key] for r in values) for key in ('sources','truncated_sources','sources_with_lost_query_terms','lost_distinct_source_term_occurrences','full_characters','visible_characters')}
        report['arms'][arm]['mean_retrieval_seconds'] = statistics.mean(r['retrieval_seconds'] for r in values)
    report['input_sha256'] = {'queries': manifest['files_sha256']['queries.jsonl'], 'corpus_snapshot': pipeline.ranker.snapshot}
    report['scope'] = 'Post-hoc local 80 development needs, fixed candidate20 and one evidence record per candidate; adaptive path uses candidate50. Counts unique token types per source, not omitted facts, relevance or semantic support; full-source and prefix tokenization can differ at the truncation boundary. No model call or independent annotation. Timing excludes initialization, model reasoning and index construction; one alternating measurement per query. Query and source text remain private.'
    output.mkdir(parents=True, exist_ok=False); report['code_sha256'] = {str(p): sha(p) for p in (Path(__file__),Path('src/expert_search/candidate_evidence_pipeline.py'),Path('src/expert_search/retrieval.py'))}
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n'); print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, required=True); main(parser.parse_args().output)
