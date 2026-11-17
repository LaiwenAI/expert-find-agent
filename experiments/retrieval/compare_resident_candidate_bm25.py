"""Replay full-need retrieval with SQLite and resident candidate postings."""
import argparse
import hashlib
import json
import platform
import resource
import statistics
import time
from pathlib import Path

from expert_search.candidate_evidence_pipeline import CandidateEvidencePipeline


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main(output):
    data = Path('artifacts/dataset-release/fresh-enterprise-eval-v1'); manifest = json.loads((data / 'manifest.json').read_text())
    if sha(data / 'queries.jsonl') != manifest['files_sha256']['queries.jsonl']: raise ValueError('Changed query set')
    queries = [r for r in map(json.loads, (data / 'queries.jsonl').read_text().splitlines()) if r['split'] == 'development']
    if len(queries) != 80 or len({r['id'] for r in queries}) != 80: raise ValueError('Changed cohort')
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    engine = CandidateEvidencePipeline(scoring='resident'); engine.evidence.prepare_adaptive_scoring()
    resident = engine.resident
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rows = []
    for i, q in enumerate(queries):
        arms = {}; result = {}
        for name in (('sqlite','resident') if i % 2 == 0 else ('resident','sqlite')):
            engine.resident = None if name == 'sqlite' else resident
            start = time.perf_counter(); value = engine.search_reranked(q['text'], k=20, candidates=50, evidence_per_expert=1)
            arms[name] = time.perf_counter() - start; result[name] = value['results']
        if result['sqlite'] != result['resident']: raise ValueError('Changed scores, ranking or evidence')
        rows.append({'id': q['id'], 'seconds': arms})
        if (i+1) % 20 == 0: print('Completed', i+1, flush=True)
    engine.evidence.db.close()
    report = {'queries': len(rows), 'identical_complete_results': len(rows), 'resident_build_seconds': resident.initialization_seconds,
              'resident_posting_array_bytes': resident.array_bytes, 'distinct_terms': len(resident.offsets),
              'process_peak_rss_before_build': before, 'process_peak_rss_after_build': after,
              'peak_rss_units': 'bytes on macOS; KiB on Linux', 'platform': platform.system(),
              'mean_seconds': {name: statistics.mean(r['seconds'][name] for r in rows) for name in ('sqlite','resident')},
              'resident_faster_queries': sum(r['seconds']['resident'] < r['seconds']['sqlite'] for r in rows),
              'input_sha256': {'queries': manifest['files_sha256']['queries.jsonl'], 'corpus_snapshot': engine.ranker.snapshot},
              'scope': 'Post-hoc 80 full enterprise development needs, candidate50 return20, one owned evidence item; identical complete results required. Resident index reads all BM25 postings and must be rebuilt for changed sources; array bytes exclude Python dictionaries and process memory. Peak RSS includes pipeline initialization and prior allocations and is not exact incremental index size. One alternating measurement per query excludes model reasoning, index construction and initialization. No independent relevance, semantic support or production latency claim. Query texts and per-query IDs remain private.'}
    output.mkdir(parents=True, exist_ok=False); (output / 'private-results.json').write_text(json.dumps(rows, indent=2)+'\n')
    report['results_sha256'] = sha(output / 'private-results.json'); report['code_sha256'] = {str(p): sha(p) for p in (Path(__file__),Path('src/expert_search/resident_candidate_bm25.py'),Path('src/expert_search/fast_retrieval.py'))}
    (output / 'summary.json').write_text(json.dumps(report, indent=2)+'\n'); print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True); main(p.parse_args().output)
