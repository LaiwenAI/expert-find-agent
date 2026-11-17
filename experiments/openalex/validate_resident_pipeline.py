"""Validate optional resident candidate scoring on the public proxy."""
import argparse
import hashlib
import json
import statistics
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
    engine = CandidateEvidencePipeline(data, output / 'lm', output / 'fast.sqlite', scoring='resident')
    resident = engine.resident; engine.evidence.prepare_adaptive_scoring()
    queries = [json.loads(line) for line in (data / 'queries.jsonl').read_text().splitlines()]; rows = []
    for i, query in enumerate(queries):
        comparisons = {}
        for mode in ('max','vote'):
            results = {}; times = {}
            for arm in (('sqlite','resident') if i % 2 == 0 else ('resident','sqlite')):
                engine.resident = None if arm == 'sqlite' else resident
                start = time.perf_counter(); result = engine.search_reranked(query['text'],k=10,candidates=50,evidence_per_expert=3,aggregation=mode)
                times[arm] = time.perf_counter() - start; results[arm] = result['results']
            if results['sqlite'] != results['resident']: raise ValueError('Changed ranking, scores or evidence')
            comparisons[mode] = times
        rows.append({'id': query['id'], 'seconds': comparisons})
    engine.evidence.db.close()
    report = {'queries': len(rows), 'matching_max_and_vote': len(rows), 'sqlite_index_build_seconds': build_seconds,
              'resident_build_seconds': resident.initialization_seconds, 'resident_array_bytes': resident.array_bytes,
              'mean_seconds': {mode: {arm: statistics.mean(r['seconds'][mode][arm] for r in rows) for arm in ('sqlite','resident')} for mode in ('max','vote')},
              'dataset_manifest_sha256': sha(data / 'manifest.json'),
              'scope': 'Reused public OpenAlex author-recovery proxy. Identical scores, rankings and evidence do not establish relevance or personal contribution. One alternating timing per query/method excludes indexing, initialization and model reasoning; resident arrays exclude dictionary and process overhead. Small corpus does not establish general scalability.'}
    (output / 'results.json').write_text(json.dumps(rows, indent=2)+'\n'); report['results_sha256'] = sha(output / 'results.json')
    report['code_sha256'] = {str(p): sha(p) for p in (Path(__file__),Path('src/expert_search/resident_candidate_bm25.py'),Path('src/expert_search/candidate_evidence_pipeline.py'))}
    (output / 'summary.json').write_text(json.dumps(report, indent=2)+'\n'); print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True); main(p.parse_args().output)
