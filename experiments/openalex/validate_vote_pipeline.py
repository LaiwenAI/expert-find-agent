"""Compare reusable candidate-first voting with the prior evidence-fetch path."""
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


def main(output):
    data = Path('benchmarks/openalex-author-recovery-v1'); verify(data)
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter(); index(data, output / 'source.sqlite')
    build(data, output / 'source.sqlite', output / 'lm', output / 'fast.sqlite')
    build_seconds = time.perf_counter() - start
    pipeline = CandidateEvidencePipeline(data, output / 'lm', output / 'fast.sqlite')
    pipeline.evidence.prepare_adaptive_scoring()
    queries = [json.loads(line) for line in (data / 'queries.jsonl').read_text().splitlines()]
    labels = {r['query_id']: set(r['known_author_ids']) for r in map(json.loads, (data / 'qrels.jsonl').read_text().splitlines())}
    rows = []
    for i, query in enumerate(queries):
        arms = {}
        for arm in (('prior', 'reuse') if i % 2 == 0 else ('reuse', 'prior')):
            start = time.perf_counter()
            if arm == 'prior':
                response = pipeline.search(query['text'], k=50, evidence_per_expert=5)
                candidates = sorted(response['results'], key=lambda r: (-sum(e['score'] for e in r['evidence']), r['expert']['id']))
                results = [dict(expert=r['expert'], score=sum(e['score'] for e in r['evidence']), evidence=r['evidence']) for r in candidates if any(e['score'] > 0 for e in r['evidence'])][:10]
            else:
                results = pipeline.search_reranked(query['text'], k=10, candidates=50, evidence_per_expert=5, aggregation='vote')['results']
            arms[arm] = {'seconds': time.perf_counter() - start, 'results': results}
        if arms['prior']['results'] != arms['reuse']['results']:
            raise ValueError('Changed ranking, score, or evidence')
        ranking = [r['expert']['id'] for r in arms['reuse']['results']]; truth = labels[query['id']]
        rows.append({'query_id': query['id'], 'seconds': {a: arms[a]['seconds'] for a in arms},
                     'hit': int(bool(set(ranking) & truth)), 'known_author_recall': len(set(ranking) & truth) / len(truth)})
    pipeline.evidence.db.close()
    report = {'queries': len(rows), 'identical_results': len(rows), 'build_seconds': build_seconds,
              'mean_seconds': {a: statistics.mean(r['seconds'][a] for r in rows) for a in ('prior', 'reuse')},
              'hit_at_10': sum(r['hit'] for r in rows), 'known_author_recall_at_10': statistics.mean(r['known_author_recall'] for r in rows),
              'dataset_manifest_sha256': hashlib.sha256((data / 'manifest.json').read_bytes()).hexdigest(),
              'scope': 'Paired local replay on 99 reused public author-recovery proxy queries; one rotating-order timing per path. Both return up to ten experts with positive lexical evidence; source association is not individual contribution or industrial fit. Timing excludes initialization and model reasoning. OpenAlex identities and semantic leakage are not independently verified.'}
    (output / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
    report['results_sha256'] = hashlib.sha256((output / 'results.json').read_bytes()).hexdigest()
    report['code_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), Path('src/expert_search/candidate_evidence_pipeline.py'), Path('src/expert_search/fast_retrieval.py'))}
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n'); print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
