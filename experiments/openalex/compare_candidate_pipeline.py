"""Compare candidate-first retrieval with lexical baselines on the frozen proxy."""
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

from expert_search.candidate_evidence_pipeline import CandidateEvidencePipeline
from expert_search.retrieval import Search, index
from expert_search.fast_retrieval import FastSearch
from experiments.retrieval.build_candidate_pipeline import build
from .verify_author_recovery import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = Path('benchmarks/openalex-author-recovery-v1')
    verify(data)
    args.output.mkdir(parents=True, exist_ok=False)
    database = args.output / 'source.sqlite'
    start = time.perf_counter()
    index(data, database)
    lexical_build = time.perf_counter() - start
    start = time.perf_counter()
    build(data, database, args.output / 'candidate-lm', args.output / 'evidence.sqlite')
    candidate_build = time.perf_counter() - start
    start = time.perf_counter()
    lexical = Search(data, database)
    lexical_init = time.perf_counter() - start
    compiled_lexical = FastSearch(data, args.output / 'evidence.sqlite')
    start = time.perf_counter()
    candidate = CandidateEvidencePipeline(data, args.output / 'candidate-lm', args.output / 'evidence.sqlite')
    candidate_init = time.perf_counter() - start
    queries = [json.loads(x) for x in (data / 'queries.jsonl').read_text().splitlines()]
    labels = {r['query_id']: set(r['known_author_ids']) for r in map(json.loads, (data / 'qrels.jsonl').read_text().splitlines())}
    methods = ('bm25-max', 'bm25-vote', 'compiled-bm25-vote', 'candidate-lm-evidence', 'candidate50-bm25-vote')
    rows = []
    for i, query in enumerate(queries):
        results = {}
        for method in methods[i % len(methods):] + methods[:i % len(methods)]:
            start = time.perf_counter()
            if method == 'candidate-lm-evidence':
                response = candidate.search(query['text'], k=10, evidence_per_expert=3)
                ids = [r['expert']['id'] for r in response['results']]
            elif method == 'candidate50-bm25-vote':
                response = candidate.search(query['text'], k=50, evidence_per_expert=5)
                ranked = sorted(response['results'], key=lambda r: (-sum(e['score'] for e in r['evidence']), r['expert']['id']))
                ids = [r['expert']['id'] for r in ranked if any(e['score'] > 0 for e in r['evidence'])][:10]
            elif method == 'compiled-bm25-vote':
                ids = [r['id'] for r in compiled_lexical.search(query['text'], 10, 'bm25-vote')]
            else:
                ids = [r['id'] for r in lexical.search(query['text'], 10, method)]
            elapsed = time.perf_counter() - start
            truth = labels[query['id']]
            ranks = [rank for rank, expert in enumerate(ids, 1) if expert in truth]
            results[method] = {'ranking': ids, 'seconds': elapsed, 'returned': len(ids), 'hit_at_10': int(bool(ranks)),
                               'mrr_at_10': 1 / min(ranks) if ranks else 0,
                               'known_author_recall_at_10': len(set(ids) & truth) / len(truth)}
        rows.append({'query_id': query['id'], 'methods': results})
    metrics = ('seconds', 'returned', 'hit_at_10', 'mrr_at_10', 'known_author_recall_at_10')
    summary = {'queries': len(rows),
               'compiled_vote_identical_rankings': sum(r['methods']['compiled-bm25-vote']['ranking'] == r['methods']['bm25-vote']['ranking'] for r in rows),
               'lexical_build_seconds': lexical_build,
               'candidate_additional_build_seconds': candidate_build,
               'lexical_initialization_seconds': lexical_init, 'candidate_initialization_seconds': candidate_init,
               'methods': {m: {key: statistics.mean(r['methods'][m][key] for r in rows) for key in metrics} for m in methods},
               'dataset_manifest_sha256': hashlib.sha256((data / 'manifest.json').read_bytes()).hexdigest(),
               'scope': 'OpenAlex temporal author-recovery proxy, not independently judged enterprise expertise. Non-authors are unjudged. One rotating-order timing per method and query; initialization and index builds are reported separately. No model reasoning. The threshold was chosen after inspecting a preliminary snapshot; semantic leakage and author disambiguation remain unaudited.'}
    (args.output / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    lexical.db.close()
    candidate.evidence.db.close()
    compiled_lexical.db.close()


if __name__ == '__main__':
    main()
