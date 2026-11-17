"""Attribute known-author recovery misses to shortlist or reranking stages."""
import argparse
import hashlib
import json
from pathlib import Path
from expert_search.candidate_evidence_pipeline import CandidateEvidencePipeline
from expert_search.fast_retrieval import FastSearch
from .verify_author_recovery import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='Output directory from compare_candidate_pipeline')
    args = parser.parse_args()
    data = Path('benchmarks/openalex-author-recovery-v1')
    verify(data)
    summary = json.loads((args.run / 'summary.json').read_text())
    manifest_hash = hashlib.sha256((data / 'manifest.json').read_bytes()).hexdigest()
    if summary['dataset_manifest_sha256'] != manifest_hash:
        raise ValueError('Dataset differs from comparison')
    results = {r['query_id']: r for r in json.loads((args.run / 'results.json').read_text())}
    queries = [json.loads(x) for x in (data / 'queries.jsonl').read_text().splitlines()]
    labels = {r['query_id']: set(r['known_author_ids']) for r in map(json.loads, (data / 'qrels.jsonl').read_text().splitlines())}
    if set(results) != {q['id'] for q in queries}:
        raise ValueError('Comparison query IDs differ')
    engine = CandidateEvidencePipeline(data, args.run / 'candidate-lm', args.run / 'evidence.sqlite')
    baseline_engine = FastSearch(data, args.run / 'evidence.sqlite')
    counts = {'queries': len(queries), 'baseline_hit': 0, 'shortlist_hit': 0, 'rerank_hit': 0,
              'baseline_hit_shortlist_miss': 0, 'baseline_hit_shortlist_hit_rerank_miss': 0,
              'baseline_miss_rerank_hit': 0, 'known_positive_links': 0, 'shortlisted_known_links': 0}
    for query in queries:
        qid = query['id']; truth = labels[qid]
        pool = engine.search(query['text'], k=50, evidence_per_expert=5)['results']
        shortlist = [r['expert']['id'] for r in pool]
        ranking = [r['expert']['id'] for r in sorted(pool, key=lambda r: (-sum(e['score'] for e in r['evidence']), r['expert']['id'])) if any(e['score'] > 0 for e in r['evidence'])][:10]
        previous = results[qid]['methods']
        if ranking != previous['candidate50-bm25-vote']['ranking']:
            raise ValueError('Cannot reproduce recorded reranking')
        baseline_rank = [r['id'] for r in baseline_engine.search(query['text'], 10, 'bm25-vote')]
        if baseline_rank != previous['compiled-bm25-vote']['ranking']:
            raise ValueError('Cannot reproduce recorded baseline')
        baseline = bool(set(baseline_rank) & truth)
        pooled = bool(set(shortlist) & truth); hit = bool(set(ranking) & truth)
        counts['baseline_hit'] += baseline; counts['shortlist_hit'] += pooled; counts['rerank_hit'] += hit
        counts['baseline_hit_shortlist_miss'] += baseline and not pooled
        counts['baseline_hit_shortlist_hit_rerank_miss'] += baseline and pooled and not hit
        counts['baseline_miss_rerank_hit'] += not baseline and hit
        counts['known_positive_links'] += len(truth); counts['shortlisted_known_links'] += len(set(shortlist) & truth)
    counts['scope'] = 'Post-hoc diagnostic on reused author-recovery queries. A known author missing from top 50 cannot be recovered by this reranker; non-authors are unjudged. Baseline-hit attribution uses any known author, not necessarily the same author across stages.'
    engine.evidence.db.close()
    baseline_engine.db.close()
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    main()
