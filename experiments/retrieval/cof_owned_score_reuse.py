"""Compare evidence-score reuse on a fully judged public reviewer task."""
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

from expert_search.candidate_evidence_pipeline import CandidateEvidencePipeline
from expert_search.cof_benchmark import load, precision_metrics
from expert_search.retrieval import index
from experiments.retrieval.build_candidate_pipeline import build


def main(output):
    root, papers, reviewers, labels = load('SIGIR')
    inputs = ('SIGIR_papers_test.json', 'SIGIR_reviewers_test.json', 'SIGIR_queries_test_soft.json', 'SIGIR_queries_test_hard.json')
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in inputs}
    if set(labels['soft']) != set(labels['hard']) or any(set(labels[t][q]) != set(reviewers) for t in labels for q in labels[t]):
        raise ValueError('Incomplete relevance judgments')
    output.mkdir(parents=True, exist_ok=False)
    corpus = output / 'corpus'; corpus.mkdir()
    (corpus / 'experts.jsonl').write_text(''.join(json.dumps({'id': eid}) + '\n' for eid in sorted(reviewers)))
    evidence = []
    for eid, reviewer in sorted(reviewers.items()):
        for pid in sorted(set(reviewer['papers'])):
            paper = papers[pid]
            evidence.append({'id': json.dumps([eid, pid], separators=(',', ':')), 'expert_id': eid, 'paper_id': pid,
                             'text': (paper.get('title', '') + '\n' + paper.get('abstract', '')).strip()})
    (corpus / 'evidence.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in evidence))
    start = time.perf_counter()
    index(corpus, output / 'source.sqlite')
    build(corpus, output / 'source.sqlite', output / 'lm', output / 'fast.sqlite')
    build_seconds = time.perf_counter() - start
    pipeline = CandidateEvidencePipeline(corpus, output / 'lm', output / 'fast.sqlite')
    pipeline.evidence.prepare_adaptive_scoring()
    reuse = pipeline.evidence.evidence_from_scores
    query = ''
    def prior(experts, sources, scores, n):
        return pipeline.evidence.evidence_for_selective(query, experts, n)
    rows = []
    for i, qid in enumerate(sorted(labels['soft'])):
        paper = papers[qid]; query = (paper.get('title', '') + '\n' + paper.get('abstract', '')).strip()
        values = {}
        for arm in (('prior', 'reuse') if i % 2 == 0 else ('reuse', 'prior')):
            pipeline.evidence.evidence_from_scores = prior if arm == 'prior' else reuse
            values[arm] = pipeline.search_reranked(query, k=10, candidates=50, evidence_per_expert=3)
        if values['prior']['results'] != values['reuse']['results']:
            raise ValueError('Result mismatch')
        ranking = [r['expert']['id'] for r in values['reuse']['results']]
        rows.append({'query_id': qid, 'seconds': {arm: values[arm]['timing']['total_seconds'] for arm in values},
                     'evidence_seconds': {arm: values[arm]['timing']['evidence_seconds'] for arm in values},
                     'metrics': {task: precision_metrics(ranking, labels[task][qid]) for task in labels}})
    pipeline.evidence.db.close()
    report = {'queries': len(rows), 'reviewers': len(reviewers), 'owned_sources': len(evidence), 'identical_complete_results': len(rows),
              'index_build_seconds': build_seconds,
              'mean_total_seconds': {arm: statistics.mean(r['seconds'][arm] for r in rows) for arm in ('prior', 'reuse')},
              'mean_evidence_seconds': {arm: statistics.mean(r['evidence_seconds'][arm] for r in rows) for arm in ('prior', 'reuse')},
              'metrics': {task: {m: statistics.mean(r['metrics'][task][m] for r in rows) for m in ('p@5', 'ndcg@10')} for task in labels},
              'input_sha256': hashes,
              'scope': 'Exploratory reused CoF SIGIR reviewer test set, not industrial fit. Per-reviewer source copies change corpus document frequencies from unique-paper baselines; no cross-corpus ranking or speed comparison is valid. One alternating timing per query excludes initialization, index build and reasoning; identity, contribution and semantic support are not independently verified.'}
    (output / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
    report['results_sha256'] = hashlib.sha256((output / 'results.json').read_bytes()).hexdigest()
    report['code_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), Path('src/expert_search/candidate_evidence_pipeline.py'), Path('src/expert_search/fast_retrieval.py'))}
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
