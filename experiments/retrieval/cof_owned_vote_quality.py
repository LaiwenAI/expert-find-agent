"""Evaluate candidate-first max and vote against global lexical ranking."""
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

from expert_search.candidate_evidence_pipeline import CandidateEvidencePipeline
from expert_search.cof_benchmark import load, precision_metrics
from expert_search.retrieval import index
from expert_search.statistics import paired_interval
from experiments.retrieval.build_candidate_pipeline import build


def main(output):
    root, papers, reviewers, labels = load('SIGIR')
    if set(labels['soft']) != set(labels['hard']) or any(set(labels[t][q]) != set(reviewers) for t in labels for q in labels[t]):
        raise ValueError('Incomplete reviewer judgments')
    output.mkdir(parents=True, exist_ok=False); corpus = output / 'corpus'; corpus.mkdir()
    (corpus / 'experts.jsonl').write_text(''.join(json.dumps({'id': e}) + '\n' for e in sorted(reviewers)))
    sources = []
    for expert, record in sorted(reviewers.items()):
        for pid in sorted(set(record['papers'])):
            p = papers[pid]
            sources.append({'id': json.dumps([expert, pid], separators=(',', ':')), 'expert_id': expert, 'paper_id': pid,
                            'text': (p.get('title', '') + '\n' + p.get('abstract', '')).strip()})
    (corpus / 'evidence.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in sources))
    start = time.perf_counter(); index(corpus, output / 'source.sqlite')
    build(corpus, output / 'source.sqlite', output / 'lm', output / 'fast.sqlite')
    build_seconds = time.perf_counter() - start
    engine = CandidateEvidencePipeline(corpus, output / 'lm', output / 'fast.sqlite')
    engine.evidence.prepare_adaptive_scoring()
    methods = ('candidate_max', 'candidate_vote', 'global_max', 'global_vote'); rows = []
    for i, qid in enumerate(sorted(labels['soft'])):
        paper = papers[qid]; query = (paper.get('title', '') + '\n' + paper.get('abstract', '')).strip(); arms = {}
        for method in methods[i % 4:] + methods[:i % 4]:
            start = time.perf_counter()
            if method.startswith('candidate'):
                response = engine.search_reranked(query, k=10, candidates=50, evidence_per_expert=3, aggregation=method.split('_')[1])
                results = response['results']; ranking = [r['expert']['id'] for r in results]
                if any(e['expert_id'] != r['expert']['id'] for r in results for e in r['evidence']): raise ValueError('Foreign evidence')
            else:
                results = engine.evidence.search(query, k=10, method='bm25-' + method.split('_')[1])
                ranking = [r['id'] for r in results]
                if any(e['expert_id'] != r['id'] for r in results for e in r['evidence']): raise ValueError('Foreign evidence')
            seconds = time.perf_counter() - start
            arms[method] = {'ranking': ranking, 'seconds': seconds,
                            'metrics': {task: precision_metrics(ranking, labels[task][qid]) for task in labels}}
        rows.append({'query_id': qid, 'arms': arms})
    engine.evidence.db.close()
    report = {'queries': len(rows), 'sources': len(sources), 'reviewers': len(reviewers), 'index_build_seconds': build_seconds,
              'methods': {m: {'mean_seconds': statistics.mean(r['arms'][m]['seconds'] for r in rows),
                              'metrics': {t: {k: statistics.mean(r['arms'][m]['metrics'][t][k] for r in rows) for k in ('p@5', 'ndcg@10')} for t in labels}} for m in methods}}
    report['paired'] = {f'{a}_minus_{b}': {t: {k: paired_interval([r['arms'][a]['metrics'][t][k] - r['arms'][b]['metrics'][t][k] for r in rows]) for k in ('p@5', 'ndcg@10')} for t in labels} for a,b in (('candidate_vote','candidate_max'), ('candidate_vote','global_vote'))}
    report['input_sha256'] = {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in ('SIGIR_papers_test.json','SIGIR_reviewers_test.json','SIGIR_queries_test_soft.json','SIGIR_queries_test_hard.json')}
    report['scope'] = 'Exploratory reused CoF SIGIR test labels; fully judged reviewer matching is not enterprise expert fit. Owner-specific copies change document frequencies relative to unique-paper baselines. Candidate methods consider 50 LM candidates; global methods visit all sources. Each returns up to three source records per expert. One rotating-order timing per query excludes initialization, index construction and reasoning. Paired query bootstrap assumes independent queries; no multiple-comparison correction or held-out policy claim.'
    (output / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
    report['results_sha256'] = hashlib.sha256((output / 'results.json').read_bytes()).hexdigest()
    report['code_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), Path('src/expert_search/candidate_evidence_pipeline.py'), Path('src/expert_search/fast_retrieval.py'))}
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n'); print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
