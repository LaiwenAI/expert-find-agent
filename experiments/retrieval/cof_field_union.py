"""Explore a fixed title/abstract candidate union on CoF SIGIR."""
import hashlib
import json
import statistics
import time
from pathlib import Path

from expert_search.certified_topk import certified_topk
from expert_search.cof_benchmark import load, precision_metrics
from expert_search.compiled_language_model import CompiledCandidateLM
from expert_search.expert_language_models import ExpertLanguageModels
from expert_search.fast_retrieval import FastSearch
from expert_search.statistics import paired_interval


def main(output):
    root, papers, reviewers, labels = load('SIGIR')
    sources = sorted({pid for reviewer in reviewers.values() for pid in reviewer['papers']})
    documents = {pid: (papers[pid].get('title', '') + '\n' + papers[pid].get('abstract', '')).strip() for pid in sources}
    profiles = {eid: sorted(set(row['papers'])) for eid, row in reviewers.items()}
    model = CompiledCandidateLM(ExpertLanguageModels(documents, profiles)); available = set(model.available)
    if available != set(reviewers) or set(labels['soft']) != set(labels['hard']) or any(set(labels[t][q]) != available for t in labels for q in labels[t]):
        raise ValueError('Incomplete reviewer labels')
    search = FastSearch(root / 'unique-profile-corpus', Path('artifacts/cof-sigir-selective-rerank-v1/selective.sqlite'))
    search.prepare_adaptive_scoring()
    methods = ('full20', 'field_union20', 'full50', 'global')
    rows = []
    for i, qid in enumerate(sorted(labels['soft'])):
        paper = papers[qid]; title = paper.get('title', '').strip(); abstract = paper.get('abstract', '').strip()
        query = (title + '\n' + abstract).strip(); arms = {}
        for method in methods[i % 4:] + methods[:i % 4]:
            start = time.perf_counter()
            if method == 'global':
                pool = sorted(available); values = search.document_scores(query)
            else:
                depth = 50 if method == 'full50' else 20
                full = [r['id'] for r in certified_topk(model, query, depth)['results']]
                if method == 'field_union20':
                    pool = []
                    for text in (title, abstract):
                        if text:
                            pool.extend(r['id'] for r in certified_topk(model, text, 10)['results'])
                    pool = list(dict.fromkeys(pool + full))[:20]
                else:
                    pool = full
                ids = sorted({pid for eid in pool for pid in profiles[eid]})
                values = search.adaptive_document_scores(query, ids)['scores']
            scores = {eid: max((values.get(pid, 0.) for pid in profiles[eid]), default=0.) for eid in pool}
            ranking = sorted(pool, key=lambda eid: (-scores[eid], eid))[:10]
            seconds = time.perf_counter() - start
            metrics = {}
            for task in labels:
                positives = {eid for eid, score in labels[task][qid].items() if score > 0}
                if not positives: raise ValueError('No positive reviewers')
                found = len(positives & set(pool))
                metrics[task] = {'p5': precision_metrics(ranking, labels[task][qid])['p@5'],
                                 'pool_recall': found / len(positives), 'pool_oracle_p5': min(found, 5) / 5}
            arms[method] = {'seconds': seconds, 'pool_size': len(pool), 'metrics': metrics}
        rows.append({'query_id': qid, 'arms': arms})
    search.db.close()
    report = {'queries': len(rows), 'methods': {m: {'mean_seconds': statistics.mean(r['arms'][m]['seconds'] for r in rows),
              'mean_pool_size': statistics.mean(r['arms'][m]['pool_size'] for r in rows),
              'metrics': {t: {metric: statistics.mean(r['arms'][m]['metrics'][t][metric] for r in rows) for metric in ('p5', 'pool_recall', 'pool_oracle_p5')} for t in labels}} for m in methods},
              'input_sha256': {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in ('SIGIR_papers_test.json', 'SIGIR_reviewers_test.json', 'SIGIR_queries_test_soft.json', 'SIGIR_queries_test_hard.json')},
              'scope': 'Post-hoc fixed 10-title/10-abstract union, filled from full-query rank to 20; labels unavailable for candidate selection. Reused fully judged SIGIR reviewer data, not enterprise fit. One rotating-order timing per query excludes initialization, index build, evidence display and reasoning. Oracle ceiling is not achievable quality. No held-out or superiority claim.'}
    report['paired_union_minus_full20'] = {task: {metric: paired_interval([r['arms']['field_union20']['metrics'][task][metric] - r['arms']['full20']['metrics'][task][metric] for r in rows]) for metric in ('p5', 'pool_recall')} for task in labels}
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    (out / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
    report['results_sha256'] = hashlib.sha256((out / 'results.json').read_bytes()).hexdigest()
    report['code_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), Path('src/expert_search/certified_topk.py'), Path('src/expert_search/fast_retrieval.py'))}
    (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n'); print(json.dumps(report, indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
