"""Measure label-based oracle ceilings for frozen reviewer shortlists."""
import hashlib
import json
import statistics
from pathlib import Path

from expert_search.certified_topk import certified_topk
from expert_search.cof_benchmark import load
from expert_search.compiled_language_model import CompiledCandidateLM
from expert_search.expert_language_models import ExpertLanguageModels


def main():
    root, papers, reviewers, labels = load('SIGIR')
    source_ids = sorted({pid for reviewer in reviewers.values() for pid in reviewer['papers']})
    documents = {pid: (papers[pid].get('title', '') + '\n' + papers[pid].get('abstract', '')).strip() for pid in source_ids}
    model = CompiledCandidateLM(ExpertLanguageModels(documents, {eid: reviewer['papers'] for eid, reviewer in reviewers.items()}))
    if set(model.available) != set(reviewers) or set(labels['soft']) != set(labels['hard']):
        raise ValueError('Incomplete candidate or query set')
    if any(set(labels[task][qid]) != set(reviewers) for task in labels for qid in labels[task]):
        raise ValueError('Incomplete reviewer judgments')
    rows = []
    for qid in sorted(labels['soft']):
        paper = papers[qid]; query = (paper.get('title', '') + '\n' + paper.get('abstract', '')).strip()
        ranking = [item['id'] for item in certified_topk(model, query, 50)['results']]
        row = {'query_id': qid, 'tasks': {}}
        for task in labels:
            positive = {eid for eid, score in labels[task][qid].items() if score > 0}
            if not positive:
                raise ValueError('No positive reviewers for query')
            metrics = {}
            for k in (20, 50):
                pool = set(ranking[:k]); found = len(positive & pool)
                metrics[str(k)] = {'relevant_in_pool': found, 'relevant_total': len(positive),
                                   'recall': found / len(positive), 'oracle_p5': min(found, 5) / 5}
            row['tasks'][task] = metrics
        rows.append(row)
    report = {'queries': len(rows), 'reviewers': len(reviewers), 'tasks': {}}
    for task in labels:
        report['tasks'][task] = {}
        for k in ('20', '50'):
            stats = [row['tasks'][task][k] for row in rows]
            report['tasks'][task][k] = {'mean_recall': statistics.mean(x['recall'] for x in stats),
                                        'mean_oracle_p5': statistics.mean(x['oracle_p5'] for x in stats),
                                        'queries_with_fewer_than_5_positives': sum(x['relevant_in_pool'] < 5 for x in stats)}
    report['input_sha256'] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in ('SIGIR_papers_test.json', 'SIGIR_reviewers_test.json', 'SIGIR_queries_test_soft.json', 'SIGIR_queries_test_hard.json')}
    report['code_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), Path('src/expert_search/certified_topk.py'), Path('src/expert_search/compiled_language_model.py'))}
    report['scope'] = 'Exploratory oracle uses test labels unavailable at inference; ceilings are not achievable performance. Fully judged reviewer matching, not enterprise-fit or evidence support. Model and shortlist sizes were previously examined on this test set; not a held-out policy choice.'
    out = Path('artifacts/cof-sigir-shortlist-ceiling-v1'); out.mkdir(parents=True, exist_ok=False)
    (out / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
    report['results_sha256'] = hashlib.sha256((out / 'results.json').read_bytes()).hexdigest()
    (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
