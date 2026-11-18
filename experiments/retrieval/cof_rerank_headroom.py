"""Separate candidate-pool headroom from gains available to an existing reranker."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

from expert_search.cof_benchmark import load

ROOT = Path('artifacts')
BUDGET = 12


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(path, intervention, labels):
    report = json.loads((path/'summary.json').read_text())
    assert sha(path/'rows.json') == report['rows_sha256']
    rows = json.loads((path/'rows.json').read_text())
    if len(rows) != report['queries']:
        raise ValueError('Changed query count')
    statistics = {}
    for task, judgments in labels.items():
        changes = []
        for row in rows:
            arms = row['arms']
            dense = arms['dense']
            new = arms[intervention]
            if dense['pool'] != new['pool'] or dense['sources'] != new['sources']:
                raise ValueError('Mismatched candidate pool')
            pool = dense['pool']
            relevant = {expert for expert, grade in judgments[row['query_id']].items() if grade > 0}
            base = sum(hit['expert'] in relevant for hit in dense['hits'])
            rank = sum(hit['expert'] in relevant for hit in new['hits'])
            if base/5 != dense['precision5'][task] or rank/5 != new['precision5'][task]:
                raise ValueError('Recorded estimates differ from labels')
            allowed = (sum(e in relevant for e in pool[:3]) + min(2, len(relevant & set(pool[3:7])))
                       if intervention == 'boundary' else min(5, len(relevant & set(pool))))
            if any(v < 0 for v in (allowed-base, allowed-rank)):
                raise ValueError('Observed accuracy exceeds oracle')
            changes.append({'base':base, 'rank':rank, 'allowed':allowed, 'delta':rank-base})
        upper = sorted((r['delta'] for r in changes if r['delta'] > 0), reverse=True)
        statistics[task] = {
            'actual_dense_p5': mean(r['base']/5 for r in changes),
            'actual_reranked_p5': mean(r['rank']/5 for r in changes),
            'oracle_candidate_ceiling_p5': mean(r['allowed']/5 for r in changes),
            'reranking_can_improve': sum(r['allowed'] > r['base'] for r in changes),
            'reranker_improves': len(upper),
            'reranker_worsens': sum(r['delta'] < 0 for r in changes),
            'oracle_route_all_positive_p5': mean((r['base']+max(0,r['delta']))/5 for r in changes),
            'oracle_route_budget12_p5': (sum(r['base'] for r in changes)+sum(upper[:BUDGET]))/(5*len(changes)),
            'oracle_route_budget12_calls': min(BUDGET,len(upper)),
        }
        if intervention == 'boundary':
            statistics[task]['actual_gated_p5'] = mean(r['arms']['gated']['precision5'][task] for r in rows)
    return {'queries':len(rows), 'source_job':report['job_id'], 'rows_sha256':report['rows_sha256'],
            'results':statistics}


def main(output):
    root, _, _, labels = load('SIGIR')
    paths = {task:root/f'SIGIR_queries_test_{task}.json' for task in labels}
    label_hashes = {task:sha(path) for task,path in paths.items()}
    experiments = [('cof-live-open-rerank-v1','rerank'), ('cof-live-boundary-rerank-v1','boundary')]
    results = {}
    for folder, intervention in experiments:
        path = ROOT/folder
        summary = json.loads((path/'summary.json').read_text())
        if summary['label_sha256'] != label_hashes:
            raise ValueError('Changed reviewer grades')
        results[intervention] = compare(path, intervention, labels)
    report = {'results':results, 'script_sha256':sha(Path(__file__)), 'labels_sha256':label_hashes,
              'scope':'Descriptive post-hoc analysis of previously explored CoF SIGIR rankings. Oracle candidate ceilings use reviewer self-ratings to choose the best admissible slate. Oracle routing uses those same labels to choose cases improved by a fixed reranker; the soft and hard oracles are distinct and use up to twelve calls. Oracles are diagnostic, not deployable policies. The two cohorts overlap and have different reranking actions.'}
    output.mkdir(parents=True, exist_ok=False)
    (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
