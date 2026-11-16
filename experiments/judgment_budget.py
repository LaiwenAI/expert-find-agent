"""Hide complete SIGIR labels and simulate deterministic comparison-bound audits."""
import json
import random
from pathlib import Path
from expert_search.cof_benchmark import load
from expert_search.query_lift import plan_comparison_judgments


def run():
    root, _, _, labels = load('SIGIR')
    qids = sorted(labels['soft'])
    runs = {}
    for folder in ('baseline-api', 'enrichment-v1'):
        for qid in qids:
            row = json.loads((root/folder/(qid+'.json')).read_text())
            for method, value in row['results']['soft'].items():
                runs.setdefault(method, {})[qid] = value['ranking']
    for condition in ('full', 'similarity', 'coverage'):
        runs['inline-'+condition] = {qid: json.loads((root/'packing-v1'/f'{qid}-{condition}.json').read_text())['ranking'] for qid in qids}
    report = dict(dataset='SIGIR', queries=len(qids), comparisons={},
        limitations='Retrospective exploratory simulation, not actual human annotation or system improvement. Numeric original benchmark labels are revealed, not source evidence or expert skill. Selection uses rankings only. Worst-case box bounds are conservative, not confidence intervals. Equal costs and resolvable labels assumed. Common candidate universe required; sparse query-dependent eligibility needs a different null model. Fixed comparison choices were made after exploratory baseline results.')
    for first, second in (('dense-enrichment', 'bm25-max'), ('inline-similarity', 'inline-full')):
        if first not in runs:
            # Fail clearly rather than silently substituting a different method.
            raise ValueError(f'Method unavailable: {first}; available={sorted(runs)}')
        table = {}
        for metric in ('gain', 'lift'):
            plan = plan_comparison_judgments(runs[first], runs[second], {}, 100000, metric=metric)
            prioritized = [(r['query_id'], r['expert_id']) for r in plan['requests']]
            random_order = list(prioritized)
            random.Random(20261116).shuffle(random_order)
            result = dict(nonzero_cells=len(prioritized), original_judged_pairs=sum(map(len, labels['soft'].values())), curves={})
            for task, truth in labels.items():
                exact = plan_comparison_judgments(runs[first], runs[second], truth, 0, metric=metric)['bounds']
                if not exact['identified']:
                    raise ValueError('Simulation requires complete labels')
                task_results = {}
                for strategy, order in (('coefficient_mass', prioritized), ('random_nonzero_cells', random_order)):
                    curve = []
                    for budget in sorted({0, min(50,len(order)), min(100,len(order)), min(250,len(order)),
                                          min(500,len(order)), min(1000,len(order)), min(2000,len(order)), len(order)}):
                        revealed = {}
                        for qid, eid in order[:budget]:
                            revealed.setdefault(qid, {})[eid] = truth[qid][eid]
                        bound = plan_comparison_judgments(runs[first], runs[second], revealed, 0, metric=metric)['bounds']
                        if not bound['lower']-1e-10 <= exact['lower'] <= bound['upper']+1e-10:
                            raise ValueError('Identification bound excluded exact value')
                        curve.append(dict(budget=budget, **bound, width=bound['upper']-bound['lower']))
                    task_results[strategy] = curve
                result['curves'][task] = dict(exact_difference=exact['lower'], strategies=task_results)
            table[metric] = result
        report['comparisons'][first+'-minus-'+second] = table
    destination = Path('docs/exploration/judgment-budget-sigir.json')
    destination.write_text(json.dumps(report, indent=2))
    print(destination)


if __name__ == '__main__':
    run()
