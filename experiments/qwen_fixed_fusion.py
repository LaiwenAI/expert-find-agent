"""Exploratory fixed equal-weight fusion after completed Qwen controls."""
import json
from pathlib import Path
from expert_search.cof_benchmark import load, precision_metrics
from expert_search.packing_analysis import paired_interval


def run(dataset, version):
    root, _, _, labels = load(dataset)
    directory = root / f'qwen-reranker-mlx-v{version}'
    protocol = json.loads((root / 'packing-shared-v2/protocol.json').read_text())
    rows = []
    for qid in protocol['all_query_ids']:
        result = json.loads((directory / f'{qid}.json').read_text())
        aliases = json.loads((root / 'packing-shared-v2' / f'{qid}-similarity.json').read_text())['aliases']
        baseline = json.loads((root / 'language-models-v2' / f'query-{qid}.json').read_text())
        prior = [e for e in baseline['full_rankings']['model1-document'] if e in set(aliases)]
        if set(prior) != set(aliases):
            raise ValueError('Changed candidate pool')
        feature = {e: 1-r/(len(prior)-1) if len(prior)>1 else 1. for r,e in enumerate(prior)}
        scores = {aliases[s['id']]: s['score'] for s in result['scores']}
        ranking = sorted(aliases, key=lambda e: (-.5*(feature[e]+scores[e]), e))
        rows.append({task: {'fusion': precision_metrics(ranking, truth[qid]),
                           'prior': precision_metrics(prior, truth[qid])}
                     for task, truth in labels.items()})
    return dict(dataset=dataset, queries=len(rows),
        metrics={task: {metric: sum(r[task]['fusion'][metric] for r in rows)/len(rows)
                       for metric in ('p@5','ndcg@10')} for task in labels},
        contrasts={task: {metric: paired_interval([r[task]['fusion'][metric]-r[task]['prior'][metric] for r in rows])
                         for metric in ('p@5','ndcg@10')} for task in labels},
        scope='Retrospective fixed equal-weight extension on reused benchmarks. No calibration, held-out evidence or pruning timing measured.')


if __name__ == '__main__':
    results = [run('NIPS',1), run('SIGIR',2)]
    Path('docs/exploration/qwen-fixed-fusion.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))
