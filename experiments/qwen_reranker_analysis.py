"""Revalidate complete Qwen runs against the identical Model1 candidate pools."""
import hashlib
import json
from pathlib import Path
from expert_search.cof_benchmark import load, precision_metrics
from expert_search.cof_crossencoder import checked_ranking
from expert_search.packing_analysis import paired_interval


def analyze(dataset, version):
    root, _, _, labels = load(dataset)
    inputs = root / 'packing-shared-v2'
    planned = json.loads((inputs / 'protocol.json').read_text())['all_query_ids']
    directory = root / f'qwen-reranker-mlx-v{version}'
    missing = [qid for qid in planned if not (directory / f'{qid}.json').exists()]
    if missing:
        return dict(dataset=dataset, version=version, complete=False,
                    completed=len(planned)-len(missing), planned=len(planned), missing=missing)
    differences = {task: {metric: [] for metric in ('p@5', 'ndcg@10')} for task in labels}
    hashes = {}
    for qid in planned:
        path = directory / f'{qid}.json'
        row = json.loads(path.read_text())
        source = json.loads((inputs / f'{qid}-similarity.json').read_text())
        aliases = source['aliases']
        if row['query_id'] != qid or checked_ranking(row['scores'], aliases) != row['ranking']:
            raise ValueError('Invalid query or ranking')
        if row['input_sha256'] != hashlib.sha256((inputs / f'{qid}-similarity-input.json').read_bytes()).hexdigest():
            raise ValueError('Changed source input')
        if row['aliases_sha256'] != hashlib.sha256(json.dumps(aliases).encode()).hexdigest():
            raise ValueError('Changed alias mapping')
        baseline = json.loads((root / 'language-models-v2' / f'query-{qid}.json').read_text())
        ranking = [eid for eid in baseline['full_rankings']['model1-document'] if eid in set(aliases)]
        if set(ranking) != set(aliases):
            raise ValueError('Different candidate pools')
        for task, truth in labels.items():
            observed = precision_metrics(row['ranking'], truth[qid])
            control = precision_metrics(ranking, truth[qid])
            if observed != row['metrics'][task]:
                raise ValueError('Changed stored metrics')
            for metric in differences[task]:
                differences[task][metric].append(observed[metric]-control[metric])
        hashes[qid] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(dataset=dataset, version=version, complete=True, queries=len(planned),
                qwen_minus_model1_in_same_pool={task: {metric: paired_interval(values)
                    for metric, values in metrics.items()} for task, metrics in differences.items()},
                query_hashes=hashes,
                limitations='Exploratory query bootstrap with reused benchmark queries and shared profiles. Timing configurations require their own complete runs.')


if __name__ == '__main__':
    for dataset, version in [('NIPS', 1), ('SIGIR', 2)]:
        report = analyze(dataset, version)
        Path(f'docs/exploration/qwen-{dataset.lower()}-v{version}-analysis.json').write_text(json.dumps(report, indent=2))
        print(json.dumps({k: v for k, v in report.items() if k != 'query_hashes'}, indent=2))
