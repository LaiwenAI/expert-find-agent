"""Replay frozen EvidenceBench sentence rankings under matched character caps."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

from expert_search.evidencebench_data import aspect_coverage, prepare
from expert_search.statistics import clustered_paired_interval

ROOT = Path('artifacts/evidencebench-role-test-v1')
BUDGETS = (500, 1000, 2000)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def budget_prefix(sentences, ranking, budget):
    selected = []
    used = 0
    for position in ranking:
        length = len(sentences[position].strip()) + bool(selected)
        if used + length > budget:
            break
        selected.append(position)
        used += length
    return selected, used


def main(output):
    protocol = ROOT / 'protocol.json'
    evaluation = ROOT / 'evaluation.json'
    verification = json.loads((ROOT / 'verification.json').read_text())
    for path in (protocol, evaluation):
        if sha(path) != verification['artifacts'][str(path)]:
            raise ValueError(f'Changed frozen artifact: {path}')
    data = prepare('test')
    dataset = Path('artifacts/public/evidencebench/test.json')
    if sha(dataset) != json.loads(protocol.read_text())['dataset_sha256']:
        raise ValueError('Changed frozen test set')
    previous = json.loads(evaluation.read_text())
    if len(previous) != len(data) or {r['id'] for r in previous} != set(data):
        raise ValueError('Changed query cohort')

    rows = []
    for record in previous:
        query = data[record['id']]
        sentences = query['paper_as_candidate_pool']
        results = {'id': record['id'], 'paper_id': query['paper_id'], 'budgets': {}}
        for budget in BUDGETS:
            arms = {}
            for method, selection_key, metric_key in (
                ('flat', 'baseline_selected', 'baseline'), ('role', 'selected', 'metrics')
            ):
                ranking = record[selection_key]
                if aspect_coverage(query, ranking) != record[metric_key]:
                    raise ValueError('Saved evidence metrics differ')
                selected, used = budget_prefix(sentences, ranking, budget)
                arms[method] = {**aspect_coverage(query, selected), 'characters': used}
            results['budgets'][str(budget)] = arms
        rows.append(results)

    summary = {'queries': len(rows), 'budgets': {}, 'input_sha256': {
        'evaluation': sha(evaluation), 'protocol': sha(protocol), 'dataset': sha(dataset),
        'script': sha(Path(__file__)),
    }, 'scope': 'Post-hoc replay of 293 previously evaluated EvidenceBench test queries. Uses saved top-ten sentence lists and strips outer whitespace. One character per join separator. Stops before the first sentence exceeding the shared cap. Measures labeled aspect coverage and characters, excluding loading, tokenization, metadata, generation and model verification. Cluster intervals are descriptive and unadjusted across budgets.'}
    for cap in BUDGETS:
        arms = [r['budgets'][str(cap)] for r in rows]
        summary['budgets'][str(cap)] = {
            method: {metric: mean(a[method][metric] for a in arms)
                     for metric in ('aspect_recall', 'characters', 'selected_sentences')}
            for method in ('flat', 'role')
        }
        summary['budgets'][str(cap)]['role_minus_flat'] = clustered_paired_interval(
            [a['role']['aspect_recall'] - a['flat']['aspect_recall'] for a in arms],
            [r['paper_id'] for r in rows],
        )
    output.mkdir(parents=True, exist_ok=False)
    (output/'rows.json').write_text(json.dumps(rows, indent=2)+'\n')
    summary['rows_sha256'] = sha(output/'rows.json')
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
