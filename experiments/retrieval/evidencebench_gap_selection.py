"""Compare label-free evidence selection rules and an annotation-based upper bound."""
import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean

from experiments.retrieval.evidencebench_character_budget import ROOT, budget_prefix, sha
from expert_search.evidencebench_data import aspect_coverage, prepare
from expert_search.statistics import clustered_paired_interval

CAPS = (500, 1000, 2000)


def words(text):
    return set(re.findall(r'[a-z0-9]{3,}', text.casefold()))


def select(ranking, costs, terms, weights, cap, track_coverage):
    remaining = list(ranking)
    covered = set()
    chosen = []
    spent = 0
    while remaining:
        affordable = [i for i in remaining if spent + costs[i] <= cap]
        if not affordable:
            break
        def priority(i):
            relevant = terms[i] - covered if track_coverage else terms[i]
            return sum(weights[t] for t in sorted(relevant)) / costs[i]
        candidate = max(affordable, key=priority)
        chosen.append(candidate)
        remaining.remove(candidate)
        spent += costs[candidate]
        covered.update(terms[candidate])
    return chosen


def oracle_coverage(ranking, costs, annotations, cap):
    aspects = sorted({a for i in ranking for a in annotations.get(str(i), ())})
    bit = {a: 1 << j for j, a in enumerate(aspects)}
    min_cost = {0: 0}
    for i in ranking:
        mask = sum(bit[a] for a in set(annotations.get(str(i), ())))
        for previous, spent in list(min_cost.items()):
            total = spent + costs[i]
            if total <= cap:
                union = previous | mask
                min_cost[union] = min(total, min_cost.get(union, cap + 1))
    return max(mask.bit_count() for mask in min_cost)


def main(output):
    evaluation = ROOT / 'evaluation.json'
    protocol = ROOT / 'protocol.json'
    verified = json.loads((ROOT / 'verification.json').read_text())['artifacts']
    for path in (evaluation, protocol):
        if sha(path) != verified[str(path)]:
            raise ValueError(f'Changed artifact: {path}')
    data = prepare('test')
    dataset = Path('artifacts/public/evidencebench/test.json')
    if sha(dataset) != json.loads(protocol.read_text())['dataset_sha256']:
        raise ValueError('Changed test set')
    records = json.loads(evaluation.read_text())
    if len(records) != len(data) or {r['id'] for r in records} != set(data):
        raise ValueError('Changed query cohort')

    rows = []
    for record in records:
        query = data[record['id']]
        sentences = query['paper_as_candidate_pool']
        ranking = record['selected']
        if aspect_coverage(query, ranking) != record['metrics']:
            raise ValueError('Changed role ranking')
        # Charge one join character per sentence and add one to each budget.
        costs = {i: len(sentences[i].strip()) + 1 for i in ranking}
        query_terms = words(query['hypothesis'])
        terms = {i: words(sentences[i]) & query_terms for i in ranking}
        counts = Counter(t for i in ranking for t in terms[i])
        weights = {t: 1 / count for t, count in counts.items()}
        row = {'id': record['id'], 'paper_id': query['paper_id'], 'caps': {}}
        for cap in CAPS:
            prefix, _ = budget_prefix(sentences, ranking, cap)
            selections = {'prefix': prefix}
            for method in ('static_terms', 'uncovered_terms'):
                selections[method] = select(ranking, costs, terms, weights, cap + 1,
                                            track_coverage=method == 'uncovered_terms')
            scores = {}
            for method, selected in selections.items():
                used = sum(costs[i] for i in selected) - bool(selected)
                if used > cap:
                    raise ValueError('Exceeded character budget')
                scores[method] = {**aspect_coverage(query, selected), 'characters': used}
            oracle = oracle_coverage(ranking, costs, query['sentence_index2aspects'], cap + 1)
            if oracle < max(score['covered_aspects'] for score in scores.values()):
                raise ValueError('Oracle failed to upper-bound tested selections')
            row['caps'][str(cap)] = {'arms': scores,
                'oracle_recall': oracle / scores['prefix']['total_aspects']}
        rows.append(row)

    summary = {'queries': len(rows), 'caps': {}, 'inputs_sha256': {
        'evaluation': sha(evaluation), 'protocol': sha(protocol), 'dataset': sha(dataset),
        'script': sha(Path(__file__)),
    }, 'scope': 'Post-hoc replay on the same frozen role-prioritized top-ten pool. Term policies use hypothesis words, inverse pool frequency, and sentence length. They do not use evaluation labels. The oracle uses annotations to optimize coverage and cannot be deployed. Character budgets omit metadata and model overhead. This measures benchmark aspect coverage, not expert fit or end-to-end cost.'}
    for cap in CAPS:
        batch = [r['caps'][str(cap)] for r in rows]
        methods = batch[0]['arms']
        summary['caps'][str(cap)] = {
            'arms': {m: {k: mean(item['arms'][m][k] for item in batch)
                         for k in ('aspect_recall', 'characters', 'selected_sentences')} for m in methods},
            'oracle_recall': mean(item['oracle_recall'] for item in batch),
            'uncovered_minus_static': clustered_paired_interval(
                [item['arms']['uncovered_terms']['aspect_recall'] - item['arms']['static_terms']['aspect_recall']
                 for item in batch], [r['paper_id'] for r in rows]),
        }
    output.mkdir(parents=True, exist_ok=False)
    (output / 'rows.json').write_text(json.dumps(rows, indent=2) + '\n')
    summary['rows_sha256'] = sha(output / 'rows.json')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
