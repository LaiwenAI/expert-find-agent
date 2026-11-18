"""Test missing-term complementarity on annotated cross-document evidence."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

from experiments.evidencebench_pilot import bm25
from experiments.retrieval.evidencebench_gap_selection import words

DATA_SHA256 = '4e9ecb5c8d3b719f624d66b60f8d56bf227f03914f5f0753d6fa1b359d7104ea'
CANDIDATE_LIMIT = 5


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assess(row):
    question = row['question']
    docs = row['context']
    titles = [t for t, _ in docs]
    excerpts = ['\n'.join([title, *sentences]) for title, sentences in docs]
    if len(set(titles)) != len(titles) or len(excerpts) < 2:
        raise ValueError('Invalid passage inventory')
    ranking = bm25(question, excerpts)
    first, second, *possible = ranking[:CANDIDATE_LIMIT]
    remaining = words(question)-words(excerpts[first])
    feasible = [i for i in possible if len(excerpts[i]) <= len(excerpts[second])]
    winner = max([second, *feasible], key=lambda i:len(remaining & words(excerpts[i])))

    declared = {tuple(fact) for fact in row['supporting_facts']}
    positions = {title:len(sentences) for title,sentences in docs}
    if not declared or any(t not in positions or type(i) is not int or not 0 <= i < positions[t]
                           for t, i in declared):
        return None
    target_titles = {t for t, _ in declared}
    if len(target_titles) < 2:
        raise ValueError('Expected cross-document supervision')
    def evaluate(i):
        selected = {titles[first], titles[i]}
        found = sum(title in selected for title, _ in declared)
        return {'all_supporting_docs':target_titles <= selected,
                'fact_recall':found/len(declared),
                'chars':len(excerpts[first])+len(excerpts[i])+1}
    arms = {'bm25':evaluate(second), 'complement':evaluate(winner)}
    oracle = any(evaluate(i)['all_supporting_docs'] for i in [second, *feasible])
    if arms['complement']['chars'] > arms['bm25']['chars']:
        raise ValueError('Character cap exceeded')
    return {'id':row['_id'], 'type':row['type'], 'first_supporting':titles[first] in target_titles,
            'switched':winner != second, 'oracle_complete':oracle, 'arms':arms}


def summarize(rows):
    result = {'questions':len(rows), 'switched':sum(r['switched'] for r in rows),
              'oracle_complete':sum(r['oracle_complete'] for r in rows)}
    for arm in ('bm25','complement'):
        measures = [r['arms'][arm] for r in rows]
        result[arm] = {'complete':sum(m['all_supporting_docs'] for m in measures),
                       'mean_fact_recall':mean(m['fact_recall'] for m in measures),
                       'mean_chars':mean(m['chars'] for m in measures)}
    before_after = [(r['arms']['bm25']['all_supporting_docs'],r['arms']['complement']['all_supporting_docs']) for r in rows]
    result['completed_by_switch'] = sum(not prior and new for prior,new in before_after)
    result['broken_by_switch'] = sum(prior and not new for prior,new in before_after)
    return result


def main(dataset, output):
    if sha(dataset) != DATA_SHA256:
        raise ValueError('Changed HotpotQA distractor development file')
    data = json.loads(dataset.read_text())
    if len(data) != 7405 or len({r['_id'] for r in data}) != 7405:
        raise ValueError('Changed benchmark cohort')
    rows = []
    excluded = []
    for record in data:
        result = assess(record)
        if result is None:
            excluded.append(record['_id'])
        else:
            rows.append(result)
    report = {'excluded_invalid_support_indices':excluded,'dataset_sha256':sha(dataset), 'script_sha256':sha(Path(__file__)),
              'all':summarize(rows), 'first_supporting':summarize([r for r in rows if r['first_supporting']]),
              'by_type':{t:summarize([r for r in rows if r['type']==t]) for t in sorted({r['type'] for r in rows})},
              'scope':'Public HotpotQA distractor development set. Given paragraphs, not open-corpus retrieval. Fixed first BM25 paragraph. Replace second with an affordable candidate in positions 3 to 5 only when it covers more missing question terms. Each paragraph is read in full. Gold annotations used only for metrics and a post-hoc oracle. Counts describe this split, with no answer generation, conflict judgment or enterprise expert validity.'}
    output.mkdir(parents=True,exist_ok=False)
    (output/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
    report['rows_sha256'] = sha(output/'rows.json')
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    main(args.dataset,args.output)
