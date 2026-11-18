"""Audit whether lexical substitutions add annotated evidence beyond the first sentence."""
import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean

from experiments.retrieval.evidencebench_character_budget import ROOT, sha
from experiments.retrieval.evidencebench_gap_selection import words
from expert_search.evidencebench_data import prepare
from expert_search.statistics import clustered_paired_interval


def evaluate(data, rankings):
    rows = []
    for record in rankings:
        query = data[record['id']]
        first, second, alternative = record['selected'][:3]
        sentences = query['paper_as_candidate_pool']
        lengths = {i: len(sentences[i].strip()) for i in (first,second,alternative)}
        missing = words(query['hypothesis']) - words(sentences[first])
        feasible = lengths[alternative] <= lengths[second]
        switch = feasible and len(missing & words(sentences[alternative])) > len(missing & words(sentences[second]))
        chosen = alternative if switch else second
        annotations = query['sentence_index2aspects']
        known = set(annotations.get(str(first), ()))
        def gain(i):
            return len(set(annotations.get(str(i), ())) - known)
        def category(i):
            observed = set(annotations.get(str(i), ()))
            if observed - known:
                return 'adds_annotated_aspect'
            return 'only_previously_covered_aspects' if observed else 'no_annotated_aspect'
        rows.append({
            'id':record['id'], 'paper':query['paper_id'], 'first':first, 'baseline_second':second,
            'selected_second':chosen, 'alternative':alternative, 'alternative_within_budget':feasible,
            'first_has_annotation':bool(known), 'switched':switch,
            'baseline_new_aspects':gain(second), 'selected_new_aspects':gain(chosen),
            'baseline_type':category(second), 'selected_type':category(chosen),
            'oracle_new_aspects':max(gain(second), gain(alternative)) if feasible else gain(second),
            'baseline_chars':lengths[first]+lengths[second]+1,
            'selected_chars':lengths[first]+lengths[chosen]+1,
        })
    return rows


def summarize(rows):
    differences = [r['selected_new_aspects']-r['baseline_new_aspects'] for r in rows]
    return {
        'queries':len(rows),
        'baseline_categories':dict(Counter(r['baseline_type'] for r in rows)),
        'selected_categories':dict(Counter(r['selected_type'] for r in rows)),
        'more_new_aspects':sum(d>0 for d in differences), 'fewer_new_aspects':sum(d<0 for d in differences),
        'mean_new_aspects':{arm:mean(r[f'{arm}_new_aspects'] for r in rows)
                            for arm in ('baseline','selected','oracle')},
        'mean_input_chars':{arm:mean(r[f'{arm}_chars'] for r in rows)
                            for arm in ('baseline','selected')},
        'difference':clustered_paired_interval(differences,[r['paper'] for r in rows]),
    }


def main(output):
    protocol = ROOT/'protocol.json'
    evaluation = ROOT/'evaluation.json'
    verified = json.loads((ROOT/'verification.json').read_text())['artifacts']
    for path in (protocol, evaluation):
        if sha(path) != verified[str(path)]:
            raise ValueError(f'Changed frozen artifact: {path}')
    data = prepare('test')
    dataset = Path('artifacts/public/evidencebench/test.json')
    if sha(dataset) != json.loads(protocol.read_text())['dataset_sha256']:
        raise ValueError('Changed benchmark data')
    records = json.loads(evaluation.read_text())
    if len(records) != len(data) or {r['id'] for r in records} != set(data):
        raise ValueError('Changed query cohort')
    rows = evaluate(data, records)
    if any(r['selected_chars'] > r['baseline_chars'] for r in rows):
        raise ValueError('Selection exceeded source budget')
    switched = [r for r in rows if r['switched']]
    report = {'inputs_sha256':{p.name:sha(p) for p in (dataset, evaluation, Path(__file__))},
              'all':summarize(rows), 'switched':summarize(switched) if switched else None,
              'feasible_alternatives':sum(r['alternative_within_budget'] for r in rows),
              'scope':'Post-hoc reuse of frozen EvidenceBench test rankings. Fixed first sentence. The third can replace the second if it fits within the original second sentence character budget and overlaps more unmet hypothesis terms. Uses full sentence annotations after selection. No annotation is counted as no labeled aspect, without inferring absence of support. This is within-paper sentence complementarity. No conflict, individual attribution, enterprise fit, or measured latency.'}
    output.mkdir(parents=True, exist_ok=False)
    (output/'rows.json').write_text(json.dumps(rows, indent=2)+'\n')
    report['rows_sha256'] = sha(output/'rows.json')
    (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
