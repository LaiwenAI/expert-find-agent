"""Screen validation needs for asymmetric character-shingle reuse."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import re
from pathlib import Path


def grams(text, width=4):
    compact = re.sub(r'\s+', '', text)
    return {compact[i:i + width] for i in range(max(0, len(compact) - width + 1))}


def analyze(rows):
    construction = [r for r in rows if r['split'] == 'construction']
    validation = [r for r in rows if r['split'] == 'validation']
    if not construction or not validation or len({r['id'] for r in construction + validation}) != len(construction) + len(validation):
        raise ValueError('Invalid cohort')
    if {r['group_id'] for r in construction} & {r['group_id'] for r in validation}:
        raise ValueError('Group crosses splits')
    reference = [grams(r['zh']) for r in construction]
    index = defaultdict(list)
    for i, source in enumerate(reference):
        for gram in source: index[gram].append(i)
    records = []
    for row in validation:
        query = grams(row['zh'])
        if not query: raise ValueError('Empty shingle set')
        counts = Counter(i for gram in query for i in index[gram])
        # Fraction of the validation need covered; unlike Jaccard this can
        # expose an embedded short need inside a much longer reference need.
        best = max(counts.items(), key=lambda item: (item[1], -item[0])) if counts else (None, 0)
        score = best[1] / len(query)
        records.append({'id': row['id'], 'best_construction_id': construction[best[0]]['id'] if best[0] is not None else None,
                        'validation_shingles': len(query), 'covered_fraction': score})
    report = {'construction': len(construction), 'validation': len(validation),
              'max_validation_containment': max(r['covered_fraction'] for r in records),
              'threshold_counts': {str(t): sum(r['covered_fraction'] >= t for r in records) for t in (.5, .7, .8, .9)},
              'scope': 'Post-hoc whitespace-stripped character four-gram containment, validation against all construction needs. High containment is a review flag, not a duplicate or relevance label; templates and common terminology can inflate it. Low containment does not rule out paraphrases, aliases or semantic leakage. No sealed queries were used.'}
    return records, report


def main(source, output):
    raw = Path(source).read_bytes(); rows = [json.loads(line) for line in raw.splitlines()]
    records, report = analyze(rows)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    (output / 'private-results.json').write_text(json.dumps(records, indent=2) + '\n')
    report['input_sha256'] = hashlib.sha256(raw).hexdigest()
    report['results_sha256'] = hashlib.sha256((output / 'private-results.json').read_bytes()).hexdigest()
    report['code_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path); parser.add_argument('output', type=Path)
    args = parser.parse_args(); main(args.source, args.output)
