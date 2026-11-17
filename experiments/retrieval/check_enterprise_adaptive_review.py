"""Validate blinded evidence reviews and summarize missing-label bounds."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

FIELDS = {'factual_support': {'supported', 'unsupported', 'uncertain'},
          'query_relevance': {'supported', 'unsupported', 'uncertain'},
          'personal_role_support': {'supported', 'unknown'}}
MUTABLE = set(FIELDS) | {'review_notes', 'reviewer_id'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def indexed(rows):
    result = {}
    for row in rows:
        key = row['case_id']
        if key in result:
            raise ValueError('Duplicate case ID')
        result[key] = row
    return result


def check(root, review):
    root, review = Path(root), Path(review)
    manifest = json.loads((root / 'manifest.json').read_text())
    baseline, mapping_path = root / 'blind-review.jsonl', root / 'private-mapping.json'
    if digest(baseline) != manifest['blind_review_sha256'] or digest(mapping_path) != manifest['private_mapping_sha256']:
        raise ValueError('Changed original packet or mapping')
    original = indexed(json.loads(line) for line in baseline.read_text().splitlines())
    judged = indexed(json.loads(line) for line in review.read_text().splitlines())
    mapping = indexed(json.loads(mapping_path.read_text()))
    if original.keys() != judged.keys() or original.keys() != mapping.keys() or len(original) != manifest['cases']:
        raise ValueError('Changed case set')
    grouped = {}
    for case, source in original.items():
        row, match = judged[case], mapping[case]
        if row.keys() != source.keys() or any(row[k] != source[k] for k in source if k not in MUTABLE):
            raise ValueError('Changed review content')
        if any(row[field] is not None and row[field] not in options for field, options in FIELDS.items()):
            raise ValueError('Invalid label')
        if any(row[field] is not None for field in FIELDS):
            if any(not isinstance(row[k], str) or not row[k].strip() for k in ('review_notes', 'reviewer_id')):
                raise ValueError('Missing reviewer ID or rationale')
        arm = match['arm']
        grouped.setdefault(arm, []).append(row)
    arms = {}
    for arm, rows in sorted(grouped.items()):
        arms[arm] = {}
        for field in FIELDS:
            counts = Counter(row[field] if row[field] is not None else 'missing' for row in rows)
            n = len(rows); supported = counts['supported']; unknown = counts['missing'] + counts['uncertain'] + counts['unknown']
            arms[arm][field] = {'counts': dict(sorted(counts.items())), 'cases': n,
                                'supported_fraction_bounds': [supported / n, (supported + unknown) / n]}
    return {'cases': len(original), 'arms': arms, 'review_sha256': digest(review),
            'scope': 'Descriptive bounds among returned cases only: uncertain, unknown and missing labels may be positive or negative. Arms may have different candidate sets, repeated experts and failed attempts omitted from returned cases; bounds are not paired query metrics, confidence intervals, expert-level recall, end-to-end accuracy or proof of personal contribution.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('review', type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.root, args.review), indent=2))
