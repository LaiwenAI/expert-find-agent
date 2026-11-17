"""Validate one completed or partial blind review against its frozen export."""
import argparse
import collections
import hashlib
import json
from pathlib import Path


FIELDS = ('chinese_grounding', 'english_grounding', 'bilingual_equivalence')
LABELS = {'supported', 'changed_or_unsupported', 'uncertain'}


def read(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    result = {r['case_id']: r for r in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate case identifiers')
    return result


def check(baseline, review, manifest):
    if hashlib.sha256(baseline.read_bytes()).hexdigest() != manifest['output_sha256']:
        raise ValueError('Baseline hash mismatch')
    original, judged = read(baseline), read(review)
    if original.keys() != judged.keys() or len(original) != manifest['cases']:
        raise ValueError('Missing or extra cases')
    counts = {field: collections.Counter() for field in FIELDS}
    mutable = set(FIELDS) | {'rationale', 'reviewer_id'}
    for key, row in judged.items():
        if row.keys() != original[key].keys() or any(row[k] != original[key][k] for k in row if k not in mutable):
            raise ValueError('Source fields changed')
        labels = [row[field] for field in FIELDS]
        if any(label is not None and label not in LABELS for label in labels):
            raise ValueError('Invalid label')
        if any(label is not None for label in labels) and not all(isinstance(row[k], str) and row[k].strip() for k in ('rationale', 'reviewer_id')):
            raise ValueError('Labeled records require rationale and reviewer ID')
        for field, label in zip(FIELDS, labels):
            counts[field][label if label is not None else 'missing'] += 1
    return {field: dict(counter) for field, counter in counts.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('review', type=Path)
    parser.add_argument('--root', type=Path, default=Path('artifacts/grounding-human-review-v1'))
    args = parser.parse_args()
    print(json.dumps(check(args.root / 'blind-review.jsonl', args.review,
                           json.loads((args.root / 'manifest.json').read_text())), indent=2))
