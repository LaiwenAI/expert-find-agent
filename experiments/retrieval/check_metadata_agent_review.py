"""Validate a completed or partial private explanation audit."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


FIELDS=('explanation_support','individual_contribution','date_representation')
LABELS={'supported','changed_or_unsupported','uncertain'}


def read(path):
    rows=[json.loads(line) for line in path.read_text().splitlines()];result={r['case_id']:r for r in rows}
    if len(result)!=len(rows):raise ValueError('Duplicate review cases')
    return result


def check(packet,review):
    manifest=json.loads((packet/'manifest.json').read_text());baseline=packet/'blind-review.jsonl';mapping=packet/'private-mapping.json'
    if hashlib.sha256(baseline.read_bytes()).hexdigest()!=manifest['review_sha256'] or hashlib.sha256(mapping.read_bytes()).hexdigest()!=manifest['mapping_sha256']:raise ValueError('Changed packet')
    original=read(baseline);judged=read(review)
    if set(original)!=set(judged) or len(judged)!=manifest['cases']:raise ValueError('Changed case inventory')
    counts={f:Counter() for f in FIELDS};mutable=set(FIELDS)|{'rationale','reviewer_id'}
    for case,row in judged.items():
        if row.keys()!=original[case].keys() or any(row[k]!=original[case][k] for k in row if k not in mutable):raise ValueError('Changed source or explanation')
        if any(row[f] is not None and row[f] not in LABELS for f in FIELDS):raise ValueError('Invalid review label')
        if any(row[f] is not None for f in FIELDS) and not all(isinstance(row[k],str) and row[k].strip() for k in ('rationale','reviewer_id')):raise ValueError('Reviewer and rationale required')
        for field in FIELDS:counts[field][row[field] if row[field] is not None else 'missing']+=1
    return {'counts':{f:dict(v) for f,v in counts.items()},'review_sha256':hashlib.sha256(review.read_bytes()).hexdigest(),
            'scope':'Structural validation only; reviewer independence, factual dates, source entailment and personal contribution remain unverified.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('packet',type=Path);p.add_argument('review',type=Path);a=p.parse_args();print(json.dumps(check(a.packet,a.review),indent=2))
