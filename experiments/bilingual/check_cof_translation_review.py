"""Validate partial or complete independent CoF translation review."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def read(path):
    rows=[json.loads(line) for line in path.read_text().splitlines()];mapped={r['case_id']:r for r in rows}
    if len(mapped)!=len(rows):raise ValueError('Duplicate case')
    return mapped


def check(packet,review):
    manifest=json.loads((packet/'manifest.json').read_text());baseline=packet/'blind-review.jsonl';mapping=packet/'private-mapping.json'
    if hashlib.sha256(baseline.read_bytes()).hexdigest()!=manifest['review_sha256'] or hashlib.sha256(mapping.read_bytes()).hexdigest()!=manifest['mapping_sha256']:raise ValueError('Changed review packet')
    original=read(baseline);judged=read(review)
    if set(original)!=set(judged) or len(judged)!=manifest['cases']:raise ValueError('Changed cases')
    counts=Counter();mutable={'equivalence','rationale','reviewer_id'}
    for cid,row in judged.items():
        if row.keys()!=original[cid].keys() or any(row[k]!=original[cid][k] for k in row if k not in mutable):raise ValueError('Changed bilingual views')
        label=row['equivalence']
        if label is not None and label not in ('faithful','changed','uncertain'):raise ValueError('Invalid review label')
        if label is not None and not all(isinstance(row[k],str) and row[k].strip() for k in ('rationale','reviewer_id')):raise ValueError('Reviewer and rationale required')
        counts[label if label is not None else 'missing']+=1
    return {'counts':dict(counts),'review_sha256':hashlib.sha256(review.read_bytes()).hexdigest(),
            'scope':'Structural validation only; reviewer independence and translation correctness require external assessment.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('packet',type=Path);p.add_argument('review',type=Path);a=p.parse_args();print(json.dumps(check(a.packet,a.review),indent=2))
