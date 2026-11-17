"""Validate a partial or completed local evidence review without changing labels."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


FIELDS=('evidence_fit','core_capability_support')
LABELS={'supported','changed_or_unsupported','uncertain'}


def read(path):
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    indexed={r['case_id']:r for r in rows}
    if len(indexed)!=len(rows):raise ValueError('Duplicate case ID')
    return indexed


def check(root,review):
    manifest=json.loads((root/'manifest.json').read_text());baseline=root/'blind-review.jsonl'
    if hashlib.sha256(baseline.read_bytes()).hexdigest()!=manifest['review_sha256']:raise ValueError('Changed baseline')
    original=read(baseline);judged=read(review)
    if original.keys()!=judged.keys() or len(original)!=manifest['cases']:raise ValueError('Changed case set')
    counts={field:Counter() for field in FIELDS};mutable=set(FIELDS)|{'reviewer_id','rationale'}
    for key,row in judged.items():
        if row.keys()!=original[key].keys() or any(row[k]!=original[key][k] for k in row if k not in mutable):raise ValueError('Changed source fields')
        if any(row[f] is not None and row[f] not in LABELS for f in FIELDS):raise ValueError('Invalid label')
        if any(row[f] is not None for f in FIELDS) and not all(isinstance(row[k],str) and row[k].strip() for k in ('reviewer_id','rationale')):raise ValueError('Missing reviewer or rationale')
        for field in FIELDS:counts[field][row[field] or 'missing']+=1
    return {field:dict(count) for field,count in counts.items()}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('review',type=Path)
    args=parser.parse_args();print(json.dumps(check(args.root,args.review),indent=2))
