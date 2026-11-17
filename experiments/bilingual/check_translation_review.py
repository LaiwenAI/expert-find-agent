"""Validate one partial or complete full-translation review copy."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


LABELS={'faithful_complete','changed_or_incomplete','uncertain'}


def read(path):
    rows=[json.loads(line) for line in path.read_text().splitlines()];index={r['case_id']:r for r in rows}
    if len(rows)!=len(index):raise ValueError('Duplicate case ID')
    return index


def check(root,review):
    manifest=json.loads((root/'manifest.json').read_text());baseline=root/'blind-review.jsonl'
    if hashlib.sha256(baseline.read_bytes()).hexdigest()!=manifest['review_sha256']:raise ValueError('Changed baseline')
    source=read(baseline);judged=read(review)
    if source.keys()!=judged.keys() or len(source)!=manifest['cases']:raise ValueError('Changed case set')
    counts=Counter()
    for cid,row in judged.items():
        original=source[cid]
        if row.keys()!=original.keys() or any(row[k]!=original[k] for k in ('original_need','english_translation')):raise ValueError('Changed source or translation')
        if row['label'] is not None and row['label'] not in LABELS:raise ValueError('Invalid label')
        if row['label'] is not None and not all(isinstance(row[k],str) and row[k].strip() for k in ('reason','reviewer_id')):raise ValueError('Missing reason or reviewer')
        counts[row['label'] or 'missing']+=1
    return dict(counts)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('review',type=Path);args=parser.parse_args()
    print(json.dumps(check(args.root,args.review),indent=2))
