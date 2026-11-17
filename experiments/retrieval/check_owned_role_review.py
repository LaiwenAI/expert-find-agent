"""Validate a completed or partial model-blind role review."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


LABELS={'method','result','interpretation','background','other','mixed','uncertain'}


def read(path):
    rows=[json.loads(line) for line in path.read_text().splitlines()];index={r['case_id']:r for r in rows}
    if len(index)!=len(rows):raise ValueError('Duplicate case ID')
    return index


def check(root,review):
    manifest=json.loads((root/'manifest.json').read_text());baseline=root/'blind-review.jsonl'
    if hashlib.sha256(baseline.read_bytes()).hexdigest()!=manifest['review_sha256']:raise ValueError('Changed baseline')
    source=read(baseline);judged=read(review)
    if source.keys()!=judged.keys() or len(source)!=manifest['cases']:raise ValueError('Changed case set')
    counts=Counter()
    for cid,row in judged.items():
        original=source[cid]
        if set(row)!=set(original) or row['source_text']!=original['source_text'] or len(row['windows'])!=len(original['windows']):raise ValueError('Changed source')
        for window,expected in zip(row['windows'],original['windows']):
            if set(window)!=set(expected) or any(window[k]!=expected[k] for k in ('id','start','end','text')):raise ValueError('Changed window')
            label=window['label']
            if label is not None and label not in LABELS:raise ValueError('Invalid role label')
            if label is not None and not all(isinstance(v,str) and v.strip() for v in (row['reviewer_id'],window['reason'])):raise ValueError('Missing reviewer or reason')
            counts[label or 'missing']+=1
    if sum(counts.values())!=manifest['windows']:raise ValueError('Window count mismatch')
    return dict(counts)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('review',type=Path);args=parser.parse_args()
    print(json.dumps(check(args.root,args.review),indent=2))
