"""Validate partial joint-pool judgments against their immutable source views."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

FIELDS=('method_fit','constraint_support','personal_contribution')
LABELS={'supported','not_supported_by_view','uncertain'}
MUTABLE=set(FIELDS)|{'rationale','reviewer_id'}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    index={row['case_id']:row for row in rows}
    if len(index)!=len(rows):raise ValueError('Duplicate case ID')
    return index


def check(root,review):
    manifest=json.loads((root/'manifest.json').read_text())
    for name in ('blind-review.jsonl','private-mapping.jsonl'):
        if sha(root/name)!=manifest['outputs_sha256'][name]:raise ValueError('Frozen packet hash mismatch')
    original=load(root/'blind-review.jsonl');mapping=load(root/'private-mapping.jsonl');judged=load(review)
    if set(original)!=set(mapping) or set(original)!=set(judged) or len(original)!=manifest['candidate_views']:
        raise ValueError('Changed case set')
    counts={field:Counter() for field in FIELDS}
    for case,base in original.items():
        row=judged[case];private=mapping[case]
        if set(row)!=set(base) or any(row[k]!=base[k] for k in base if k not in MUTABLE):
            raise ValueError('Changed need or source view')
        aliases={v['alias']:v for v in private['sources']}
        if len(aliases)!=len(private['sources']) or {v['id'] for v in base['sources']}!=set(aliases):
            raise ValueError('Changed source mapping')
        for source in base['sources']:
            view=aliases[source['id']]
            if hashlib.sha256(source['text'].encode()).hexdigest()!=view['sha256'] or len(source['text'])!=view['end']-view['start']:
                raise ValueError('Source view mismatch')
        for field in FIELDS:
            label=row[field]
            if label is not None and (not isinstance(label,str) or label not in LABELS):raise ValueError('Invalid label')
            counts[field][label if label is not None else 'missing']+=1
        if any(row[f] is not None for f in FIELDS) and any(not isinstance(row[k],str) or not row[k].strip() for k in ('reviewer_id','rationale')):
            raise ValueError('Judgments require reviewer and rationale')
    return {'cases':len(original),'counts':{f:dict(c) for f,c in counts.items()},
            'baseline_sha256':sha(root/'blind-review.jsonl'),'mapping_sha256':sha(root/'private-mapping.jsonl'),'review_sha256':sha(review),
            'scope':'Validation of fixed displayed views, not proof of expert capability, source entailment, independent review, or completeness. Missing and uncertain are distinct; not_supported_by_view is not inability.'}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);parser.add_argument('review',type=Path)
    args=parser.parse_args();print(json.dumps(check(args.root,args.review),indent=2))
