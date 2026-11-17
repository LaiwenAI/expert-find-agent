"""Validate completed source-only SciFact review without inferring missing labels."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from experiments.scifact.decisions_oracle_pilot import sha


def check(root,review):
    manifest=json.loads((root/'manifest.json').read_text());mapping_path=root/'private-mapping.json'
    if sha(mapping_path)!=manifest['mapping_sha256']:raise ValueError('Changed frozen review mapping')
    mapping=json.loads(mapping_path.read_text());rows=json.loads(review.read_text())
    if not isinstance(rows,list) or len(rows)!=manifest['cases'] or len(mapping)!=len(rows):raise ValueError('Changed inventory')
    by_id={m['case_id']:m for m in mapping}
    if len(by_id)!=len(mapping) or {r.get('case_id') for r in rows}!=set(by_id):raise ValueError('Changed or duplicate review IDs')
    counts=Counter()
    for row in rows:
        if set(row)!={'case_id','claim','title','abstract','view_sha256','verdict','evidence_quote','reviewer_id','notes'}:raise ValueError('Changed review schema')
        view={key:row[key] for key in ('case_id','claim','title','abstract')}
        digest=hashlib.sha256(json.dumps(view,sort_keys=True,ensure_ascii=True).encode()).hexdigest()
        if digest!=row['view_sha256'] or digest!=by_id[row['case_id']]['view_sha256']:raise ValueError('Changed fixed view')
        verdict=row['verdict'];quote=row['evidence_quote'];reviewer=row['reviewer_id']
        if verdict is None:
            if quote is not None or reviewer is not None or row['notes'] is not None:raise ValueError('Incomplete review must remain blank')
            counts['missing']+=1;continue
        if verdict not in ('support','contradict','unknown') or not isinstance(reviewer,str) or not reviewer.strip():raise ValueError('Invalid verdict or reviewer')
        if row['notes'] is not None and not isinstance(row['notes'],str):raise ValueError('Invalid review notes')
        if verdict in ('support','contradict') and (not isinstance(quote,str) or not quote.strip() or not any(quote in sentence for sentence in row['abstract'])):
            raise ValueError('A supported or contradicted claim requires an exact abstract quote')
        if verdict=='unknown' and quote is not None and (not isinstance(quote,str) or not any(quote in sentence for sentence in row['abstract'])):raise ValueError('Invalid optional quote')
        counts[verdict]+=1
    return {'cases':len(rows),'counts':dict(counts),'review_sha256':sha(review),'mapping_sha256':sha(mapping_path),
            'scope':'Structural and exact-quote checks only; they do not validate semantics, reviewer independence or source truth. Missing remains unknown.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--review',type=Path,required=True);a=p.parse_args();print(json.dumps(check(a.root,a.review),indent=2))
