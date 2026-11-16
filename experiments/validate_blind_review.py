"""Validate returned source-review forms without assigning semantic judgments."""
import argparse
import hashlib
import json
from pathlib import Path
from expert_search.source_counterfactual import validate_judgments


def validate(path, packet='artifacts/blind-source-review-v1'):
    root=Path(packet)
    manifest=json.loads((root/'manifest.json').read_text())
    original=(root/'inputs.jsonl').read_bytes()
    if hashlib.sha256(original).hexdigest()!=manifest['input_sha256']:
        raise ValueError('Changed review packet')
    inputs={row['id']:row for row in map(json.loads,original.decode().splitlines())}
    submitted=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids=[row['id'] for row in submitted]
    if len(ids)!=len(set(ids)) or set(ids)!=set(inputs):
        raise ValueError('Require exactly one submission for every packet item')
    validated=[]
    reviewers=set()
    for row in submitted:
        reference=inputs[row['id']]
        payload={key:value for key,value in row.items() if key!='review'}
        expected={key:value for key,value in reference.items() if key!='review'}
        if payload!=expected:
            raise ValueError('Submitted source or condition text changed')
        review=row['review']
        reviewer=review.get('reviewer_id')
        if not isinstance(reviewer,str) or not reviewer.strip():
            raise ValueError('Missing reviewer ID')
        labels,related=validate_judgments(review,{s['id']:s for s in row['sources']},row['original_demand'])
        reviewers.add(reviewer)
        validated.append(dict(id=row['id'],reviewer_id=reviewer,labels=labels,demand_related=related))
    if len(reviewers)!=1:
        raise ValueError('Submit each reviewer in a separate complete file')
    return dict(items=len(validated),reviewer_id=next(iter(reviewers)),results=validated,
                packet_sha256=manifest['input_sha256'],submission_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                scope='Structural and quote-location validation. Semantic agreement and adjudication require separate review.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('submission')
    parser.add_argument('--packet',default='artifacts/blind-source-review-v1')
    args=parser.parse_args()
    print(json.dumps(validate(args.submission,args.packet),indent=2))
