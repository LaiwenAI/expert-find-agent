"""Reconcile independently attributed joint-pool reviews without guessing labels."""
import argparse
import json
from collections import Counter
from pathlib import Path
from .check_joint_review import FIELDS,check,load,sha


def reconcile(first,second,third=None):
    if set(first)!=set(second) or (third is not None and set(first)!=set(third)):
        raise ValueError('Review inventories differ')
    results={};counts={f:Counter() for f in FIELDS}
    for case in sorted(first):
        a,b=first[case],second[case];c=third[case] if third is not None else None
        if any(a[f] is not None and b[f] is not None for f in FIELDS):
            if not all(isinstance(r['reviewer_id'],str) and r['reviewer_id'].strip() for r in (a,b)) or a['reviewer_id'].strip()==b['reviewer_id'].strip():
                raise ValueError('Distinct reviewers required')
        result={}
        for f in FIELDS:
            x,y=a[f],b[f];z=c[f] if c is not None else None
            if z is not None:
                if x is None or y is None or x==y:raise ValueError('Adjudication requires a disagreement')
                if not isinstance(c['reviewer_id'],str) or not c['reviewer_id'].strip() or c['reviewer_id'].strip() in (a['reviewer_id'].strip(),b['reviewer_id'].strip()):
                    raise ValueError('Distinct adjudicator required')
                label=z;status='adjudicated' if z!='uncertain' else 'adjudicated_uncertain'
            elif x is None or y is None:label=None;status='missing_review'
            elif x!=y:label=None;status='disagreement'
            elif x=='uncertain':label='uncertain';status='agreed_uncertain'
            else:label=x;status='agreed'
            counts[f][status]+=1;result[f]={'label':label,'status':status}
        results[case]=result
    return results,{f:dict(c) for f,c in counts.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('first',type=Path);p.add_argument('second',type=Path)
    p.add_argument('--adjudication',type=Path);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    paths=[args.first,args.second]+([args.adjudication] if args.adjudication else [])
    for path in paths:check(args.root,path)
    results,counts=reconcile(load(args.first),load(args.second),load(args.adjudication) if args.adjudication else None)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'private-outcomes.json').write_text(json.dumps(results,indent=2)+'\n')
    report={'cases':len(results),'counts':counts,'packet_sha256':sha(args.root/'manifest.json'),
            'review_sha256':[sha(path) for path in paths],'outcomes_sha256':sha(args.output/'private-outcomes.json'),
            'scope':'Per-field structural reconciliation of fixed views. Distinct IDs do not prove independent or qualified human review. Missing, uncertain and disagreements are not negative labels. Output mappings are private; no inference about unpooled experts.'}
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
