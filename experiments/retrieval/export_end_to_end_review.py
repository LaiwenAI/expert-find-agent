"""Prepare local, method-blind review forms for completed search outputs."""
import argparse
import hashlib
import json
import random
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();protocol=json.loads((args.run/'protocol.json').read_text())
    attempts=[]
    for qid in protocol['query_ids']:
        for arm in protocol['arms']:
            path=args.run/(qid+'-'+arm+'.json')
            if not path.exists():raise ValueError('Cannot export an incomplete run')
            row=json.loads(path.read_text())
            if row['query_id']!=qid or row['arm']!=arm:raise ValueError('Attempt identity mismatch')
            attempts.append((path,row))
    selected=[]
    for path,attempt in attempts:
        if attempt['status']!='ok':continue
        result=attempt['result']
        for rank,item in enumerate(result['results'],1):
            eid=item['expert']['id'];evidence=item['evidence']
            if not evidence or any(e['expert_id']!=eid for e in evidence):raise ValueError('Invalid evidence ownership')
            selected.append((path,attempt,rank,item))
    rng=random.Random(20261121);rng.shuffle(selected);forms=[];mapping=[]
    for number,(path,attempt,rank,item) in enumerate(selected,1):
        case=f'case-{number:03d}'
        evidence=[{'id':f'source-{i}', 'text':e['text'][:6000]} for i,e in enumerate(item['evidence'],1)]
        forms.append({'case_id':case,'original_need':attempt['result']['query'],'evidence':evidence,
                      'evidence_fit':None,'core_capability_support':None,'rationale':None,'reviewer_id':None})
        mapping.append({'case_id':case,'query_id':attempt['query_id'],'arm':attempt['arm'],'rank':rank,
                        'expert_id':item['expert']['id'],'source_ids':[e['id'] for e in item['evidence']]})
    args.output.mkdir(parents=True,exist_ok=False)
    payload=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in forms).encode()
    private=(json.dumps(mapping,indent=2,ensure_ascii=False)+'\n').encode()
    (args.output/'blind-review.jsonl').write_bytes(payload);(args.output/'private-mapping.json').write_bytes(private)
    report={'cases':len(forms),'attempts':len(attempts),'failed_attempts':sum(r['status']!='ok' for _,r in attempts),
            'seed':20261121,'protocol_sha256':hashlib.sha256((args.run/'protocol.json').read_bytes()).hexdigest(),
            'attempt_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p,_ in attempts},
            'review_sha256':hashlib.sha256(payload).hexdigest(),'mapping_sha256':hashlib.sha256(private).hexdigest(),
            'scope':'Local private evidence text; hides method, rank, expert ID and generated explanations from forms, but text can reveal source or method. Only returned cited evidence is shown; these are not exhaustive expert relevance labels. No human judgments yet.'}
    (args.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'cases':len(forms),'failed_attempts':report['failed_attempts'],'scope':report['scope']}))


if __name__=='__main__':main()
