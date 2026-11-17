"""Prepare method-blind review forms for retrieved-pool coverage results."""
import argparse
import hashlib
import json
import random
from pathlib import Path

from .fixed_evidence_routing import preserve_candidate_coverage


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();protocol=json.loads((args.run/'protocol.json').read_text())
    query_file=Path('artifacts/dataset-release/fresh-enterprise-eval-v1/queries.jsonl')
    if hashlib.sha256(query_file.read_bytes()).hexdigest()!=protocol['query_file_sha256']:raise ValueError('Changed queries')
    queries={r['id']:r for r in map(json.loads,query_file.read_text().splitlines()) if r['split']=='development'}
    cases=[];attempts={}
    for qid in protocol['query_ids']:
        if qid not in queries:raise ValueError('Query is not in development')
        raw=(args.run/(qid+'-pool.json')).read_bytes();digest=hashlib.sha256(raw).hexdigest();hits=json.loads(raw)
        evidence={e['id']:e for h in hits for e in h['evidence']}
        for arm in protocol['arms']:
            path=args.run/(qid+'-'+arm+'.json')
            if not path.exists():raise ValueError('Incomplete run')
            row=json.loads(path.read_text());attempts[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
            if (row['query_id'],row['arm'],row['input_sha256'])!=(qid,arm,digest):raise ValueError('Input mismatch')
            if row['status']!='ok':continue
            shown=set(evidence) if arm=='direct' else set(preserve_candidate_coverage(hits,[]))
            outputs=row['trace'][-1]['output']['results']
            for rank,eid in enumerate(row['ranking'],1):
                citations=[]
                for candidate in outputs:
                    if candidate.get('expert_id')!=eid:continue
                    citations=list(dict.fromkeys(c for c in candidate.get('citations',[]) if isinstance(c,str) and c in shown and evidence[c]['expert_id']==eid))
                    if citations:break
                if not citations:raise ValueError('Missing valid citations')
                cases.append((qid,arm,rank,eid,citations,evidence))
    random.Random(20261122).shuffle(cases);forms=[];mapping=[]
    for i,(qid,arm,rank,eid,citations,evidence) in enumerate(cases,1):
        cid=f'case-{i:03d}'
        forms.append({'case_id':cid,'original_need':queries[qid]['text'],
                      'evidence':[{'id':f'source-{j}','text':evidence[src]['text'][:6000]} for j,src in enumerate(citations,1)],
                      'evidence_fit':None,'core_capability_support':None,'rationale':None,'reviewer_id':None})
        mapping.append({'case_id':cid,'query_id':qid,'arm':arm,'rank':rank,'expert_id':eid,'source_ids':citations})
    args.output.mkdir(parents=True,exist_ok=False)
    review=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in forms).encode()
    private=(json.dumps(mapping,indent=2,ensure_ascii=False)+'\n').encode()
    (args.output/'blind-review.jsonl').write_bytes(review);(args.output/'private-mapping.json').write_bytes(private)
    manifest={'cases':len(forms),'seed':20261122,'review_sha256':hashlib.sha256(review).hexdigest(),
              'mapping_sha256':hashlib.sha256(private).hexdigest(),'attempt_sha256':attempts,
              'scope':'Local private text; source identity may be apparent. Only returned citations are reviewed; no full-corpus recall or expert ability labels. No human judgments yet.'}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'cases':len(forms),'scope':manifest['scope']}))


if __name__=='__main__':main()
