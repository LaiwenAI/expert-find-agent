"""Create local method-blind review of screening and direct-agent outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import random


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main(output):
    prior=Path('artifacts/enterprise-agent-budget-pilot-v1');screened=Path('artifacts/enterprise-decisions-explain-v1')
    a=json.loads((prior/'protocol.json').read_text());b=json.loads((screened/'protocol.json').read_text())
    if a['query_ids']!=b['query_ids'] or sha(prior/'protocol.json')!=b['prior_protocol_sha256']:
        raise ValueError('Changed paired cohorts')
    cases=[];hashes={}
    for qid in a['query_ids']:
        for arm,path in (('direct',prior/(qid+'-default.json')),('decisions_explain',screened/(qid+'.json'))):
            raw=path.read_bytes(); hashes[qid+'-'+arm]=hashlib.sha256(raw).hexdigest(); row=json.loads(raw)
            if row['query_id']!=qid:raise ValueError('Changed query identity')
            if row['status']!='valid':continue
            for rank,item in enumerate(row['result']['results'],1):
                sources=[]
                for i,e in enumerate(item['evidence'],1):
                    if e['expert_id']!=item['expert']['id'] or e['text'][e['start']:e['end']]!=e['quote']:
                        raise ValueError('Changed owned citation')
                    sources.append({'source':f'source-{i}','visible_text':e['text'],'quoted_text':e['quote']})
                cases.append({'query_id':qid,'arm':arm,'expert_id':item['expert']['id'],'rank':rank,'query':row['result']['query'],
                              'reason':item['reason'],'missing':item['missing'],'sources':sources})
    random.Random(20261121).shuffle(cases);forms=[];mapping=[]
    for i,c in enumerate(cases,1):
        case=f'case-{i:03d}'
        forms.append({'case_id':case,'query':c['query'],'reason':c['reason'],'missing':c['missing'],'sources':c['sources'],
                      'factual_support':None,'query_relevance':None,'personal_role_support':None,'review_notes':None,'reviewer_id':None})
        mapping.append({'case_id':case,'query_id':c['query_id'],'arm':c['arm'],'expert_id':c['expert_id'],'rank':c['rank']})
    output.mkdir(parents=True,exist_ok=False)
    (output/'blind-review.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in forms))
    (output/'private-mapping.json').write_text(json.dumps(mapping,indent=2)+'\n')
    report={'cases':len(forms),'seed':20261121,'blind_review_sha256':sha(output/'blind-review.jsonl'),
            'private_mapping_sha256':sha(output/'private-mapping.json'),'attempt_sha256':hashes,'code_sha256':sha(Path(__file__)),
            'scope':'Private blinded returned-output review. Source text may reveal identity; labels and model choices are not shown. Arms have different output budgets, and repeated experts are dependent. No human labels yet; unreturned or failed candidates remain unjudged.'}
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'cases':len(forms),'scope':report['scope']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
