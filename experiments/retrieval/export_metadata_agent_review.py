"""Create a private blind audit of date-aware candidate-agent explanations."""
import argparse
import hashlib
import json
import random
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(run,output):
    protocol=json.loads((run/'protocol.json').read_text());summary=json.loads((run/'summary.json').read_text())
    if sha(run/'protocol.json')!=summary['protocol_sha256'] or sha(run/'view-hashes.json')!=summary['view_hashes_sha256']:raise ValueError('Changed pilot artifacts')
    items=[];input_hashes={}
    for qid in protocol['query_ids']:
        path=run/(hashlib.sha256(qid.encode()).hexdigest()+'.json');input_hashes[path.name]=sha(path);row=json.loads(path.read_text())
        if row['query_id']!=qid:raise ValueError('Changed query identity')
        if row['status']!='valid':continue
        result=row['result']
        for rank,expert in enumerate(result['results'],1):
            eid=expert['expert']['id']
            if not expert['evidence'] or any(e['expert_id']!=eid or e['text'][e['start']:e['end']]!=e['quote'] for e in expert['evidence']):raise ValueError('Changed cited evidence')
            items.append({'query_id':qid,'rank':rank,'expert_id':eid,'need':result['query'],'reason':expert['reason'],'missing':expert['missing'],'evidence':expert['evidence']})
    random.Random(20261122).shuffle(items);forms=[];mapping=[]
    for i,item in enumerate(items,1):
        case=f'case-{i:03d}';evidence=[]
        for j,e in enumerate(item['evidence'],1):
            evidence.append({'source_id':f'source-{j}','text':e['text'],'quote':e['quote'],'kind':e['kind'],'publication':e['publication'],'contribution_status':e['contribution_status']})
        forms.append({'case_id':case,'original_need':item['need'],'explanation':item['reason'],'missing_requirements':item['missing'],'cited_sources':evidence,
                      'explanation_support':None,'individual_contribution':None,'date_representation':None,'rationale':None,'reviewer_id':None})
        mapping.append({'case_id':case,'query_id':item['query_id'],'rank':item['rank'],'expert_id':item['expert_id'],'source_ids':[e['id'] for e in item['evidence']]})
    output.mkdir(parents=True,exist_ok=False)
    payload=''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in forms).encode();mapping_data=(json.dumps(mapping,indent=2)+'\n').encode()
    (output/'blind-review.jsonl').write_bytes(payload);(output/'private-mapping.json').write_bytes(mapping_data)
    report={'cases':len(forms),'seed':20261122,'pilot_protocol_sha256':sha(run/'protocol.json'),'pilot_summary_sha256':sha(run/'summary.json'),
            'input_sha256':input_hashes,'review_sha256':hashlib.sha256(payload).hexdigest(),'mapping_sha256':hashlib.sha256(mapping_data).hexdigest(),
            'scope':'Private local explanations and quoted sources. Review hides algorithm, rank and explicit expert ID, but text may reveal identity. Explanation is visible because its support is under review. Review date claims against supplied metadata only; metadata truth and personal role require separate verification. No human labels are supplied.'}
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'cases':len(forms),'scope':report['scope']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.run,a.output)
