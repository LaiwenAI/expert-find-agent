"""Prepare blind human review for the two enterprise agent paths."""
import hashlib
import json
import random
from pathlib import Path


def main():
    parent=Path('artifacts/enterprise-adaptive-agent-pilot-v1');protocol=json.loads((parent/'protocol.json').read_text());cases=[];hashes={}
    for qid in protocol['query_ids']:
        for arm in protocol['arms']:
            path=parent/(qid+'-'+arm+'.json');raw=path.read_bytes();hashes[path.name]=hashlib.sha256(raw).hexdigest();row=json.loads(raw)
            if row['status']!='valid':continue
            result=row['result']
            for rank,item in enumerate(result['results'],1):
                sources=[]
                for index,e in enumerate(item['evidence'],1):
                    if e['expert_id']!=item['expert']['id'] or e['text'][e['start']:e['end']]!=e['quote']:
                        raise ValueError('Changed evidence')
                    sources.append(dict(source=f'source-{index}',visible_text=e['text'],quoted_text=e['quote']))
                cases.append(dict(query_id=qid,arm=arm,expert_id=item['expert']['id'],rank=rank,
                                  query=result['query'],reason=item['reason'],missing=item['missing'],sources=sources))
    random.Random(20261127).shuffle(cases);forms=[];mapping=[]
    for i,c in enumerate(cases,1):
        case=f'case-{i:03d}'
        forms.append(dict(case_id=case,query=c['query'],reason=c['reason'],missing=c['missing'],sources=c['sources'],
                          factual_support=None,query_relevance=None,personal_role_support=None,review_notes=None,reviewer_id=None))
        mapping.append(dict(case_id=case,query_id=c['query_id'],arm=c['arm'],expert_id=c['expert_id'],rank=c['rank']))
    out=Path('artifacts/enterprise-adaptive-human-review-v1');out.mkdir(parents=True,exist_ok=False)
    form=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in forms).encode();private=(json.dumps(mapping,indent=2)+'\n').encode()
    (out/'blind-review.jsonl').write_bytes(form);(out/'private-mapping.json').write_bytes(private)
    (out/'README.txt').write_text('Review without the private mapping. For factual support and query relevance use supported, unsupported, or uncertain. For personal role use supported or unknown; a paper association alone does not establish personal contribution. Use only the visible sources. Missing evidence is unknown rather than proof of inability. Record notes and reviewer ID.\n')
    report=dict(cases=len(forms),parent_protocol_sha256=hashlib.sha256((parent/'protocol.json').read_bytes()).hexdigest(),parent_result_sha256=hashes,
                blind_review_sha256=hashlib.sha256(form).hexdigest(),private_mapping_sha256=hashlib.sha256(private).hexdigest(),seed=20261127,
                scope='Local sensitive blind packet; model path, expert IDs and ranks hidden, but source content may reveal identity. Duplicate experts can recur across paths. No human judgments yet; unresolved arm contributes no completed results.')
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='parent_result_sha256'},indent=2))


if __name__=='__main__':main()
