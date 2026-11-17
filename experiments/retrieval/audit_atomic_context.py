"""Repeat frozen atomic claim checks with full visible source context."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import hashlib
import json
from pathlib import Path
from dotenv import load_dotenv
from laiwenai import LaiwenAI
from expert_search.choice_judgments import decide
from .audit_reason_atomic_claims import OPTIONS

PARENT=Path('artifacts/anchored-temporal-candidate-pilot-v1')
CLAIMS=Path('artifacts/anchored-atomic-claims-audit-v1')
OUT=Path('artifacts/anchored-atomic-context-audit-v1')


def main():
    protocol=json.loads((CLAIMS/'protocol.json').read_text());jobs=[]
    for qid in json.loads((PARENT/'protocol.json').read_text())['queries']:
        raw=(PARENT/(qid+'.json')).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=protocol['source_sha256'][qid]:raise ValueError('Changed recommendations')
        row=json.loads(raw)
        for i,result in enumerate(row['response']['results']):
            cid=f'{qid}-{i}';claim_path=CLAIMS/'records'/(cid+'.json');record=json.loads(claim_path.read_text())
            if record['status']!='valid':continue
            facts=[(j,c) for j,c in enumerate(record['claims']) if c['kind']=='source_fact']
            if any(c['text'] not in result['reason'] for _,c in facts):raise ValueError('Changed claim')
            jobs.append(dict(id=cid,claims=facts,passages=[e['text'] for e in result['evidence']],previous=record['source_fact_judgments']))
    OUT.mkdir(parents=True,exist_ok=False);load_dotenv('.env')
    frozen=dict(source_protocol_sha256=hashlib.sha256((CLAIMS/'protocol.json').read_bytes()).hexdigest(),
                scope='Repeat on identical model-extracted source-fact claims with full visible source instead of quotes. No source truth, personal role, transfer validity, independent judge or causal latency inference.')
    (OUT/'protocol.json').write_text(json.dumps(frozen,indent=2)+'\n')
    def one(job):
        result=dict(id=job['id'],status='valid',labels={})
        if job['claims']:
            try:
                payload=dict(passages=job['passages'],claims={f'claim_{i}':c['text'] for i,c in job['claims']})
                questions=[dict(type='choice',name=f'claim_{i}',instructions=f'Do the passages explicitly support the exact factual statement named claim_{i}? Do not infer additional facts.',choices=OPTIONS) for i,_ in job['claims']]
                result['labels']=decide(LaiwenAI(model='gpt-6-luna',max_retries=0),payload,questions)['labels']
            except Exception as error:result.update(status='unresolved',error_type=type(error).__name__)
        result['previous']=job['previous'];return result
    with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(one,jobs))
    pairs=Counter((r['previous'][key],value) for r in rows if r['status']=='valid' for key,value in r['labels'].items())
    report=dict(cases=len(rows),valid=sum(r['status']=='valid' for r in rows),paired_claims=sum(pairs.values()),
                transitions=[dict(quote=a,context=b,count=n) for (a,b),n in sorted(pairs.items())],scope=frozen['scope'])
    (OUT/'results.json').write_text(json.dumps(rows,indent=2)+'\n');(OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
