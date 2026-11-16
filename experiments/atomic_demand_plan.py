"""Demand-only atomic decomposition with explicit residual coverage obligations."""
import hashlib
import json
from pathlib import Path

PROMPT='''Return one JSON object. Decompose the supplied technical requirement into 1–6 independently verifiable atomic research conditions. Preserve named application objects and limitations. Do not turn a specific application into generic related expertise. Retain project future targets as future_target, never prior achievements. Do not see or optimize against expert publications. Return {"atoms":[{"id":str,"text":str,"quote":str,"role":"research_topic|future_target|explicit_prior_qualification"}],"composition":"all_of|single|unresolved","residual_requirements":[str]}. Quotes must be unique contiguous substrings of the original requirement, at least 4 characters. Each atom must describe one operation or one property with its object. For conjunction, use all_of and preserve any cross-atom relation as an explicit residual requirement. If alternatives, negation, or dependencies cannot be represented faithfully, use unresolved and record them. Never claim that supported atoms alone prove a residual relationship or full expert suitability.'''


def run():
    from laiwenai import LaiwenAI
    parent=Path('artifacts/demand-first-pool-v2/jobs.jsonl');spec=json.loads(Path('artifacts/demand-first-audit-v2/protocol.json').read_text())
    jobs=[j for j in map(json.loads,parent.read_text().splitlines()) if j['id'] in spec['ids']]
    out=Path('artifacts/atomic-demand-plan-v1');out.mkdir(exist_ok=True)
    protocol=dict(version=1,model='gpt-6-luna',prompt=PROMPT,parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),ids=spec['ids'])
    marker=out/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=protocol:raise ValueError('Changed atomic plan protocol')
    marker.write_text(json.dumps(protocol,indent=2));client=LaiwenAI(model='gpt-6-luna',max_retries=0);rows=[]
    for job in jobs:
        target=out/(job['id']+'.json')
        if target.exists():row=json.loads(target.read_text())
        else:
            row=dict(id=job['id'],status='unresolved')
            try:
                text=job['facet']['text']
                response=client.chat.completions.create(messages=[dict(role='system',content=PROMPT),dict(role='user',content=text)],response_format=dict(type='json_object'),reasoning_effort='none')
                row['response']=response.model_dump(mode='json');a=json.loads(response.choices[0].message.content)
                atoms=a['atoms']
                if not isinstance(atoms,list) or not 1<=len(atoms)<=6 or a['composition'] not in ('single','all_of','unresolved'):raise ValueError('Invalid decomposition')
                if len({x['id'] for x in atoms})!=len(atoms):raise ValueError('Duplicate atom IDs')
                for atom in atoms:
                    if atom['role'] not in ('research_topic','future_target','explicit_prior_qualification') or not atom['text'].strip() or len(atom['quote'])<4 or text.count(atom['quote'])!=1:raise ValueError('Invalid atomic anchor')
                if not isinstance(a['residual_requirements'],list) or any(not isinstance(r,str) or not r.strip() for r in a['residual_requirements']):raise ValueError('Invalid residuals')
                row.update(status='structurally_valid',plan=a)
            except Exception as error:row.update(error_type=type(error).__name__,http_status=getattr(error,'status_code',None))
            temporary=target.with_suffix('.partial');temporary.write_text(json.dumps(row,ensure_ascii=False,indent=2));temporary.replace(target)
        rows.append(row);print(job['parent_demand'],row['status'],flush=True)
    report=dict(planned=len(jobs),valid=sum(r['status']=='structurally_valid' for r in rows),atoms=sum(len(r.get('plan',{}).get('atoms',[])) for r in rows),
        warning='Demand anchors and schema only. Faithful coverage, atomicity and semantic equivalence need independent audit; no expert labels.')
    (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':
    from dotenv import load_dotenv
    load_dotenv('.env');run()
