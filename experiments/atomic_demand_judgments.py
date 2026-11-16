"""Native Decisions on demand-only atoms with unchanged expert evidence."""
import hashlib
import inspect
import json
from pathlib import Path
from expert_search.decisions_backend import DecisionsBackend


def run():
    parent=Path('artifacts/demand-first-pool-v2/jobs.jsonl')
    jobs={j['id']:j for j in map(json.loads,parent.read_text().splitlines())}
    plan_root=Path('artifacts/atomic-demand-plan-v1');out=Path('artifacts/atomic-demand-judgments-v2');out.mkdir(exist_ok=True)
    plans={p.stem:json.loads(p.read_text()) for p in sorted(plan_root.glob('*expert-*.json'))}
    plans={key:p for key,p in plans.items() if p['status']=='structurally_valid'}
    protocol=dict(version=1,model='gpt-6-luna',backend_sha256=hashlib.sha256(inspect.getsource(DecisionsBackend).encode()).hexdigest(),
        parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),plans_sha256=hashlib.sha256(json.dumps(plans,sort_keys=True).encode()).hexdigest(),
        scope='Same expert and original three-source snapshot; one batched request per plan. No new retrieval, no full-demand label.')
    marker=out/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=protocol:raise ValueError('Changed atomic judgment inputs')
    marker.write_text(json.dumps(protocol,indent=2));backend=DecisionsBackend();rows=[]
    for key,plan in plans.items():
        job=jobs[key];target=out/(key+'.json')
        if target.exists():row=json.loads(target.read_text())
        else:
            atoms=[a for a in plan['plan']['atoms'] if a['role']=='research_topic']
            row=dict(id=key,status='no_research_atoms',atoms=len(atoms),composition=plan['plan']['composition'],residual_requirements=plan['plan']['residual_requirements'])
            if atoms:
                evidence=json.dumps(job['sources'],ensure_ascii=False)
                try:
                    response=backend.judge([dict(id=a['id'],constraint=a['text'],text=evidence) for a in atoms])
                    row.update(status='valid',response=response,supported=sum(j['label']=='supported' for j in response['judgments']))
                except Exception as error:row.update(status='unresolved',error_type=type(error).__name__,http_status=getattr(error,'status_code',None))
            temporary=target.with_suffix('.partial');temporary.write_text(json.dumps(row,ensure_ascii=False,indent=2));temporary.replace(target)
        rows.append(row);print(key,row['status'],row.get('supported'),flush=True)
    valid=[r for r in rows if r['status']=='valid']
    report=dict(plans=len(plans),valid_plans=len(valid),judged_research_atoms=sum(r['atoms'] for r in valid),supported_atoms=sum(r['supported'] for r in valid),
        partially_supported_plans=sum(0<r['supported']<r['atoms'] for r in valid),all_research_atoms_supported_plans=sum(r['supported']==r['atoms'] for r in valid),
        accepted_dataset_labels=0,warning='Uncalibrated single-model judgments without quotes or independent decomposition audit. Partial support and even all research atoms supported do not establish residual relations, future targets, or full suitability.')
    (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':
    from dotenv import load_dotenv
    load_dotenv('.env');run()
