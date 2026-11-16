"""Auxiliary same-family reviewer; never a substitute for the frozen audit gate."""
import hashlib
import importlib.metadata
import json
import random
import time
from pathlib import Path
from expert_search.corpus import records
from expert_search.source_counterfactual import AUDITOR,build_views,validate_judgments


def write(path,value):
    temporary=path.with_suffix('.partial');temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2));temporary.replace(path)


def run(model='gpt-4o-mini'):
    from laiwenai import LaiwenAI
    parent=Path('artifacts/source-counterfactual-demand-ranked-local-v4')
    jobs={j['id']:j for _,j in records(parent/'jobs.jsonl')}
    cases=[]
    for path in sorted((parent/'generation-only').glob('*source-cf*.json')):
        result=json.loads(path.read_text())
        if result['status']=='constructed_pending_independent_audit':
            cases.append(('demand-ranked',path,jobs[result['id']],result))
    for path in sorted(Path('artifacts/witness-first-controls-v1').glob('*/result.json')):
        result=json.loads(path.read_text())
        if result['status']=='constructed_pending_independent_audit':
            cases.append(('witness-first',path,result['job'],result))
    out=Path('artifacts/witness-control-auxiliary-audit-v1' if model=='gpt-4o-mini' else 'artifacts/witness-control-audit-'+model+'-v1');out.mkdir(exist_ok=True)
    spec=dict(version=1,model=model,auditor=AUDITOR,
        openai=importlib.metadata.version('openai'),
        parents={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for _,p,_,_ in cases},
        warning='Auxiliary same-model-family review only. Does not replace the required GPT-4o/GPT-6-Luna audit, supply human gold, or establish industrial-fit labels. All constructed cases included; expected labels and planner reasoning hidden.')
    marker=out/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=spec:raise ValueError('Changed auxiliary audit inputs')
    write(marker,spec);client=LaiwenAI(model=spec['model'],max_retries=0);rows=[]
    for route,path,job,original in cases:
        example=original['example'];views=build_views(job,example)
        if views!=original['views']:raise ValueError('Changed source-control views')
        case_id=route+'-'+job['id'];order=['full','control'];random.Random(case_id).shuffle(order)
        audited={}
        for view in order:
            sources=views[view]
            payload=dict(original_demand=job['demand'],conditions=dict(a=example['topics'][0],b=example['topics'][1],relation=example['relation']),
                sources=[{k:s.get(k) for k in ('id','kind','field','title','text')} for s in sources.values()])
            fingerprint=hashlib.sha256(json.dumps([spec,payload],sort_keys=True).encode()).hexdigest()
            target=out/(case_id+'-'+view+'.json')
            if target.exists():
                result=json.loads(target.read_text())
                if result['fingerprint']!=fingerprint:raise ValueError('Changed auxiliary request')
            else:
                result=dict(fingerprint=fingerprint,status='unresolved',route=route,view=view)
                start=time.perf_counter()
                try:
                    response=client.chat.completions.create(messages=[
                        dict(role='system',content=AUDITOR+' Return one JSON object.'),
                        dict(role='user',content=json.dumps(payload,ensure_ascii=False))],response_format=dict(type='json_object'),**({'reasoning_effort':'none'} if model=='gpt-6-luna' else {}))
                    result['response']=response.model_dump(mode='json')
                    verdict=json.loads(response.choices[0].message.content)
                    labels,related=validate_judgments(verdict,sources,job['demand'])
                    result.update(status='valid',labels=labels,demand_related=related)
                except Exception as error:
                    result.update(error_type=type(error).__name__,http_status=getattr(error,'status_code',None))
                result['seconds']=time.perf_counter()-start;write(target,result)
            audited[view]=result
        expected={'full':dict(a='supported',b='supported',relation='supported'),
                  'control':dict(a='supported',b='supported',relation='unknown')}
        status=('unresolved' if any(a['status']!='valid' for a in audited.values()) else
            'auxiliary_agreement_not_final' if all(a['labels']==expected[v] and a['demand_related'] for v,a in audited.items()) else 'auxiliary_disagreement')
        rows.append(dict(id=case_id,status=status,views={v:{k:a.get(k) for k in ('status','labels','demand_related','error_type')} for v,a in audited.items()}))
        print(case_id,status,flush=True)
    report=dict(cases=len(cases),rows=rows,accepted_dataset_labels=0,warning=spec['warning'])
    write(out/'summary.json',report);write(Path('docs/exploration/witness-control-auxiliary-audit.json' if model=='gpt-4o-mini' else 'docs/exploration/witness-control-audit-'+model+'.json'),report)
    return report


if __name__=='__main__':
    import argparse
    from dotenv import load_dotenv
    load_dotenv('.env')
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=['gpt-4o-mini','gpt-4o','gpt-6-luna'],default='gpt-4o-mini')
    print(json.dumps(run(parser.parse_args().model),indent=2))
