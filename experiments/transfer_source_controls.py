"""Same-owner real-paper substitutions for capability facts, not inability labels."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import time
from expert_search.expert_catalog import ExpertCatalog
from expert_search.decisions_backend import DecisionsBackend
from expert_search.source_counterfactual import normalized

ROOT=Path('artifacts/demand-transfer-controls-v1')


def screen(item):
    job,proposal,control=item;path=ROOT/(job['id']+'.json')
    if path.exists():return json.loads(path.read_text())
    from laiwenai import LaiwenAI
    result=dict(id=job['id'],**control)
    if control['status']!='ready':
        path.write_text(json.dumps(result,indent=2,ensure_ascii=False));return result
    judge=DecisionsBackend(LaiwenAI(model='gpt-6-luna',max_retries=0));order=['full','control'];random.Random(job['id']).shuffle(order)
    start=time.perf_counter();judgments={}
    constraint='At least one supplied paper documents this complete capability as its own actual work, without combining independent papers into an already demonstrated joint method: '+proposal['capability']
    for view in order:
        try:
            response=judge.judge([dict(id='capability',constraint=constraint,text=json.dumps(control['views'][view],ensure_ascii=False))])
            judgments[view]=dict(status='valid',response=response)
        except Exception as error:judgments[view]=dict(status='unresolved',error_type=type(error).__name__,http_status=getattr(error,'status_code',None))
    result['judgments']=judgments;result['request_seconds']=time.perf_counter()-start
    if all(j['status']=='valid' for j in judgments.values()):
        labels={v:j['response']['judgments'][0]['label'] for v,j in judgments.items()}
        result.update(labels=labels,status='native_source_control_candidate' if labels==dict(full='supported',control='unknown') else 'source_control_disagreement')
    else:result['status']='unresolved'
    temp=path.with_suffix('.partial');temp.write_text(json.dumps(result,indent=2,ensure_ascii=False));temp.replace(path)
    print(job['parent_demand'],result['status'],result.get('labels'),flush=True);return result


def run():
    job_path=Path('artifacts/demand-first-paper-jobs-v1/jobs.jsonl');jobs=[json.loads(s) for s in job_path.read_text().splitlines()]
    generation=Path('artifacts/demand-transfer-generation-v1');audit_root=Path('artifacts/demand-transfer-audit-v1')
    candidates=[job for job in jobs if json.loads((audit_root/(job['id']+'.json')).read_text()).get('status')=='model_screened_transfer_candidate'] if all((audit_root/(job['id']+'.json')).exists() for job in jobs) else []
    # Generation refusals have no audit file, and are retained in the generation funnel.
    candidates=[job for job in jobs if (audit_root/(job['id']+'.json')).exists() and json.loads((audit_root/(job['id']+'.json')).read_text())['status']=='model_screened_transfer_candidate']
    catalog=ExpertCatalog();inventory=catalog.for_experts([job['expert_id'] for job in candidates]);by_owner={}
    for source in inventory:
        if source['kind']=='paper':by_owner.setdefault(source['expert_id'],[]).append(source)
    ROOT.mkdir(exist_ok=True);items=[]
    for job in candidates:
        proposal=json.loads((generation/(job['id']+'.json')).read_text());sid=proposal['citations'][0]['evidence_id'];witness=job['sources'][sid]
        alias=lambda s:normalized(s['text'])==normalized(witness['text']) or normalized(s['text'].splitlines()[0])==normalized(witness['text'].splitlines()[0])
        removed={key for key,source in job['sources'].items() if alias(source)}
        replacements=[dict(s,text=s['text'][:1200]) for s in by_owner.get(job['expert_id'],[]) if s['id'] not in job['sources'] and not alias(s)]
        replacements.sort(key=lambda s:(-len(s['text']),s['id']))
        result=dict(status='insufficient_real_replacements',removed=sorted(removed))
        if len(replacements)>=len(removed):
            control={key:s for key,s in job['sources'].items() if key not in removed};control.update({s['id']:s for s in replacements[:len(removed)]})
            if len(control)!=len(job['sources']) or any(s['expert_id']!=job['expert_id'] for s in control.values()):raise ValueError('Control changed count or owner')
            if sum(len(s['text']) for s in control.values())>=.8*sum(len(s['text']) for s in job['sources'].values()):
                result.update(status='ready',views=dict(full=job['sources'],control=control))
            else:result['status']='replacement_too_short'
        items.append((job,proposal,result))
    protocol=dict(model='gpt-6-luna',parent_sha256=hashlib.sha256(job_path.read_bytes()).hexdigest(),
                  source_snapshot=json.loads(Path('artifacts/expert-catalog-v1/manifest.json').read_text()),
                  audit_parents={job['id']:hashlib.sha256((audit_root/(job['id']+'.json')).read_bytes()).hexdigest() for job in candidates},
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  rule='All screened transfer candidates. Exact text/title-first-line alias closure of one witness paper; replace with longest real outside-view same-owner paper prefixes. Same document count, >=80percent total characters. Random order, separate native fact requests,3workers,no retries.',
                  scope='Capability-fact controls only; conditional target bridge is not a source fact. Unknown is missing support, not expert inability or unfitness. Exact aliases do not prove semantic-duplicate removal. One model, no independent/human gold.')
    marker=ROOT/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=protocol:raise ValueError('Changed transfer control protocol')
    marker.write_text(json.dumps(protocol,indent=2))
    with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(screen,items))
    summary=dict(screened_candidates=len(items),statuses={s:sum(r['status']==s for r in results) for s in sorted({r['status'] for r in results})},warning='Native screened source controls, not accepted expert-fit gold. All failures/refusals retained, no human labels, same-model screening bias. No deployment or ranking accuracy inference.')
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':
    from dotenv import load_dotenv
    load_dotenv('.env');run()
