"""Fixed-input, alternating compact singleton/batch model assessment comparison."""
import hashlib
import json
import time
from pathlib import Path
from expert_search.batch_applicability import BatchApplicability
from expert_search.expert_catalog import ExpertCatalog


def run():
    parent=Path('artifacts/grounded-pipeline-integration-v1');files=sorted(parent.glob('demand-*.json'))
    jobs={r['id']:r for r in map(json.loads,Path('artifacts/paper-backed-overlap-v1/jobs.jsonl').read_text().splitlines())}
    catalog=ExpertCatalog();cases=[]
    for p in files:
        saved=json.loads(p.read_text());events=saved['events'];requirements=next(e['plan']['requirements'] for e in events if e['stage']=='planning')
        hits=next(e['route']['hits'] for e in events if e['stage']=='retrieval');sources={s['id']:s for s in catalog.for_experts([h['id'] for h in hits])};candidates={}
        for h in hits:
            candidates[h['id']]={}
            for witness in h['evidence']:
                sid=witness['id'];s=sources[sid]
                if s['expert_id']!=h['id']:raise ValueError('Changed source owner')
                candidates[h['id']][sid]=dict(id=sid,expert_id=h['id'],text=s['text'][:1200])
        cases.append(dict(id=saved['id'],demand=jobs[saved['id']]['demand'],requirements=requirements,candidates=candidates))
    out=Path('artifacts/compact-assessment-comparison-v1');out.mkdir(exist_ok=True)
    protocol=dict(version=1,model='gpt-6-luna',repetitions=2,inputs_sha256=hashlib.sha256(json.dumps(cases,sort_keys=True).encode()).hexdigest(),
        parents={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        code_sha256={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in ('src/expert_search/batch_applicability.py','src/expert_search/span_applicability.py',__file__)},
        design='Two saved demands, fixed exact plans/candidates/source prefixes. Alternating singleton/batch order by query and repetition. Same compact prompt and limits. No expected labels in requests. Historical verbose timing is not paired.')
    marker=out/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=protocol:raise ValueError('Changed compact comparison inputs')
    marker.write_text(json.dumps(protocol,indent=2));assessor=BatchApplicability();rows=[]
    for ci,case in enumerate(cases):
        for repeat in range(2):
            modes=('singleton','batch') if (ci+repeat)%2==0 else ('batch','singleton')
            for mode in modes:
                target=out/(case['id']+'-'+str(repeat)+'-'+mode+'.json')
                if target.exists():row=json.loads(target.read_text())
                else:
                    row=dict(id=case['id'],repeat=repeat,mode=mode,status='unresolved');start=time.perf_counter();calls=[];outcomes={}
                    try:
                        groups=[case['candidates']] if mode=='batch' else [{eid:s} for eid,s in case['candidates'].items()]
                        for group in groups:
                            call=assessor.assess_many(case['demand'],case['requirements'],group);calls.append(call);outcomes.update(call['outcomes'])
                        row.update(status='executed',calls=calls,outcomes=outcomes)
                    except Exception as error:row.update(error_type=type(error).__name__,http_status=getattr(error,'status_code',None),calls=calls,outcomes=outcomes)
                    row['wall_seconds']=time.perf_counter()-start;temp=target.with_suffix('.partial');temp.write_text(json.dumps(row,ensure_ascii=False,indent=2));temp.replace(target)
                rows.append(row);print(case['id'],repeat,mode,row['status'],round(row['wall_seconds'],3),flush=True)
    paired=[]
    for case in cases:
        for repeat in range(2):
            a,b=[next(r for r in rows if r['id']==case['id'] and r['repeat']==repeat and r['mode']==m) for m in ('singleton','batch')]
            agreements=0;comparable=0
            for eid in case['candidates']:
                aa,bb=a.get('outcomes',{}).get(eid,{}),b.get('outcomes',{}).get(eid,{})
                if aa.get('status')!= 'structurally_valid' or bb.get('status')!='structurally_valid':continue
                for rid in case['requirements']:
                    comparable+=1;agreements+=aa['result']['assessments'][rid]['level']==bb['result']['assessments'][rid]['level']
            paired.append(dict(id=case['id'],repeat=repeat,singleton_seconds=a['wall_seconds'],batch_seconds=b['wall_seconds'],comparable_labels=comparable,agreeing_labels=agreements))
    report=dict(runs=len(rows),executed=sum(r['status']=='executed' for r in rows),
        modes={mode:dict(total_wall_seconds=sum(r['wall_seconds'] for r in rows if r['mode']==mode),structurally_valid_candidates=sum(o['status']=='structurally_valid' for r in rows if r['mode']==mode for o in r.get('outcomes',{}).values()),completion_tokens=sum(call.get('response',{}).get('usage',{}).get('completion_tokens',0) for r in rows if r['mode']==mode for call in r.get('calls',[]))) for mode in ('singleton','batch')},paired=paired,
        warning='Fixed source support views but no applicability gold. Label agreement is not accuracy/equivalence. Compact limits and shared context may change judgments. Only two reused development queries, two trials; no production latency claim. Planning/encoding excluded.')
    (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':
    from dotenv import load_dotenv
    load_dotenv('.env');run()
