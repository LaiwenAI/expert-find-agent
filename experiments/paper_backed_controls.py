"""Matched same-owner source controls for four dual-model silver candidates."""
import hashlib
import json
import inspect
import random
from pathlib import Path
from expert_search.decisions_backend import DecisionsBackend
from expert_search.source_counterfactual import normalized


def run(input_path='artifacts/paper-backed-overlap-v1/jobs.jsonl',audit_path='artifacts/paper-backed-span-audit-v3',generation_path='artifacts/paper-backed-generation-v1',output='artifacts/paper-backed-controls-v1',replacement_corpus=None):
    source=Path(input_path);jobs={j['id']:j for j in map(json.loads,source.read_text().splitlines())}
    summary=json.loads((Path(audit_path)/'summary.json').read_text())
    out=Path(output);out.mkdir(exist_ok=True);rows=[];backend=DecisionsBackend()
    protocol=dict(version=1,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        audit_summary_sha256=hashlib.sha256(json.dumps(summary,sort_keys=True).encode()).hexdigest(),
        backend_sha256=hashlib.sha256(inspect.getsource(DecisionsBackend).encode()).hexdigest(),
        policy='Union of both reviewers cited papers and exact normalized title/text aliases removed; longest real outside-snapshot same-owner replacements. Same count, >=80 percent total chars; separate randomized-order requests.')
    replacement_sources=None
    if replacement_corpus is not None:
        from expert_search.expert_catalog import ExpertCatalog
        catalog=ExpertCatalog(replacement_corpus)
        evidence=Path(replacement_corpus)/'evidence.jsonl'
        with evidence.open('rb') as stream:
            protocol['replacement_snapshot_sha256']=hashlib.file_digest(stream,'sha256').hexdigest()
        accepted=[jobs[r['id']]['expert_id'] for r in summary['rows'] if r['status']=='dual_model_silver_candidate']
        replacement_sources={}
        for row in catalog.for_experts(accepted):
            if row.get('kind')=='paper':replacement_sources.setdefault(row['expert_id'],{})[row['id']]=row
    marker=out/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=protocol:raise ValueError('Changed control protocol')
    marker.write_text(json.dumps(protocol,indent=2))
    for record in summary['rows']:
        if record['status']!='dual_model_silver_candidate':continue
        key=record['id'];job=jobs[key];g=json.loads((Path(generation_path)/(key+'.json')).read_text())
        witness=set()
        for model in ('gpt-4o','gpt-6-luna'):
            a=json.loads((Path(audit_path)/(key+'-'+model+'.json')).read_text())['audit'];witness.add(a['evidence_id'])
        cited=[job['sources'][sid]['text'] for sid in witness]
        alias=lambda text:any(normalized(text)==normalized(c) or normalized(text.splitlines()[0])==normalized(c.splitlines()[0]) for c in cited)
        full=g['sources'];removed={sid for sid,e in full.items() if alias(job['sources'][sid]['text'])};remaining={sid:e for sid,e in full.items() if sid not in removed}
        inventory=job['sources'] if replacement_sources is None else replacement_sources.get(job['expert_id'],{})
        replacements=[dict(e,text=e['text'][:1200]) for sid,e in inventory.items() if sid not in full and not alias(e['text'])]
        replacements.sort(key=lambda e:(-len(e['text']),e['id']))
        result=dict(id=key,removed_ids=sorted(removed),status='unresolved')
        target=out/(key+'.json')
        if target.exists():
            rows.append(json.loads(target.read_text()));continue
        if len(replacements)<len(removed):result['status']='insufficient_real_replacements'
        else:
            control=dict(remaining);control.update({e['id']:e for e in replacements[:len(removed)]})
            if len(control)!=len(full) or any(e['expert_id']!=job['expert_id'] for e in control.values()):raise ValueError('Changed owner/count')
            if sum(len(e['text']) for e in control.values())<.8*sum(len(e['text']) for e in full.values()):result['status']='replacement_too_short'
            else:
                condition=g['proposal']['condition'];views=dict(full=full,control=control)
                result.update(condition=condition,views=views,input_sha256=hashlib.sha256(json.dumps([condition,views],sort_keys=True).encode()).hexdigest())
                order=['full','control'];random.Random(key).shuffle(order);judgments={}
                for view in order:
                    try:
                        response=backend.judge([dict(id='condition',constraint=condition,text=json.dumps(views[view],ensure_ascii=False))])
                        judgments[view]=dict(status='valid',response=response)
                    except Exception as error:judgments[view]=dict(status='unresolved',error_type=type(error).__name__,http_status=getattr(error,'status_code',None))
                result['judgments']=judgments
                if all(r['status']=='valid' for r in judgments.values()):
                    labels={v:r['response']['judgments'][0]['label'] for v,r in judgments.items()}
                    result.update(status='native_control_candidate' if labels==dict(full='supported',control='unknown') else 'native_control_disagreement',labels=labels)
        temporary=target.with_suffix('.partial');temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2));temporary.replace(target);rows.append(result)
        print(key,result['status'],result.get('labels'),flush=True)
    report=dict(candidates=len(rows),statuses={s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})},
        accepted_dataset_labels=0,warning='Single-model source-ablation feasibility; not human gold. Text/title alias closure does not prove semantic duplicate removal. Equal source count, same owner, at least 80% text length. No inability inference from missing support.')
    (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':
    from dotenv import load_dotenv
    load_dotenv('.env')
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--input',default='artifacts/paper-backed-overlap-v1/jobs.jsonl');parser.add_argument('--audit',default='artifacts/paper-backed-span-audit-v3');parser.add_argument('--generation',default='artifacts/paper-backed-generation-v1');parser.add_argument('--output',default='artifacts/paper-backed-controls-v1');parser.add_argument('--replacement-corpus');args=parser.parse_args();run(args.input,args.audit,args.generation,args.output,args.replacement_corpus)
