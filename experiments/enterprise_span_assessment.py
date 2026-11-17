"""Blind source-view comparison; model judgments are not independent gold."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import numpy as np
from expert_search.span_applicability import source_spans
from expert_search.span_view_applicability import SpanViewApplicability

ROOT=Path('artifacts/enterprise-span-assessment-v1')


def assess(item):
    key,payload=item;path=ROOT/'responses'/(key+'.json')
    if path.exists():return json.loads(path.read_text())
    from laiwenai import LaiwenAI
    judge=SpanViewApplicability(LaiwenAI(model='gpt-6-luna',max_retries=0))
    try:result=judge.assess_view(**payload)
    except Exception as error:result=dict(status='unresolved',error_type=type(error).__name__,http_status=getattr(error,'status_code',None))
    temp=path.with_suffix('.partial');temp.write_text(json.dumps(result,indent=2,ensure_ascii=False));temp.replace(path)
    print(key[:8],result['status'],flush=True)
    return result


def run():
    data_path=Path('artifacts/dataset-release/paper-controls-combined-v3/examples.jsonl')
    role_path=Path('artifacts/enterprise-role-budget-v1/evaluation.json')
    query_path=Path('artifacts/enterprise-query-budget-v1/evaluation.json')
    rows=[json.loads(s) for s in data_path.read_text().splitlines()]
    roles={r['id']:r for r in json.loads(role_path.read_text())}
    queries={r['id']:r for r in json.loads(query_path.read_text())}
    (ROOT/'responses').mkdir(parents=True,exist_ok=True)
    jobs={};views=[]
    for row in rows:
        key=hashlib.sha256(row['pair_id'].encode()).hexdigest()
        plan=json.loads((Path('artifacts/enterprise-query-budget-v1')/(key+'-plan.json')).read_text())
        if plan['status']!='planned':raise ValueError('Comparison requires the same successful source-blind plan')
        selection={**queries[row['id']]['selected'], 'role':next(r for r in json.loads(role_path.read_text()) if r['id']==row['id'] and r['query_kind']=='original_demand')['selected']['role'],
                   'full_source_view':list(source_spans(row['expert_id'],row['sources']).values())}
        for method,spans in selection.items():
            # Canonical ordering anonymizes retrieval order; only the view changes.
            spans=sorted(spans,key=lambda s:(s['evidence_id'],s['start'],s['end']))
            payload=dict(demand=row['original_demand'],expert_id=row['expert_id'],requirements=plan['plan']['requirements'],sources=row['sources'],spans={'p'+str(i):s for i,s in enumerate(spans)})
            digest=hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
            jobs[digest]=payload
            views.append(dict(id=row['id'],pair_id=row['pair_id'],view=row['view'],method=method,request=digest,spans=len(spans),chars=sum(len(s['quote']) for s in spans)))
    parents=[data_path,role_path,query_path,*Path('artifacts/enterprise-query-budget-v1').glob('*-plan.json')]
    protocol=dict(model='gpt-6-luna',parents={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in parents},
                  code_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['experiments/enterprise_span_assessment.py','src/expert_search/span_view_applicability.py','src/expert_search/span_applicability.py']},
                  request_hashes=sorted(jobs),views=views,
                  scope='Same original demand and source-blind max6 requirement plan,5 silver pairs/10 source views. Reviewer sees only selected original source spans, no method names, silver labels or citations. Full snapshot baseline uncompressed; other methods3 spans. Identical requests deduplicate, fixed seed shuffle,3 independent API workers. All failures retained; no retries. Structural correctness and provisional model coverage only, not gold or end-to-end expert search.')
    marker=ROOT/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=protocol:raise ValueError('Changed blind assessment protocol')
    marker.write_text(json.dumps(protocol,indent=2))
    order=sorted(jobs.items());random.Random(20261117).shuffle(order)
    with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(assess,order))
    results=[]
    for view in views:
        result=json.loads((ROOT/'responses'/(view['request']+'.json')).read_text())
        results.append(dict(view,outcome=result['status'],coverage=result.get('result',{}).get('actionable_coverage'),direct=result.get('result',{}).get('direct_coverage'),transfer=result.get('result',{}).get('transfer_coverage'),request_seconds=result.get('seconds')))
    methods=sorted({r['method'] for r in results})
    summary=dict(views=len(rows),pairs=len({r['pair_id'] for r in rows}),unique_requests=len(jobs),comparisons=len(results),
                 methods={m:dict(comparisons=sum(r['method']==m for r in results),valid=sum(r['method']==m and r['outcome']=='structurally_valid' for r in results),
                                 valid_only_mean_coverage=float(np.mean([r['coverage'] for r in results if r['method']==m and r['coverage'] is not None])) if any(r['method']==m and r['coverage'] is not None for r in results) else None,
                                 mean_input_chars=float(np.mean([r['chars'] for r in results if r['method']==m]))) for m in methods},
                 warning='Provisional ordinal model coverage, not expert accuracy or probability. Same provider as silver generation; incomplete plans may inflate coverage. Valid-only means accompanied by failure counts, no quality comparison if failures differ. Deduplicated historical calls and concurrent requests do not measure paired interface speed or full agent latency.')
    (ROOT/'evaluation.json').write_text(json.dumps(results,indent=2));(ROOT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':
    from dotenv import load_dotenv
    load_dotenv('.env');run()
