"""Fixed-sample query-only decomposition; no qrels or answers sent to the model."""
import hashlib
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from laiwenai import LaiwenAI
from .agent import ExpertAgent
from .corpus import records

INSTRUCTION=('Decompose a retrieval query only when it asks for 2–4 conjunctive attributes of the SAME '
             'target entity/document. Return {"decomposable":bool,"facets":[str],"reason":str}. Each facet '
             'must be a standalone short retrieval query, keeping the shared target type and its specific '
             'attribute. Preserve ALL conditions, modifiers, entities and relation directions. Do not add '
             'facts or answer the query. Do not turn a multi-hop relational chain into independent attributes. '
             'If this cannot be done safely, set decomposable false and facets [].')


def build(limit=100,output='artifacts/public/comlq/conjunction-plans'):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    source=Path('artifacts/public/comlq/adapted/queries.jsonl')
    # Type is used to define the evaluation slice only, never as model input.
    eligible=sorted((r for _,r in records(source) if r.get('query_type','').split('_')[0] in ('2i','3i')),key=lambda r:r['id'])
    selected=random.Random(42).sample(eligible,min(limit,len(eligible)))
    spec=dict(seed=42,eligible=len(eligible),query_ids=[r['id'] for r in selected],
        input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),model='gpt-6-luna',reasoning_effort='none',
        instruction=INSTRUCTION,scope='Pure conjunction slice selected without relevance outcomes; type not sent to planner.')
    marker=out/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=spec:raise ValueError('Plan protocol changed')
    marker.write_text(json.dumps(spec,indent=2))
    def one(query):
        path=out/(query['id']+'.json')
        if path.exists():return json.loads(path.read_text())['status']
        agent=ExpertAgent.__new__(ExpertAgent);agent.client=LaiwenAI(model='gpt-6-luna')
        agent.trace=[];agent.usage={'prompt_tokens':0,'completion_tokens':0};agent.request_options={'reasoning_effort':'none'}
        start=time.perf_counter()
        result=dict(query_id=query['id'],query=query['query'])
        try:
            plan=agent.ask('conjunction_plan',INSTRUCTION,{'query':query['query']})
            if type(plan.get('decomposable'))!=bool or not isinstance(plan.get('facets'),list):raise ValueError('Invalid plan schema')
            facets=plan['facets']
            if plan['decomposable'] and (not 2<=len(facets)<=4 or any(not isinstance(f,str) or not f.strip() for f in facets) or len(set(facets))!=len(facets)):
                raise ValueError('Invalid conjunctive facets')
            if not plan['decomposable'] and facets:raise ValueError('Rejected plan has facets')
            result.update(status='ok',plan=plan)
        except Exception as error:result.update(status='error',error=dict(type=type(error).__name__,message=str(error)[:200]))
        result.update(seconds=time.perf_counter()-start,trace=agent.trace,usage=agent.usage)
        path.write_text(json.dumps(result,ensure_ascii=False,indent=2));return result['status']
    with ThreadPoolExecutor(max_workers=2) as pool:statuses=list(pool.map(one,selected))
    summary=dict(requested=len(selected),statuses={s:statuses.count(s) for s in set(statuses)},
                 warning='Model-produced decomposition, not verified semantic equivalence. Selected conjunction slice only; not all ComLQ queries.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2));return summary


if __name__=='__main__':print(json.dumps(build(),indent=2))
