"""Real-source evidence-budget diagnostic, not expert-ranking ground truth."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from expert_search.source_roles import normalize_roles, prioritize_roles, ROLES
from expert_search.span_applicability import source_spans
from evidencebench_role_index import PROMPT

ROOT=Path('artifacts/enterprise-role-budget-v1')
DATA=Path('artifacts/dataset-release/paper-controls-combined-v3/examples.jsonl')
TASK='Find researchers whose documented research background could help address the technical requirement.'
INSTRUCTION=PROMPT+' Here the numbered items are consecutive 400-character source fragments, possibly incomplete sentences, from one publication or profile. Use uncertain for mixed or ambiguous attribution. Only the supplied prefix is available; do not assume omitted content.'


def compile_source(item):
    key,source=item;target=ROOT/'roles'/(key+'.json')
    if target.exists():return json.loads(target.read_text())
    from laiwenai import LaiwenAI
    fragments=[source['text'][i:i+400] for i in range(0,len(source['text']),400)]
    start=time.perf_counter();result=dict(id=key,status='fallback')
    try:
        client=LaiwenAI(model='gpt-6-luna',max_retries=0)
        response=client.chat.completions.create(messages=[dict(role='system',content=INSTRUCTION),dict(role='user',content=json.dumps(dict(sentences=dict(enumerate(fragments))),ensure_ascii=False))],reasoning_effort='none',response_format=dict(type='json_object'),max_completion_tokens=1024)
        result['response']=response.model_dump(mode='json')
        result.update(normalize_roles(json.loads(response.choices[0].message.content),len(fragments)));result['status']='normalized'
    except Exception as error:result.update(error_type=type(error).__name__,http_status=getattr(error,'status_code',None))
    result['offline_seconds']=time.perf_counter()-start
    temp=target.with_suffix('.partial');temp.write_text(json.dumps(result,indent=2));temp.replace(target)
    return result


def identity(source):
    return hashlib.sha256(json.dumps([source['expert_id'],source['id'],source['text']],ensure_ascii=False).encode()).hexdigest()


def citation_retention(row,spans,selected):
    """Silver quote character retention only; never future project fit."""
    results={}
    for reviewer,audit in row['audits'].items():
        citation=audit['judgment'].get('citation')
        if row['label']!='supported' or not citation:continue
        sid=citation['evidence_id'];start,end=citation['start'],citation['end']
        if row['sources'][sid]['text'][start:end]!=citation['text'] or start>=end:raise ValueError('Changed silver citation')
        covered=set()
        for i in selected:
            span=spans[i]
            if span['evidence_id']==sid:covered.update(range(max(start,span['start']),min(end,span['end'])))
        results[reviewer]=dict(fraction=len(covered)/(end-start),complete=len(covered)==end-start)
    return results


def run():
    rows=[json.loads(line) for line in DATA.read_text().splitlines()]
    inventory={identity(source):source for row in rows for source in row['sources'].values()}
    (ROOT/'roles').mkdir(parents=True,exist_ok=True)
    protocol=dict(dataset_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest(),source_ids=sorted(inventory),model='gpt-6-luna',prompt=INSTRUCTION,
                  code_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['experiments/enterprise_role_budget.py','src/expert_search/source_roles.py','src/expert_search/span_applicability.py']},
                  role_compile_input='One source prefix only; never query, conditions, silver citations or labels.',
                  width=400,k=3,embedding=dict(task=TASK,max_length=4096,batch_size=8,backend='mlx'),
                  queries=['original_demand','condition'],scope='All10 existing development silver source views,5 pairs. Candidate expert supplied by construction; no expert-retrieval/ability accuracy claim. Exact cited-character retention, not semantic support. Both query versions fixed before output; narrow condition is a diagnostic, not a deployment oracle.')
    marker=ROOT/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=protocol:raise ValueError('Changed enterprise role-budget protocol')
    marker.write_text(json.dumps(protocol,indent=2))
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(compile_source,sorted(inventory.items())):print(result['id'][:8],result['status'],flush=True)
    from expert_search.qwen_embedding import QwenEmbedding
    model=QwenEmbedding(max_length=4096,batch_size=8)
    fragments={};roles={};texts=[];refs=[]
    for key,source in sorted(inventory.items()):
        result=json.loads((ROOT/'roles'/(key+'.json')).read_text())
        rolemap={i:role for role,ids in result.get('roles',{r:[] for r in ROLES}).items() for i in ids}
        for i,start in enumerate(range(0,len(source['text']),400)):
            fragments[(key,start)]=len(texts);texts.append(source['text'][start:start+400] or '[empty]');refs.append((key,start));roles[(key,start)]=rolemap.get(i,'uncertain')
    start=time.perf_counter();vectors=model.encode(texts);offline_vector_seconds=time.perf_counter()-start
    results=[]
    for row in rows:
        spans=list(source_spans(row['expert_id'],row['sources']).values())
        indices=[fragments[(identity(row['sources'][s['evidence_id']]),s['start'])] for s in spans]
        local=vectors[indices];role_groups={r:[] for r in ROLES}
        for i,s in enumerate(spans):role_groups[roles[(identity(row['sources'][s['evidence_id']]),s['start'])]].append(i)
        for query_kind in protocol['queries']:
            start=time.perf_counter();q=model.encode(['Instruct: '+TASK+'\nQuery: '+row[query_kind]])[0]
            scores=local@q;rank=sorted(range(len(spans)),key=lambda i:(-float(scores[i]),i));selected=dict(qwen=rank[:3],role=prioritize_roles(rank,role_groups)[:3])
            online_seconds=time.perf_counter()-start
            results.append(dict(id=row['id'],pair_id=row['pair_id'],view=row['view'],query_kind=query_kind,online_seconds=online_seconds,
                                selected={m:[dict(spans[i]) for i in ids] for m,ids in selected.items()},
                                citation_retention={m:citation_retention(row,spans,ids) for m,ids in selected.items()}))
    report=dict(views=len(rows),pairs=len({r['pair_id'] for r in rows}),unique_sources=len(inventory),source_fragments=len(texts),offline_vector_seconds=offline_vector_seconds,
                offline_request_seconds=sum(json.loads((ROOT/'roles'/(key+'.json')).read_text())['offline_seconds'] for key in inventory),
                mean_shared_online_seconds=float(np.mean([r['online_seconds'] for r in results])),
                retention={q:{m:dict(reviewer_quotes=sum(len(r['citation_retention'][m]) for r in results if r['query_kind']==q),complete=sum(a['complete'] for r in results if r['query_kind']==q for a in r['citation_retention'][m].values()),mean_fraction=float(np.mean([a['fraction'] for r in results if r['query_kind']==q for a in r['citation_retention'][m].values()]))) for m in ['qwen','role']} for q in protocol['queries']},
                warning='Two reviewers per supported view are correlated silver quotes, not independent cases. No human labels, no expert ranking or overall enterprise precision. Shared encoding/timing for both orders, no separate latency speedup. Offline compilation input is query-blind but source cohort is selected by prior synthetic construction.')
    (ROOT/'evaluation.json').write_text(json.dumps(results,indent=2,ensure_ascii=False));(ROOT/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':
    from dotenv import load_dotenv
    load_dotenv('.env');run()
