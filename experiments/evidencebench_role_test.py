"""Frozen original-test check of the development-derived source-role rule."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from expert_search.evidencebench_data import prepare, aspect_coverage
from expert_search.source_roles import normalize_roles, prioritize_roles, ROLES
from expert_search.statistics import paired_interval
from evidencebench_role_index import PROMPT
from evidencebench_pilot import TASK

ROOT=Path('artifacts/evidencebench-role-test-v1')


def freeze(data):
    ROOT.mkdir(exist_ok=True)
    spec=dict(split='original_test',records=len(data),model='gpt-6-luna',prompt=PROMPT,
              dataset_sha256=hashlib.sha256(Path('artifacts/public/evidencebench/test.json').read_bytes()).hexdigest(),
              source_sha256={k:hashlib.sha256(json.dumps(r['paper_as_candidate_pool'],ensure_ascii=False).encode()).hexdigest() for k,r in data.items()},
              code_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['experiments/evidencebench_role_test.py','src/expert_search/source_roles.py','src/expert_search/qwen_embedding_torch.py']},
              rule='Exactly the development-derived six-role source-only compilation. Within-role duplicates deduplicate; conflicts and omissions uncertain; malformed/foreign IDs raw fallback. Stable role preference over full-pool Qwen ranking, top10. No retries, no labels/queries in compilation.',
              embedding=dict(backend='torch',max_length=512,batch_size=8,task=TASK),
              scope='Original293 test check after37 development; not enterprise expert suitability. Fixed single contrast versus same-query Qwen. No parameter search or claim of independent human gold. Offline time includes all compile calls, no observed reuse amortization.')
    path=ROOT/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=spec:raise ValueError('Changed test protocol')
    path.write_text(json.dumps(spec,indent=2))


def compile_one(item):
    from laiwenai import LaiwenAI
    key,sentences=item
    path=ROOT/'roles'/(key+'.json')
    if path.exists():return json.loads(path.read_text())
    start=time.perf_counter();result=dict(id=key,status='fallback')
    try:
        client=LaiwenAI(model='gpt-6-luna',max_retries=0)
        response=client.chat.completions.create(messages=[dict(role='system',content=PROMPT),dict(role='user',content=json.dumps(dict(sentences=dict(enumerate(sentences))),ensure_ascii=False))],reasoning_effort='none',response_format=dict(type='json_object'),max_completion_tokens=8192)
        result['response']=response.model_dump(mode='json')
        result.update(normalize_roles(json.loads(response.choices[0].message.content),len(sentences)))
        result['status']='normalized'
    except Exception as error:result.update(error_type=type(error).__name__,http_status=getattr(error,'status_code',None))
    result['offline_seconds']=time.perf_counter()-start
    temp=path.with_suffix('.partial');temp.write_text(json.dumps(result,indent=2));temp.replace(path)
    return result


def main(stage):
    data=prepare('test');freeze(data)
    if stage=='freeze':return
    if stage=='compile':
        (ROOT/'roles').mkdir(exist_ok=True)
        # API requests, not agent delegation. Query and labels never enter workers.
        with ThreadPoolExecutor(max_workers=4) as pool:
            for result in pool.map(compile_one,[(key,row['paper_as_candidate_pool']) for key,row in data.items()]):
                print(result['id'],result['status'],round(result['offline_seconds'],3),flush=True)
        return
    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    model=QwenTorchEmbedding(max_length=512,batch_size=8)
    (ROOT/'vectors').mkdir(exist_ok=True)
    if stage=='vectors':
        for key,row in data.items():
            path=ROOT/'vectors'/(key+'.npy')
            if not path.exists():
                vectors=model.encode([s if s.strip() else '[empty sentence]' for s in row['paper_as_candidate_pool']])
                temp=path.with_suffix('.partial')
                with temp.open('wb') as stream:np.save(stream,vectors)
                temp.replace(path)
            print(key,'vectors',flush=True)
        return
    rows=[]
    for key,row in data.items():
        source=json.loads((ROOT/'roles'/(key+'.json')).read_text())
        vectors=np.load(ROOT/'vectors'/(key+'.npy'),allow_pickle=False)
        if vectors.shape!=(len(row['paper_as_candidate_pool']),1024) or not np.isfinite(vectors).all():raise ValueError('Invalid test vectors')
        roles=source.get('roles',{r:[] for r in ROLES})
        start=time.perf_counter()
        q=model.encode(['Instruct: '+TASK+'\nQuery: '+row['hypothesis']])[0]
        scores=vectors@q;rank=sorted(range(len(scores)),key=lambda i:(-float(scores[i]),i))
        selected=prioritize_roles(rank,roles)[:10]
        seconds=time.perf_counter()-start
        rows.append(dict(id=key,status=source['status'],selected=selected,baseline_selected=rank[:10],metrics=aspect_coverage(row,selected),baseline=aspect_coverage(row,rank[:10]),online_seconds=seconds,offline_seconds=source['offline_seconds']))
    summary=dict(papers=len(rows),normalized=sum(r['status']=='normalized' for r in rows),aspect_recall=float(np.mean([r['metrics']['aspect_recall'] for r in rows])),baseline=float(np.mean([r['baseline']['aspect_recall'] for r in rows])),difference=paired_interval([r['metrics']['aspect_recall']-r['baseline']['aspect_recall'] for r in rows]),mean_online_seconds=float(np.mean([r['online_seconds'] for r in rows])),sum_offline_request_seconds=sum(r['offline_seconds'] for r in rows),warning='Four concurrent offline workers: summed request durations are not wall time. Online includes fresh query encoding, sorting and role preference, excludes initialization/source vectors. No enterprise-fit, new metric, attribution correctness or full-agent latency claim.')
    (ROOT/'evaluation.json').write_text(json.dumps(rows,indent=2));(ROOT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['freeze','compile','vectors','evaluate'],required=True)
    args=parser.parse_args()
    if args.stage=='compile':
        from dotenv import load_dotenv
        load_dotenv('.env')
    main(args.stage)
