"""Factorial serialization/choice-order controls plus known synthetic contrasts."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random
import time
from expert_search.choice_judgments import decide
from .audit_disagreement import EQUIVALENCE, GROUNDING, CHOICES

ROOT = Path('artifacts/bilingual-order-controls-v1')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def synthetic_cases():
    templates = [
        ('\u5bfb\u627e\u539a\u5ea6\u81f3\u5c115 mm\u7684\u52a0\u5de5\u65b9\u6cd5\u3002', 'Find a processing method for thickness at least 5 mm.', 'Find a processing method for thickness at most 5 mm.'),
        ('\u5bfb\u627e\u538b\u529b\u4e0d\u8d85\u8fc710 MPa\u7684\u63a7\u5236\u65b9\u6cd5\u3002', 'Find a control method for pressure no greater than 10 MPa.', 'Find a control method for pressure no greater than 100 MPa.'),
        ('\u5bfb\u627e\u6d53\u5ea6\u81f3\u5c112 mM\u7684\u6eb6\u6db2\u68c0\u6d4b\u65b9\u6cd5\u3002', 'Find a solution detection method for a concentration of at least 2 mM.', 'Find a solution detection method for a concentration of at least 2 mm.'),
        ('\u5bfb\u627e\u6548\u7387\u81f3\u5c1195%\u7684\u8ba1\u7b97\u65b9\u6cd5\u3002', 'Find a calculation method with efficiency of at least 95%.', 'Find a calculation method with efficiency of at least 59%.'),
        ('\u5bfb\u627e\u6c34\u6d41\u6d4b\u91cf\u65b9\u6cd5\u3002', 'Find a method for measuring water flow.', 'Find a method for measuring oil flow.'),
        ('\u5bfb\u627e\u6fc0\u5149\u5207\u5272\u65b9\u6cd5\u3002', 'Find a laser cutting method.', 'Find a laser welding method.'),
    ]
    cases = []
    for index,(zh,correct,changed) in enumerate(templates):
        for variant,en,expected in [('faithful',correct,'supported'),('changed',changed,'changed_or_unsupported')]:
            payload = dict(intent_type='component_method',original_need=zh,cited_segments=[dict(id='s0',start=0,end=len(zh),zh=zh)],zh=zh,en=en)
            cases.append(dict(id='control-'+str(index)+'-'+variant,group='synthetic',stratum='known_'+variant,payload=payload,expected=expected))
    return cases


def request(case,choice_order,record_order):
    keys = ['intent_type','original_need','cited_segments','zh','en'] if record_order==0 else ['en','zh','intent_type','cited_segments','original_need']
    payload = {key:case['payload'][key] for key in keys}
    choices = [CHOICES[i] for i in choice_order]
    qs = [dict(type='choice',name='equivalence',instructions=EQUIVALENCE,choices=choices),dict(type='choice',name='grounding',instructions=GROUNDING,choices=choices)]
    return payload,qs


def prepare(controls_only):
    cases = synthetic_cases()
    input_hashes = {}
    if not controls_only:
        source = Path('artifacts/bilingual-disagreement-audit-v1/jobs.jsonl')
        original = list(map(json.loads,source.read_text().splitlines()))
        for kind in sorted({r['intent_type'] for r in original}):
            for stratum in ['low','middle','high']:
                pool = sorted([r for r in original if r['intent_type']==kind and r['stratum']==stratum],key=lambda r:r['id'])[:2]
                if len(pool)!=2:raise ValueError('Incomplete real factorial selection')
                cases.extend(dict(id=r['id'],group='real',stratum=stratum,payload=r['payload'],expected=None) for r in pool)
        input_hashes[str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
    tasks = []
    for case in cases:
        permutations = list(itertools.permutations(range(3))) if case['group']=='synthetic' else [(0,1,2),(2,1,0)]
        repeats = range(1) if case['group']=='synthetic' else range(2)
        for order,record,repeat in itertools.product(permutations,[0,1],repeats):
            payload,qs=request(case,order,record)
            identity=dict(case_id=case['id'],choice_order=list(order),record_order=record,repeat=repeat)
            tasks.append(dict(**identity,key=digest(identity)[:24],input_sha256=digest(dict(model='gpt-6-luna',input=json.dumps(payload,ensure_ascii=False),questions=qs))))
    random.Random(20261117).shuffle(tasks)
    protocol = dict(model='gpt-6-luna',controls_only=controls_only,real_cases=sum(r['group']=='real' for r in cases),synthetic_cases=12,requests=len(tasks),real_design='30cases:2 per intent/overlap stratum;2choice orders x2record orders x2repeat blocks. Questions fixed. All8requests contemporaneous, shuffled together.',synthetic_design='6faithful and6deliberately changed technical pairs; all6choice permutations x2record orders,144requests. Expected labels excluded from payloads. No uncertain-gold controls.',choices=CHOICES,instructions=dict(equivalence=EQUIVALENCE,grounding=GROUNDING),workers=3,sdk_retries=0,input_sha256=input_hashes,cases_sha256=digest(cases),tasks_sha256=digest(tasks),code_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('experiments/bilingual/audit_disagreement.py'),Path('src/expert_search/choice_judgments.py')]},scope='Endpoint workflow sensitivity on one configured provider/model; no claim about hidden implementation or universal Decisions behavior. Known contrasts test basic scope/quantity/object changes, not real enterprise translation quality. Factorial real cases are selected diagnostics, not population estimates or human gold.')
    ROOT.mkdir(exist_ok=True)
    marker=ROOT/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=protocol:raise ValueError('Changed frozen ordering control')
    marker.write_text(json.dumps(protocol,indent=2)+'\n')
    (ROOT/'cases.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in cases))
    (ROOT/'tasks.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in tasks))
    (ROOT/'predictions').mkdir(exist_ok=True)
    return {r['id']:r for r in cases},tasks


def one(task,cases):
    path=ROOT/'predictions'/(task['key']+'.json')
    if path.exists():
        old=json.loads(path.read_text())
        if old['input_sha256']!=task['input_sha256']:raise ValueError('Stale order-control result')
        return old
    from laiwenai import LaiwenAI
    client=LaiwenAI(model='gpt-6-luna',max_retries=0)
    payload,qs=request(cases[task['case_id']],task['choice_order'],task['record_order'])
    if digest(dict(model='gpt-6-luna',input=json.dumps(payload,ensure_ascii=False),questions=qs))!=task['input_sha256']:raise ValueError('Changed audit request')
    row=dict(**task,status='unresolved');start=time.perf_counter()
    try:
        result=decide(client,payload,qs);row.update(status='valid',labels=result['labels'],response=result['response'])
    except Exception as error:row.update(error_type=type(error).__name__,http_status=getattr(error,'status_code',None))
    row['seconds']=time.perf_counter()-start
    temp=path.with_suffix('.partial');temp.write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n');temp.replace(path)
    print(task['key'],row['status'],flush=True)
    return row


if __name__=='__main__':
    from dotenv import load_dotenv
    load_dotenv('.env')
    parser=argparse.ArgumentParser();parser.add_argument('--controls-only',action='store_true');args=parser.parse_args()
    if args.controls_only:ROOT=Path('artifacts/bilingual-order-synthetic-only-v1')
    cases,tasks=prepare(args.controls_only)
    with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(lambda task:one(task,cases),tasks))
    summary=dict(terminal_requests=len(results),valid=sum(r['status']=='valid' for r in results),scope='Progress only; full factorial/contrast analysis required.')
    (ROOT/'progress.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
