"""Arm-blind same-model witness screening of generated top3 positive claims.

A model audit is a diagnostic, not independent correctness or human gold.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
from expert_search.choice_judgments import decide
from annotate_fresh_evaluation import WITNESS_CHOICES,WITNESS_INSTRUCTION

DATA=Path('artifacts/focused-partial-comparison-v1');ROOT=Path('artifacts/focused-witness-audit-v1')


def one(item):
    row,job=item;path=ROOT/(row['query_id']+'.json')
    if path.exists():return json.loads(path.read_text())
    records=[]
    for arm,a in row['arms'].items():
        review=a['review']
        if review['status']!='structurally_valid':continue
        for rank,proposal in enumerate(review['result']['ranking'][:3]):
            if proposal['level'] not in ('transferable_method','direct_application'):continue
            eid=proposal['expert_id'];sources=job['candidates'][eid];cited={c['evidence_id'] for e in proposal['evidence'] for c in e['citations']}
            if not cited<=set(sources):raise ValueError('Foreign audit source')
            claim=dict(original_demand=job['demand'],relation=proposal['level'],target_spans=proposal['demand_citations'],evidence=proposal['evidence'],mechanism=proposal['mechanism'],assumptions=proposal['assumptions'],remaining_gaps=proposal['remaining_demand_gaps'],source_context={sid:sources[sid] for sid in sorted(cited)})
            records.append((dict(arm=arm,expert_id=eid,rank=rank+1),claim))
    random.Random(row['query_id']+':blind-witness').shuffle(records)
    origins={f'p{i}':meta for i,(meta,claim) in enumerate(records)};payload={f'p{i}':claim for i,(meta,claim) in enumerate(records)}
    result=dict(query_id=row['query_id'],origins=origins,proposal_count=len(records),input_sha256=hashlib.sha256(json.dumps(payload,ensure_ascii=False).encode()).hexdigest())
    if payload:
        from laiwenai import LaiwenAI
        client=LaiwenAI(model='gpt-6-luna',max_retries=0)
        questions=[dict(type='choice',name=name,instructions=WITNESS_INSTRUCTION+' Proposal: '+name,choices=WITNESS_CHOICES) for name in payload]
        try:result['audit']=decide(client,payload,questions)
        except Exception as error:result['error_type']=type(error).__name__
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2));print(row['query_id'],len(records),'audited' if 'audit' in result else 'no_verdict',flush=True);return result


def run():
    protocol=json.loads((DATA/'protocol.json').read_text());paths=[DATA/(q+'.json') for q in protocol['query_ids']];rows=[json.loads(p.read_text()) for p in paths]
    jobs_path=Path('artifacts/fresh-method-data-v1/evaluation-jobs.jsonl');jobs={j['id']:j for j in map(json.loads,jobs_path.read_text().splitlines()) if j['split']=='development'}
    if any(r['query_id'] not in jobs for r in rows):raise ValueError('Not development')
    ROOT.mkdir(exist_ok=True)
    frozen=dict(model='gpt-6-luna',inputs_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},jobs_sha256=hashlib.sha256(jobs_path.read_bytes()).hexdigest(),code_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('experiments/annotate_fresh_evaluation.py'),Path('src/expert_search/choice_judgments.py')]},instructions=WITNESS_INSTRUCTION,choices=WITNESS_CHOICES,scope='Only valid top3 actionable claims, randomized blind arm names, no previous silver/native verdict supplied. Same model/provider multi-process diagnostic, not independent truth; not part of measured online arms.3workers,no retries. Heldout unused.')
    marker=ROOT/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=frozen:raise ValueError('Changed witness audit')
    marker.write_text(json.dumps(frozen,indent=2))
    with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(one,[(r,jobs[r['query_id']]) for r in rows]))
    common={r['query_id'] for r in rows if all(a['review']['status']=='structurally_valid' for a in r['arms'].values())};summary={}
    for scope in ['all','paired_valid']:
        counts={arm:Counter() for arm in ['full','focused']}
        for r in results:
            if scope=='paired_valid' and r['query_id'] not in common:continue
            for name,meta in r['origins'].items():
                label=r.get('audit',{}).get('labels',{}).get(name,'unresolved');counts[meta['arm']][label]+=1
        summary[scope]={a:dict(c) for a,c in counts.items()}
    report=dict(queries=len(rows),paired_valid_queries=len(common),results=summary,scope=frozen['scope']);(ROOT/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':
    from dotenv import load_dotenv
    load_dotenv('.env');run()
