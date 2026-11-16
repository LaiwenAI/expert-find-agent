"""Paired native Decisions batching pilot, with deterministic randomized order."""
import hashlib
import json
import random
import inspect
from pathlib import Path
from expert_search.decisions_backend import DecisionsBackend


def run():
    source=Path('artifacts/blind-source-review-v1/inputs.jsonl')
    rows=[json.loads(line) for line in source.read_text().splitlines()]
    out=Path('artifacts/decisions-batch-comparison-v2');out.mkdir(exist_ok=True)
    protocol=dict(version=2,backend_sha256=hashlib.sha256(inspect.getsource(DecisionsBackend).encode()).hexdigest(),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        model='gpt-6-luna',seed=20261116,repetitions=1,
        comparison='Three sequential single-question requests versus one three-question request; same evidence and question IDs.')
    marker=out/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=protocol:raise ValueError('Changed protocol')
    marker.write_text(json.dumps(protocol,indent=2))
    backend=DecisionsBackend();results=[]
    for row in rows:
        evidence=json.dumps(row['sources'],ensure_ascii=False)
        pairs=[dict(id=c['id'],constraint=c['text'],text=evidence) for c in row['conditions']]
        order=['single','batch'];random.Random(str(protocol['seed'])+row['id']).shuffle(order)
        observations={}
        for mode in order:
            target=out/(row['id']+'-'+mode+'.json')
            if target.exists():result=json.loads(target.read_text())
            else:
                result=dict(id=row['id'],mode=mode,order=order,status='pending',calls=[])
                try:
                    for group in ([pairs] if mode=='batch' else [[p] for p in pairs]):
                        result['calls'].append(backend.judge(group))
                    result['status']='valid'
                except Exception as error:
                    result.update(status='error',error_type=type(error).__name__,http_status=getattr(error,'status_code',None))
                temporary=target.with_suffix('.partial');temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2));temporary.replace(target)
            observations[mode]=result
        if all(r['status']=='valid' for r in observations.values()):
            labels={m:{j['id']:j['label'] for call in r['calls'] for j in call['judgments']} for m,r in observations.items()}
            results.append(dict(id=row['id'],seconds={m:sum(c['seconds'] for c in r['calls']) for m,r in observations.items()},
                agreement=sum(labels['single'][p['id']]==labels['batch'][p['id']] for p in pairs),labels=labels))
        print(row['id'],[observations[m]['status'] for m in order],flush=True)
    report=dict(protocol=protocol,planned_views=len(rows),complete_views=len(results),rows=results,
        notice='Sequential request latency, including SDK transport; one repetition, no human gold and no accuracy conclusion.')
    if len(results)==len(rows):
        total={m:sum(r['seconds'][m] for r in results) for m in ('single','batch')}
        report.update(total_seconds=total,speed_ratio=total['single']/total['batch'],
                      label_agreement=sum(r['agreement'] for r in results),judgments=len(rows)*3)
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__':
    from dotenv import load_dotenv
    load_dotenv('.env');run()
