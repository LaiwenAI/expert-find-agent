"""Separate repeat variability, record order and choice order in full controls."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from .audit_order_controls import digest,request

ROOT=Path('artifacts/bilingual-order-controls-v1')


def run():
    protocol=json.loads((ROOT/'protocol.json').read_text())
    cases=list(map(json.loads,(ROOT/'cases.jsonl').read_text().splitlines()))
    tasks=list(map(json.loads,(ROOT/'tasks.jsonl').read_text().splitlines()))
    if digest(cases)!=protocol['cases_sha256'] or digest(tasks)!=protocol['tasks_sha256']:raise ValueError('Changed control design')
    by_id={r['id']:r for r in cases};rows=[];hashes={}
    for task in tasks:
        path=ROOT/'predictions'/(task['key']+'.json')
        if not path.exists():raise ValueError('Refuse incomplete ordering controls')
        row=json.loads(path.read_text())
        if any(row[k]!=task[k] for k in task):raise ValueError('Changed request identity')
        payload,qs=request(by_id[row['case_id']],row['choice_order'],row['record_order'])
        if digest(dict(model='gpt-6-luna',input=json.dumps(payload,ensure_ascii=False),questions=qs))!=row['input_sha256']:raise ValueError('Changed wire request')
        if 'expected' in payload or 'group' in payload or 'stratum' in payload:raise ValueError('Gold leaked into inference')
        if row['status'] not in ('valid','unresolved'):raise ValueError('Invalid terminal state')
        if row['status']=='valid' and (set(row['labels'])!={'equivalence','grounding'} or any(v not in {'supported','changed_or_unsupported','uncertain'} for v in row['labels'].values())):raise ValueError('Invalid labels')
        hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest();rows.append(row)
    synthetic=[r for r in rows if by_id[r['case_id']]['group']=='synthetic']
    def control_stats(subset):
        return dict(requests=len(subset),valid=sum(r['status']=='valid' for r in subset),by_question={q:dict(correct=sum(r['status']=='valid' and r['labels'][q]==by_id[r['case_id']]['expected'] for r in subset),wrong=sum(r['status']=='valid' and r['labels'][q]!=by_id[r['case_id']]['expected'] for r in subset),labels=dict(Counter(r['labels'][q] for r in subset if r['status']=='valid'))) for q in ['equivalence','grounding']})
    controls=dict(all=control_stats(synthetic),by_permutation={''.join(map(str,order)):control_stats([r for r in synthetic if r['choice_order']==order]) for order in sorted({tuple(r['choice_order']) for r in synthetic}) for order in [list(order)]},by_expected={label:control_stats([r for r in synthetic if by_id[r['case_id']]['expected']==label]) for label in ['supported','changed_or_unsupported']})
    real=[r for r in rows if by_id[r['case_id']]['group']=='real']
    lookup={(r['case_id'],tuple(r['choice_order']),r['record_order'],r['repeat']):r for r in real}
    ids=sorted({r['case_id'] for r in real});orders=[(0,1,2),(2,1,0)]
    contrasts={}
    for factor in ['choice_order','record_order','repeat']:
        pairs=[]
        for case_id in ids:
            if factor=='choice_order':pairs.extend((lookup[case_id,orders[0],record,repeat],lookup[case_id,orders[1],record,repeat]) for record in [0,1] for repeat in [0,1])
            elif factor=='record_order':pairs.extend((lookup[case_id,order,0,repeat],lookup[case_id,order,1,repeat]) for order in orders for repeat in [0,1])
            else:pairs.extend((lookup[case_id,order,record,0],lookup[case_id,order,record,1]) for order in orders for record in [0,1])
        valid=[(a,b) for a,b in pairs if a['status']==b['status']=='valid']
        contrasts[factor]=dict(comparisons=len(pairs),both_valid=len(valid),questions={q:dict(disagreements=sum(a['labels'][q]!=b['labels'][q] for a,b in valid),support_rate_first=sum(a['labels'][q]=='supported' for a,b in valid)/len(valid) if valid else None,support_rate_second=sum(b['labels'][q]=='supported' for a,b in valid)/len(valid) if valid else None) for q in ['equivalence','grounding']})
    cells={}
    for order in orders:
        for record in [0,1]:
            subset=[r for r in real if tuple(r['choice_order'])==order and r['record_order']==record]
            cells[str(order)+'_record'+str(record)]=dict(requests=len(subset),labels={q:dict(Counter(r['labels'][q] for r in subset if r['status']=='valid')) for q in ['equivalence','grounding']})
    summary=dict(terminal_requests=len(rows),statuses=dict(Counter(r['status'] for r in rows)),synthetic_controls=controls,real_factorial_cases=len(ids),real_contrasts=contrasts,real_cells=cells,protocol_sha256=hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest(),inputs_sha256=hashes,analysis_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope=protocol['scope']+' Record-order changes preserve all values; choice permutations preserve label descriptions. Simple synthetic contrast correctness does not validate complex-query semantic judgments. Real-case labels remain model outputs, not truths; correlated comparisons are not independent samples.')
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['inputs_sha256','synthetic_controls','real_cells']},indent=2));print('Synthetic controls:',json.dumps(controls['all']))


if __name__=='__main__':run()
