"""Query-clustered factorial effects on model-screened claims, not accuracy."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from expert_search.evaluation_gain import actionable_gain_interval

ARMS=['base','language','bridge','both']


def run():
    root=Path('artifacts/factorial-partial-comparison-v1');audit=Path('artifacts/factorial-witness-audit-v1');protocol=json.loads((root/'protocol.json').read_text());rows=[json.loads((root/(q+'.json')).read_text()) for q in protocol['query_ids']];screens={q:json.loads((audit/(q+'.json')).read_text()) for q in protocol['query_ids']}
    grades={}
    for r in map(json.loads,Path('artifacts/dataset-release/fresh-enterprise-eval-v1/judgments.jsonl').read_text().splitlines()):grades.setdefault(r['query_id'],{})[r['expert_id']]=r['grade']
    result={};matrix=[];common=[];timing=[];inputs={}
    for row in rows:
        q=row['query_id'];screen=screens[q];support=Counter()
        for name,origin in screen['origins'].items():
            if screen.get('audit',{}).get('labels',{}).get(name)=='supported_scoped_claim':support[origin['arm']]+=1
        matrix.append([support[a]/3 for a in ARMS]);common.append(all(v['review']['status']=='structurally_valid' for v in row['arms'].values()));timing.append([row['arms'][a]['seconds'] for a in ARMS])
        for p in [root/(q+'.json'),audit/(q+'.json')]:inputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    for ai,arm in enumerate(ARMS):
        counts=Counter();levels=Counter();gains=[];usage=[];screens_count=Counter()
        for row in rows:
            review=row['arms'][arm]['review'];valid=review['status']=='structurally_valid';ranking=review['result']['ranking'] if valid else [];names=[r['expert_id'] for r in ranking];positive={e for e,g in grades[row['query_id']].items() if g is not None and g>=2};counts.update(valid=valid,returned=len(names),known_silver_positives=len(positive),returned_known_silver_positives=len(positive&set(names)));levels.update(r['level'] for r in ranking);gains.append(actionable_gain_interval(names,grades[row['query_id']],3))
            for round in review.get('rounds',[]):
                n=(round['response'].get('usage') or {}).get('completion_tokens')
                if type(n) is int:usage.append(n)
            screen=screens[row['query_id']]
            for name,origin in screen['origins'].items():
                if origin['arm']==arm:screens_count[screen.get('audit',{}).get('labels',{}).get(name,'unresolved')]+=1
        times=np.asarray(timing)[:,ai];result[arm]=dict(counts,levels=dict(levels),screen=dict(screens_count),screen_supported_per_60_slots=sum(screens_count[k] for k in ['supported_scoped_claim'])/60,mean_seconds=float(times.mean()),median_seconds=float(np.median(times)),empirical_p95_seconds=float(np.quantile(times,.95)),mean_completion_tokens=float(np.mean(usage)) if usage else None,token_observed_requests=len(usage),silver_gain_lower=float(np.mean([g['lower'] for g in gains])),silver_gain_upper=float(np.mean([g['upper'] for g in gains])))
    def effects(values):
        x=np.asarray(values);v=dict(language=(x[:,1]-x[:,0]+x[:,3]-x[:,2])/2,bridge=(x[:,2]-x[:,0]+x[:,3]-x[:,1])/2,interaction=x[:,3]-x[:,2]-x[:,1]+x[:,0],both_minus_base=x[:,3]-x[:,0]);rng=np.random.default_rng(20261117);indices=rng.integers(0,len(x),size=(10000,len(x)));return {k:dict(mean=float(a.mean()),query_bootstrap95=list(map(float,np.quantile(a[indices].mean(axis=1),[.025,.975])))) for k,a in v.items()}
    report=dict(queries=len(rows),arms=result,screen_effects_per_display_slot_all=effects(matrix),paired_valid_queries=sum(common),screen_effects_per_display_slot_paired_valid=effects(np.asarray(matrix)[common]),latency_effects_seconds_all=effects(timing),inputs_sha256=inputs,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Fixed20dev queries, balanced4armorder, same7input/top3output. Screen factors measure same-model audit behavior, not independent accuracy. Main effects average over the other factor; interaction is difference-in-differences. Slot denominator3 prevents rewarding fewer positive claims alone; missing outputs contribute no completed screen credit, not negative ability. Bootstrap over20queries (19all-arm-valid secondary) not general population/semantic confidence. No retrieval E2E, own training or heldout usage.')
    (root/'analysis.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='inputs_sha256'},indent=2))


if __name__=='__main__':run()
