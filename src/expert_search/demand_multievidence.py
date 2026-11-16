"""Matched single-vs-multiple source experiment for original industrial clauses."""
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from laiwenai import LaiwenAI
from .agent import ExpertAgent
from .corpus import records
from .decision_cache import DecisionCache
from .fast_retrieval import FastSearch


def run(source='artifacts/demand-evidence-pilot-v2', output='artifacts/demand-multievidence'):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    original=[r for _,r in records(Path(source)/'pairs.jsonl')]
    search=FastSearch('artifacts/corpus','artifacts/fast-search.sqlite')
    started=time.perf_counter();contexts=[];pairs=[]
    for row in original:
        hits=search.search(row['clause'],2,'bm25-max')
        selected=next((h for h in hits if h['id']==row['expert_id']),None)
        if selected is None or selected['evidence'][0]['id']!=row['evidence_id']:
            raise ValueError('Candidate or top source changed; matched experiment invalid')
        evidence=[dict(id=e['id'],expert_id=e['expert_id'],source=e['source'],text=e['text'][:4000])
                  for e in selected['evidence']]
        if len({e['expert_id'] for e in evidence})!=1:
            raise ValueError('Sources belong to different experts')
        if evidence[0]['text']!=row['evidence_text']:
            raise ValueError('Single source changed')
        context=dict(id=row['id'],demand_id=row['demand_id'],expert_id=row['expert_id'],
                     clause=row['clause'],evidence=evidence)
        contexts.append(context)
        for size in (1,3):
            # Plain source text, with no generated summaries or inferred capabilities.
            text='\n\n'.join(e['text'] for e in evidence[:size])
            pairs.append(dict(constraint=row['clause'],text=text))
    retrieval_seconds=time.perf_counter()-started
    manifest=json.dumps(contexts,ensure_ascii=False,sort_keys=True)
    fingerprint=hashlib.sha256(manifest.encode()).hexdigest()
    (out/'contexts.json').write_text(manifest)
    decisions={};timings={}
    for model in ('gpt-4o','gpt-6-luna'):
        agent=ExpertAgent(client=LaiwenAI(model=model))
        if model=='gpt-6-luna':agent.request_options={'reasoning_effort':'none'}
        version=model+(':none' if model=='gpt-6-luna' else '')+':multi-source-v1'
        cache=DecisionCache(str(out/(model+'.sqlite')),version,
            lambda instruction,payload:agent.ask('industrial_support',instruction,payload),batch_size=10)
        start=time.perf_counter();decisions[model]=cache.judge(pairs)
        timings[model]=dict(wall_seconds=time.perf_counter()-start,trace=agent.trace,
                            network_calls=decisions[model]['network_calls'])
        (out/(model+'-trace.json')).write_text(json.dumps(timings[model],ensure_ascii=False,indent=2))
    results=[]
    for i,context in enumerate(contexts):
        conditions={}
        for j,size in enumerate((1,3)):
            judgments={m:decisions[m]['judgments'][2*i+j] for m in decisions}
            for judgment in judgments.values():
                if judgment['label']!='unknown' and not any(
                    judgment['quote'] in e['text'] for e in context['evidence'][:size]):
                    raise ValueError('Quote crosses source boundaries')
            labels=[a['label'] for a in judgments.values()]
            conditions[str(size)]=dict(judgments=judgments,agreement=len(set(labels))==1,
                                       agreed_label=labels[0] if len(set(labels))==1 else None)
        results.append(dict(id=context['id'],demand_id=context['demand_id'],
                            expert_id=context['expert_id'],conditions=conditions))
    transitions=Counter((r['conditions']['1']['agreed_label'],r['conditions']['3']['agreed_label'])
                        for r in results)
    summary=dict(pairs=len(results),source_fingerprint=fingerprint,retrieval_seconds=retrieval_seconds,
        transitions=[dict(single=a,multiple=b,count=n) for (a,b),n in transitions.items()],
        model_calls={m:timings[m]['network_calls'] for m in timings},
        warning='Model-agreement support diagnostic, not human gold or actual expert ability. Same experts and complete clauses in both conditions. Multiple means up to three sources, each truncated at 4000 characters. Calls include source-order and batch-context effects; no causal accuracy claim.')
    (out/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary


if __name__=='__main__':print(json.dumps(run(),ensure_ascii=False,indent=2))
