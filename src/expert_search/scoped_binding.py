"""Probe joint-application support separately from supported capability endpoints."""
import json
import time
from collections import Counter
from pathlib import Path
from laiwenai import LaiwenAI
from .agent import ExpertAgent
from .decision_cache import DecisionCache
from .scope_metrics import scoped_gain


def run(source='artifacts/synthesis-scope-v5',output='artifacts/scoped-binding-v2'):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);contexts=[];pairs=[]
    for path in sorted(Path(source).glob('*candidate*.json')):
        record=json.loads(path.read_text())
        if record['status']!='silver_candidate':continue
        texts=[c['text'] for c in record['example']['constraints']]
        constraint=('\u8d44\u6599\u660e\u786e\u8bb0\u8f7d\u540c\u4e00\u9879\u7814\u7a76\u5c06\u4ee5\u4e0b\u5404\u9879\u7814\u7a76\u4e3b\u9898\u8054\u5408\u5e94\u7528\u4e8e\u4e00\u5957\u6280\u672f\u65b9\u6848\uff1a'+'\uff1b'.join(texts)+
                    '\u3002\u5206\u522b\u5217\u51fa\u8fd9\u4e9b\u4e3b\u9898\u3001\u5206\u522b\u5b58\u5728\u4e0d\u540c\u9879\u76ee\u6216\u8bba\u6587\uff0c\u4e0d\u8db3\u4ee5\u8bc1\u660e\u6b64\u9879\u8054\u5408\u5e94\u7528\u8981\u6c42\u3002')
        evidence=list(record['sources'].values())
        if any(e['expert_id']!=record['expert_id'] for e in evidence):raise ValueError('Mixed expert sources')
        pairs.append(dict(constraint=constraint,text='\n\n'.join(e['text'] for e in evidence)))
        contexts.append(dict(id=record['id'],expert_id=record['expert_id'],
            atomic_constraints=texts,joint_query=constraint,evidence=evidence))
    (out/'contexts.json').write_text(json.dumps(contexts,ensure_ascii=False,indent=2))
    decisions={}
    for model in ('gpt-4o','gpt-6-luna'):
        agent=ExpertAgent(client=LaiwenAI(model=model))
        if model=='gpt-6-luna':agent.request_options={'reasoning_effort':'none'}
        cache=DecisionCache(str(out/(model+'.sqlite')),model+':scope-binding-v2:'+str(agent.request_options if hasattr(agent,'request_options') else {}),
            lambda instruction,payload:agent.ask('joint_application_audit',instruction+
                ' Use one short contiguous verbatim quote for support/contradiction, with no ellipsis, '
                'translation or joining of passages. Merely mentioning all topics is not joint-application support.',payload),batch_size=7)
        try:decisions[model]=cache.judge(pairs)
        finally:
            # Cache-only replay must never erase prior network-call traces.
            if agent.trace:
                (out/(model+'-trace-'+str(time.time_ns())+'.json')).write_text(
                    json.dumps(agent.trace,ensure_ascii=False,indent=2))
    rows=[]
    for i,context in enumerate(contexts):
        judgments={m:decisions[m]['judgments'][i] for m in decisions}
        for answer in judgments.values():
            if answer['label']!='unknown' and not any(answer['quote'] in e['text'] for e in context['evidence']):
                raise ValueError('Cross-source quote')
        labels={a['label'] for a in judgments.values()}
        schema={str(j):dict(kind='atomic',weight=1,mandatory=True) for j in range(len(context['atomic_constraints']))}
        independent={k:'supported' for k in schema}
        schema['joint']=dict(kind='relation',weight=1,mandatory=True,dependencies=list(independent))
        label=next(iter(labels)) if len(labels)==1 else None
        rows.append(dict(id=context['id'],joint_query=context['joint_query'],judgments=judgments,
            agreed_label=label,scoped_metric=scoped_gain(schema,dict(independent,joint=label)) if label is not None else None,
            atomic_labels_provenance='Prior full-base-query cross-model audit; not rejudged and not human gold'))
    summary=dict(pairs=len(rows),agreement_counts=dict(Counter(r['agreed_label'] or 'disagreement' for r in rows)),
        warning='Model-audited source support only. Metric sensitivity illustration; no human-quality validation or algorithm superiority. Unknown binding is not inability. Source support for nodes never alone proves joint application.')
    (out/'results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    (out/'summary.json').write_text(json.dumps(summary,indent=2));return summary


if __name__=='__main__':print(json.dumps(run(),indent=2))
