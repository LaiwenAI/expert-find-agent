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
        constraint=('资料明确记载同一项研究将以下各项研究主题联合应用于一套技术方案：'+'；'.join(texts)+
                    '。分别列出这些主题、分别存在不同项目或论文，不足以证明此项联合应用要求。')
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
