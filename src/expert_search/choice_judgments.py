"""Validated fixed-choice transport for annotation, with no hidden model fallback."""
import json
import math
import time


def decide(client,records,questions):
    if not 1<=len(questions)<=100:raise ValueError('Invalid choice question budget')
    allowed={q['name']:{c['value'] for c in q['choices']} for q in questions}
    if len(allowed)!=len(questions):raise ValueError('Duplicate choice names')
    payload=dict(model='gpt-6-luna',input=json.dumps(records,ensure_ascii=False),questions=questions)
    start=time.perf_counter();create=getattr(getattr(client,'decisions',None),'create',None)
    raw=create(**payload).model_dump(mode='json') if callable(create) else client.post('/decisions',cast_to=dict[str,object],body=payload)
    elapsed=time.perf_counter()-start;answers=raw.get('answers',[])
    if len(answers)!=len(questions) or {a.get('name') for a in answers}!=set(allowed):raise ValueError('Missing or foreign annotation answers')
    labels={}
    for a in answers:
        choices=allowed[a['name']];probs=a.get('probabilities',[])
        if a.get('type')!='choice' or a.get('choice') not in choices:raise ValueError('Refused annotation')
        if len(probs)!=len(choices) or {p.get('value') for p in probs}!=choices:raise ValueError('Invalid annotation probabilities')
        values=[p.get('probability') for p in probs]
        if any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in values) or not math.isclose(sum(values),1,abs_tol=1e-3):raise ValueError('Invalid annotation probability values')
        labels[a['name']]=a['choice']
    return dict(labels=labels,response=raw,seconds=elapsed,notice='Fixed choices are model judgments, not calibrated certainty or independent human labels.')
