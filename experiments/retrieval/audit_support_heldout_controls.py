"""Fresh handcrafted support controls for Decisions and explained Chat."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from laiwenai import LaiwenAI
from expert_search.choice_judgments import decide
from .audit_anchored_reasons import INSTRUCTION
from .audit_support_prompt_ablation import YESNO


def cases():
    pairs=[
      ('Find water purification methods.','The paper uses activated carbon to remove dye from water.','The paper removes dye from water using activated carbon.','The paper removes salt from water using activated carbon.'),
      ('Find machine inspection methods.','The work detects cracks in steel using ultrasonic signals.','The work detects cracks in steel with ultrasound.','The work detects cracks in plastic with ultrasound.'),
      ('Find robotics navigation methods.','The robot navigates using a camera and a map.','The robot uses a camera and a map for navigation.','The robot navigates without a camera.'),
      ('Find polymer recycling research.','The method recycles polyethylene by dissolving it.','The method recycles polyethylene using dissolution.','The method recycles polyethylene by burning it.'),
      ('Find gas sensing research.','The sensor detects ammonia at room temperature.','The sensor detects ammonia at room temperature.','The sensor detects carbon dioxide at room temperature.'),
      ('Find soil analysis research.','The analysis estimates soil moisture from radar measurements.','Radar measurements are used to estimate soil moisture.','Radar measurements are used to estimate blood pressure.')]
    return [dict(id=f'{i}-{label}',payload=dict(query=q,reason=r,quotes=[passage]),expected=label)
            for i,(q,passage,positive,negative) in enumerate(pairs)
            for label,r in [('supported',positive),('unsupported',negative)]]


def main():
    root=Path('artifacts/anchored-support-heldout-controls-v1');root.mkdir(parents=True,exist_ok=False);load_dotenv('.env')
    selected=cases();choices=YESNO
    protocol=dict(model='gpt-6-luna',case_ids=[r['id'] for r in selected],arms=['decisions','chat'],
                  scope='Post-hoc held-out handwritten controls, no human benchmark. Source and expected labels are synthetic; expected labels never enter model input. Alternating arm order, no retries; templates and paired cases are dependent.')
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    def run(item):
        index,case=item;client=LaiwenAI(model='gpt-6-luna',max_retries=0);outputs={}
        for arm in (('decisions','chat') if index%2==0 else ('chat','decisions')):
            row=dict(status='unresolved');start=time.perf_counter()
            try:
                if arm=='decisions':
                    result=decide(client,case['payload'],[dict(type='choice',name='support',instructions=INSTRUCTION,choices=choices)])
                    predicted={'yes':'supported','no':'unsupported','uncertain':'uncertain'}[result['labels']['support']]
                else:
                    schema=dict(type='object',properties=dict(reasoning=dict(type='string'),verdict=dict(type='string',enum=['supported','unsupported','uncertain'])),required=['reasoning','verdict'],additionalProperties=False)
                    result=client.chat.completions.create(messages=[dict(role='system',content=INSTRUCTION+' Explain briefly and then give the verdict in the requested JSON.'),dict(role='user',content=json.dumps(case['payload']))],
                              reasoning_effort='none',max_completion_tokens=500,response_format=dict(type='json_schema',json_schema=dict(name='support_review',strict=True,schema=schema)))
                    body=json.loads(result.choices[0].message.content)
                    if set(body)!={'reasoning','verdict'} or not isinstance(body['reasoning'],str) or not body['reasoning'].strip() or body['verdict'] not in ('supported','unsupported','uncertain'):raise ValueError('Invalid response')
                    predicted=body['verdict']
                row.update(status='valid',prediction=predicted)
            except Exception as error:row['error_type']=type(error).__name__
            row['seconds']=time.perf_counter()-start;outputs[arm]=row
        return dict(id=case['id'],expected=case['expected'],arms=outputs)
    with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(run,enumerate(selected)))
    summary={'cases':len(rows),'arms':{}}
    for arm in ('decisions','chat'):
        counts=Counter((r['expected'],r['arms'][arm].get('prediction','unresolved')) for r in rows)
        summary['arms'][arm]=dict(confusion=[dict(expected=a,predicted=b,count=n) for (a,b),n in sorted(counts.items())],
                                  mean_seconds=sum(r['arms'][arm]['seconds'] for r in rows)/len(rows))
    summary['scope']=protocol['scope']+' Different interfaces, output budgets and prompts; no causal speed attribution.'
    (root/'results.json').write_text(json.dumps(rows,indent=2)+'\n');(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
