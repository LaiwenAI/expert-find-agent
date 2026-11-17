"""Compare scoped and atomic support prompts on fixed synthetic controls."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from laiwenai import LaiwenAI
from .audit_anchored_reasons import INSTRUCTION
from .audit_support_heldout_controls import cases

ATOMIC=('Determine whether the passage explicitly supports the claim. '
        'Use only the passage; do not infer additional facts. '
        'Choose supported, unsupported, or uncertain, and explain briefly.')


def main():
    selected=cases();root=Path('artifacts/anchored-atomic-entailment-v1');root.mkdir(parents=True,exist_ok=False);load_dotenv('.env')
    protocol=dict(model='gpt-6-luna',cases=[r['id'] for r in selected],instructions={'scoped':INSTRUCTION,'atomic':ATOMIC},
                  scope='Post-hoc prompt comparison on previously inspected synthetic controls. Same Chat interface, schema, source and output cap; atomic arm omits query and other task constraints. No independent human gold or real-data accuracy claim.')
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    schema=dict(type='object',properties=dict(reasoning=dict(type='string'),verdict=dict(type='string',enum=['supported','unsupported','uncertain'])),required=['reasoning','verdict'],additionalProperties=False)
    def run(item):
        i,case=item;client=LaiwenAI(model='gpt-6-luna',max_retries=0);outputs={}
        for arm in (('scoped','atomic') if i%2==0 else ('atomic','scoped')):
            body=case['payload'] if arm=='scoped' else dict(passage=case['payload']['quotes'][0],claim=case['payload']['reason'])
            result={'status':'unresolved'};start=time.perf_counter()
            try:
                response=client.chat.completions.create(messages=[dict(role='system',content=protocol['instructions'][arm]+' Return the requested JSON.'),dict(role='user',content=json.dumps(body))],
                    reasoning_effort='none',max_completion_tokens=500,response_format=dict(type='json_schema',json_schema=dict(name='support_review',strict=True,schema=schema)))
                value=json.loads(response.choices[0].message.content)
                if set(value)!={'reasoning','verdict'} or not isinstance(value['reasoning'],str) or not value['reasoning'].strip() or value['verdict'] not in ('supported','unsupported','uncertain'):raise ValueError('Invalid response')
                result.update(status='valid',verdict=value['verdict'],reasoning=value['reasoning'])
            except Exception as error:result['error_type']=type(error).__name__
            result['seconds']=time.perf_counter()-start;outputs[arm]=result
        return dict(id=case['id'],expected=case['expected'],arms=outputs)
    with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(run,enumerate(selected)))
    report={}
    for arm in ('scoped','atomic'):
        counts=Counter((r['expected'],r['arms'][arm].get('verdict','unresolved')) for r in rows)
        report[arm]=[dict(expected=a,verdict=b,count=n) for (a,b),n in sorted(counts.items())]
    summary=dict(cases=len(rows),arms=report,scope=protocol['scope'])
    (root/'results.json').write_text(json.dumps(rows,indent=2)+'\n');(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
