"""Handwritten positive and negative controls for the quote-support auditor."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import json
from pathlib import Path
from dotenv import load_dotenv
from laiwenai import LaiwenAI
from expert_search.choice_judgments import decide
from .audit_anchored_reasons import INSTRUCTION, CHOICES


def main():
    templates=[
        ('Find temperature measurement research.','The study measures temperature using a sensor.','The study measures temperature with a sensor.','The study uses a sensor to treat disease.'),
        ('Find wastewater treatment research.','The paper describes a membrane that filters wastewater.','The paper describes a membrane for filtering wastewater.','The paper reports a nuclear reactor for treating wastewater.'),
        ('Find battery materials research.','The authors study lithium-ion battery electrodes.','The paper studies lithium-ion battery electrodes.','The paper studies solar panel manufacturing.'),
        ('Find vibration monitoring research.','The study uses accelerometers to monitor vibration.','The study monitors vibration with accelerometers.','The study monitors vibration using satellite radar.'),
        ('Find corrosion research.','The paper investigates corrosion of steel in seawater.','The paper investigates seawater corrosion of steel.','The paper establishes that steel never corrodes.')]
    cases=[]
    for i,(query,quote,good,bad) in enumerate(templates):
        for name,reason,label in [('supported',good,'supported'),('unsupported',bad,'unsupported')]:
            for order in (CHOICES,list(reversed(CHOICES))):
                cases.append(dict(id=f'{i}-{name}-{order[0]["value"]}',payload=dict(query=query,reason=reason,quotes=[quote]),expected=label,choices=order))
    out=Path('artifacts/anchored-reason-audit-controls-v1');out.mkdir(parents=True,exist_ok=False);load_dotenv('.env')
    def run(case):
        row={'id':case['id'],'expected':case['expected'],'status':'unresolved'}
        try:
            answer=decide(LaiwenAI(model='gpt-6-luna',max_retries=0),case['payload'],[dict(type='choice',name='support',instructions=INSTRUCTION,choices=case['choices'])])
            row.update(status='valid',predicted=answer['labels']['support'])
        except Exception as exc:row['error_type']=type(exc).__name__
        return row
    with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(run,cases))
    counts=Counter((r['expected'],r.get('predicted','unresolved')) for r in rows)
    result=dict(cases=len(rows),counts=[dict(expected=a,predicted=b,count=n) for (a,b),n in sorted(counts.items())],
                scope='Five handwritten English template pairs, two choice orders. Expected labels are withheld from the model. Small dependent controls, not a representative auditor accuracy estimate.')
    (out/'results.json').write_text(json.dumps(rows,indent=2)+'\n');(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
