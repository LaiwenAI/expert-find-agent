"""Cross prompt and label wording on handwritten support controls."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import json
from pathlib import Path
from dotenv import load_dotenv
from laiwenai import LaiwenAI
from expert_search.choice_judgments import decide
from .audit_anchored_reasons import INSTRUCTION,CHOICES

SIMPLE='Do the passages support every factual claim in the reason? Use uncertain if the available passages are insufficient to decide.'
YESNO=[dict(value='yes',description='The passages support the factual claims in the reason.'),
       dict(value='no',description='The passages do not support the factual claims in the reason.'),
       dict(value='uncertain',description='The available passages are insufficient to decide.')]


def main():
    templates=[
        ('Find temperature measurement research.','The study measures temperature using a sensor.','The study measures temperature with a sensor.','The study uses a sensor to treat disease.'),
        ('Find wastewater treatment research.','The paper describes a membrane that filters wastewater.','The paper describes a membrane for filtering wastewater.','The paper reports a nuclear reactor for treating wastewater.'),
        ('Find battery materials research.','The authors study lithium-ion battery electrodes.','The paper studies lithium-ion battery electrodes.','The paper studies solar panel manufacturing.'),
        ('Find vibration monitoring research.','The study uses accelerometers to monitor vibration.','The study monitors vibration with accelerometers.','The study monitors vibration using satellite radar.'),
        ('Find corrosion research.','The paper investigates corrosion of steel in seawater.','The paper investigates seawater corrosion of steel.','The paper establishes that steel never corrodes.')]
    jobs=[]
    for index,(query,quote,good,bad) in enumerate(templates):
        for expected,reason in [('supported',good),('unsupported',bad)]:
            for wording,instruction in [('original',INSTRUCTION),('simple',SIMPLE)]:
                for names,options in [('original',CHOICES),('yesno',YESNO)]:
                    for reverse in (False,True):
                        jobs.append(dict(id=f'{index}-{expected}-{wording}-{names}-{int(reverse)}',payload=dict(query=query,reason=reason,quotes=[quote]),
                                         expected=expected,wording=wording,names=names,choices=list(reversed(options)) if reverse else options,instruction=instruction))
    root=Path('artifacts/anchored-support-prompt-ablation-v1');root.mkdir(parents=True,exist_ok=False);load_dotenv('.env')
    def run(job):
        row={k:job[k] for k in ('id','expected','wording','names')};row['status']='unresolved'
        try:
            answer=decide(LaiwenAI(model='gpt-6-luna',max_retries=0),job['payload'],[dict(type='choice',name='support',instructions=job['instruction'],choices=job['choices'])])
            label=answer['labels']['support'];row.update(status='valid',prediction={'yes':'supported','no':'unsupported'}.get(label,label))
        except Exception as error:row['error_type']=type(error).__name__
        return row
    with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(run,jobs))
    report={}
    for wording in ('original','simple'):
        for names in ('original','yesno'):
            subset=[r for r in rows if r['wording']==wording and r['names']==names];counts=Counter((r['expected'],r.get('prediction','unresolved')) for r in subset)
            report[wording+'-'+names]=[dict(expected=a,predicted=b,count=n) for (a,b),n in sorted(counts.items())]
    summary=dict(calls=len(rows),valid=sum(r['status']=='valid' for r in rows),arms=report,
                 scope='Post-hoc 2x2 instruction and choice-set wording ablation on five simple handwritten English template pairs, two orderings. The choice sets change both values and descriptions, so label-token effects are not isolated. Labels are not model inputs. Dependent cases, no independent human relevance gold or industrial accuracy claim.')
    (root/'results.json').write_text(json.dumps(rows,indent=2)+'\n');(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
