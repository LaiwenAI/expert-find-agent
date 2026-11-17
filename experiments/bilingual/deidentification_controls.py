"""Synthetic controls for identity and technical-preservation screening."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from dotenv import load_dotenv
from laiwenai import LaiwenAI
from expert_search.choice_judgments import decide
from .deidentify_english_flags import IDENTITY,MEANING,YESNO


def cases():
    return [
        dict(id='company_removed',previous='Acme Corp needs a sensor above 80 C.',revised='A manufacturer needs a sensor above 80 C.',identity='no',meaning='yes'),
        dict(id='company_kept',previous='Acme Corp needs a sensor above 80 C.',revised='Acme Corp needs a sensor above 80 C.',identity='yes',meaning='yes'),
        dict(id='threshold_changed',previous='Acme Corp needs a sensor above 80 C.',revised='A manufacturer needs a sensor below 80 C.',identity='no',meaning='no'),
        dict(id='generic_role',previous='A university needs a system to measure water flow.',revised='A university needs a system to measure water flow.',identity='no',meaning='yes'),
        dict(id='object_changed',previous='A manufacturer needs to filter water.',revised='A manufacturer needs to filter oil.',identity='no',meaning='no'),
        dict(id='contact_removed',previous='Contact lab@example.org: a steel mill needs inspection.',revised='A steel mill needs inspection.',identity='no',meaning='yes'),
    ]


def main():
    out=Path('artifacts/deidentification-controls-v2');out.mkdir(parents=True,exist_ok=False);load_dotenv('.env')
    jobs=[(case,reverse) for case in cases() for reverse in (False,True)]
    def run(job):
        case,reverse=job;row=dict(id=case['id'],reverse=reverse,status='unresolved',expected={'identity':case['identity'],'meaning':case['meaning']})
        choices=list(reversed(YESNO)) if reverse else YESNO
        questions=[dict(type='choice',name='identity',instructions=IDENTITY,choices=choices),dict(type='choice',name='meaning',instructions=MEANING,choices=choices)]
        try:
            result=decide(LaiwenAI(model='gpt-6-luna',max_retries=0),dict(previous=case['previous'],revised=case['revised']),questions)
            row.update(status='valid',prediction=result['labels'])
        except Exception as error:row['error_type']=type(error).__name__
        return row
    with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(run,jobs))
    metrics={}
    for name in ('identity','meaning'):
        counts=Counter((r['expected'][name],r.get('prediction',{}).get(name,'unresolved')) for r in rows)
        metrics[name]=[dict(expected=a,predicted=b,count=n) for (a,b),n in sorted(counts.items())]
    report=dict(cases=len(rows),valid=sum(r['status']=='valid' for r in rows),metrics=metrics,
                scope='Six handcrafted synthetic pairs with two choice orders. Expected labels are not model inputs. Template-dependent controls do not establish privacy, semantic safety or accuracy on real data.')
    (out/'results.json').write_text(json.dumps(rows,indent=2)+'\n');(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
