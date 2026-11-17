"""Compare model identity judgments for flagged original and revised texts."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from dotenv import load_dotenv
from laiwenai import LaiwenAI
from expert_search.choice_judgments import decide
from .deidentify_english_flags import DATA,OUT as CANDIDATES,IDENTITY,YESNO


def main():
    source=(DATA/'queries.jsonl').read_bytes();protocol=json.loads((CANDIDATES/'protocol.json').read_text())
    if hashlib.sha256(source).hexdigest()!=protocol['dataset_sha256']:raise ValueError('Changed data')
    rows={r['id']:r for r in map(json.loads,source.splitlines())};jobs=[]
    for rid in protocol['ids']:
        path=CANDIDATES/'records'/(hashlib.sha256(rid.encode()).hexdigest()+'.json');candidate=json.loads(path.read_text())
        if candidate['id']!=rid or candidate['prior_status']!=rows[rid]['status']:raise ValueError('Changed candidate')
        if candidate.get('english') is not None:jobs.append(dict(id=rid,original=rows[rid]['english'],candidate=candidate['english']))
    out=Path('artifacts/deidentification-paired-audit-v1');out.mkdir(parents=True,exist_ok=False);load_dotenv('.env')
    def one(job):
        row=dict(id=job['id'],status='unresolved')
        try:
            questions=[dict(type='choice',name='original',instructions='For the text named original: '+IDENTITY,choices=YESNO),
                       dict(type='choice',name='candidate',instructions='For the text named candidate: '+IDENTITY,choices=YESNO)]
            result=decide(LaiwenAI(model='gpt-6-luna',max_retries=0),dict(original=job['original'],candidate=job['candidate']),questions)
            row.update(status='valid',labels=result['labels'])
        except Exception as error:row['error_type']=type(error).__name__
        return row
    with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(one,jobs))
    transitions=Counter((r['labels']['original'],r['labels']['candidate']) for r in results if r['status']=='valid')
    report=dict(cases=len(results),valid=sum(r['status']=='valid' for r in results),
                transitions=[dict(original=a,candidate=b,count=n) for (a,b),n in sorted(transitions.items())],
                scope='Same-model paired identity judgments on 21 rule-flagged texts. This is not independent human review and does not detect all indirect identifiers or determine rights or confidentiality. No source text printed.')
    (out/'results.json').write_text(json.dumps(results,indent=2)+'\n');(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
