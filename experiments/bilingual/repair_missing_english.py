"""One local follow-up for the remaining Han-containing English view."""
import hashlib
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from laiwenai import LaiwenAI
from expert_search.bilingual.postvalidation import check_pair
from expert_search.choice_judgments import decide
from .audit_release_identifiers import PATTERNS
from .repair_english_han import DATA,OUT as REPAIRS,PROMPT,AUDIT,CHOICES


def main():
    source=DATA/'synthetic-queries.jsonl';raw=source.read_bytes();protocol=json.loads((REPAIRS/'protocol.json').read_text())
    if hashlib.sha256(raw).hexdigest()!=protocol['dataset_sha256']:raise ValueError('Changed source')
    rows={r['id']:r for r in map(json.loads,raw.splitlines())};flagged=[]
    for rid in protocol['selected_ids']:
        path=REPAIRS/'records'/(hashlib.sha256(rid.encode()).hexdigest()+'.json');prior=json.loads(path.read_text())
        if prior.get('english') is None or prior.get('contains_han'):flagged.append((rid,path,prior))
    if len(flagged)!=1:raise ValueError('Expected one remaining repair')
    rid,path,prior=flagged[0];source_row=rows[rid];out=Path('artifacts/english-han-repair-followup-v1');out.mkdir(parents=True,exist_ok=False)
    report=dict(id=rid,prior_status=source_row['status'],status='unresolved',english=None,prior_record_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    load_dotenv('.env');start=time.perf_counter()
    try:
        client=LaiwenAI(model='gpt-6-luna',max_retries=0)
        response=client.chat.completions.create(messages=[dict(role='system',content=PROMPT+' The previous repair still contains Chinese characters; translate every remaining such character.'),
                    dict(role='user',content=json.dumps(dict(chinese=source_row['zh'],english=source_row['en'],previous_repair=prior.get('english')),ensure_ascii=False))],
                    reasoning_effort='none',max_completion_tokens=1400,response_format=dict(type='json_object'))
        value=json.loads(response.choices[0].message.content)
        if set(value)!={'english'} or not isinstance(value['english'],str) or not value['english'].strip():raise ValueError('Invalid follow-up')
        report['english']=value['english'];report['contains_han']=bool(PATTERNS['han_or_radical'].search(value['english']))
        report['numeric_check']=check_pair(source_row['en'],value['english'])
        if not report['contains_han']:
            result=decide(client,dict(chinese=source_row['zh'],previous_english=source_row['en'],repaired_english=value['english']),[dict(type='choice',name='equivalence',instructions=AUDIT,choices=CHOICES)])
            report['audit']=result['labels']['equivalence'];report['status']='unjudged_followup'
    except Exception as error:report['error_type']=type(error).__name__
    report['seconds']=time.perf_counter()-start
    (out/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report.get(k) for k in ('status','contains_han','audit','error_type')},indent=2))


if __name__=='__main__':main()
