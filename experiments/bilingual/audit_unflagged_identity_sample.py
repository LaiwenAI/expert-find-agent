"""Stratified model audit of unflagged English construction records."""
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import random
from pathlib import Path
from dotenv import load_dotenv
from laiwenai import LaiwenAI
from expert_search.choice_judgments import decide
from .deidentify_english_flags import IDENTITY,YESNO

DATA=Path('artifacts/dataset-release/english-alignment-gated-v1')
OUT=Path('artifacts/unflagged-identity-sample-v1')


def main():
    manifest=json.loads((DATA/'manifest.json').read_text());raw=(DATA/'queries.jsonl').read_bytes();digest=hashlib.sha256(raw).hexdigest()
    if digest!=manifest['output_sha256']:raise ValueError('Changed view')
    rows=[json.loads(line) for line in raw.splitlines()];strata=defaultdict(list)
    for row in rows:
        if row['deidentification']=='not_assessed':strata[row['intent_type'],row['status']].append(row)
    if len(strata)!=10 or sum(map(len,strata.values()))!=4979:raise ValueError('Changed cohort')
    rng=random.Random(20261130);selected=[];spec=[]
    for (intent,status),items in sorted(strata.items()):
        items.sort(key=lambda r:r['id'])
        if len(items)<20:raise ValueError('Insufficient stratum')
        sample=rng.sample(items,20);selected.extend(sample);spec.append(dict(intent_type=intent,status=status,population=len(items),sample=len(sample)))
    rng.shuffle(selected);OUT.mkdir(parents=True,exist_ok=False);(OUT/'records').mkdir();load_dotenv('.env')
    protocol=dict(source_sha256=digest,seed=20261130,strata=spec,selected_ids=[r['id'] for r in selected],model='gpt-6-luna',instruction=IDENTITY,choices=YESNO,
                  scope='Post-hoc stratified sample of 200 records not flagged by lexical rules. One same-model explicit-identity judgment per text, not a human audit, indirect-identity assessment, confidentiality review or proof of anonymity. Parent groups can repeat.')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    def run(item):
        index,row=item;result=dict(id=row['id'],intent_type=row['intent_type'],prior_status=row['status'],status='unresolved')
        try:
            choices=YESNO if index%2==0 else list(reversed(YESNO))
            answer=decide(LaiwenAI(model='gpt-6-luna',max_retries=0),dict(text=row['english']),[dict(type='choice',name='identity',instructions=IDENTITY,choices=choices)])
            result.update(status='valid',label=answer['labels']['identity'],seconds=answer['seconds'])
        except Exception as error:result['error_type']=type(error).__name__
        (OUT/'records'/(hashlib.sha256(row['id'].encode()).hexdigest()+'.json')).write_text(json.dumps(result,indent=2)+'\n');return result
    with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(run,enumerate(selected)))
    counts=Counter(r.get('label','unresolved') for r in results)
    by_status={status:dict(Counter(r.get('label','unresolved') for r in results if r['prior_status']==status)) for status in sorted({r['prior_status'] for r in results})}
    report=dict(cases=len(results),valid=sum(r['status']=='valid' for r in results),labels=dict(counts),by_status=by_status,scope=protocol['scope'])
    (OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
