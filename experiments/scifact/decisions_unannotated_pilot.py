"""Inspect abstention on cited SciFact abstracts without annotated evidence."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import time
from dotenv import load_dotenv
from laiwenai import LaiwenAI
from expert_search.choice_judgments import decide
from experiments.scifact.decisions_oracle_pilot import QUESTION, load, sha


def main(archive,output,parent):
    protocol=json.loads((parent/'protocol.json').read_text());summary=json.loads((parent/'summary.json').read_text())
    if sha(parent/'protocol.json')!=summary['protocol_sha256'] or sha(parent/'results.json')!=summary['results_sha256'] or sha(archive)!=protocol['archive_sha256']:
        raise ValueError('Changed prior pilot or official source')
    corpus,claims=load(archive);papers={str(p['doc_id']):p for p in corpus}
    prior={c['claim_id'] for c in protocol['cases']};selected=[]
    for claim in sorted(claims,key=lambda c:int(c['id'])):
        if claim['id'] in prior:continue
        candidates=sorted((str(d) for d in claim['cited_doc_ids'] if str(d) in papers and str(d) not in claim['evidence']),key=int)
        if candidates:selected.append((claim,candidates[0]))
    eligible=len(selected);selected=selected[:40]
    if len(selected)!=40:raise ValueError('Insufficient eligible claims')
    output.mkdir(parents=True,exist_ok=False);(output/'attempts').mkdir();load_dotenv('.env')
    frozen={'archive_sha256':sha(archive),'prior_protocol_sha256':sha(parent/'protocol.json'),'prior_summary_sha256':sha(parent/'summary.json'),
            'eligible_claims':eligible,'cases':[{'claim_id':c['id'],'doc_id':d} for c,d in selected],
            'selection':'After excluding prior pilot claims, first 40 dev claims with cited corpus document lacking an evidence annotation; lowest numeric document ID per claim.',
            'question':QUESTION,'model':'gpt-6-luna','workers':3,'retries':0,
            'code_sha256':{str(p):sha(p) for p in (Path(__file__),Path('experiments/scifact/decisions_oracle_pilot.py'),Path('src/expert_search/choice_judgments.py'))},
            'scope':'Cited abstracts without evidence annotations are not independent verified negatives. Post-hoc nonrandom sample, no retrieval or industrial fit. Decisions withheld annotation status.'}
    (output/'protocol.json').write_text(json.dumps(frozen,indent=2)+'\n')
    def run(item):
        claim,doc=item;paper=papers[doc];start=time.perf_counter();row={'claim_id':claim['id'],'doc_id':doc,'status':'unresolved'}
        try:
            answer=decide(LaiwenAI(model='gpt-6-luna',max_retries=0),{'claim':claim['claim'],'title':paper['title'],'abstract':paper['abstract']},[QUESTION])
            row.update(status='valid',predicted=answer['labels']['verdict'],decision_seconds=answer['seconds'])
        except Exception as error:row.update(error_type=type(error).__name__,http_status=getattr(error,'status_code',None))
        row['attempt_seconds']=time.perf_counter()-start
        (output/'attempts'/(str(claim['id'])+'.json')).write_text(json.dumps(row,indent=2)+'\n');return row
    with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(run,selected))
    (output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    report={'planned':len(rows),'valid':sum(r['status']=='valid' for r in rows),
            'outcomes':dict(Counter(r.get('predicted','unresolved') for r in rows)),
            'mean_attempt_seconds':statistics.mean(r['attempt_seconds'] for r in rows),
            'protocol_sha256':sha(output/'protocol.json'),'results_sha256':sha(output/'results.json'),
            'scope':frozen['scope']+' Outcome frequencies are not error rates or semantic gold; failures remain unresolved. Three workers and client setup prevent service-level latency conclusions.'}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--parent',type=Path,default=Path('artifacts/scifact-decisions-oracle-v1'));a=p.parse_args();main(a.archive,a.output,a.parent)
