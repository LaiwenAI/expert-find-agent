"""Compare verdicts on retrieved selected sentences versus complete abstracts."""
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
from expert_search.statistics import paired_interval
from experiments.scifact.decisions_oracle_pilot import QUESTION,load,sha
from experiments.scifact.rationale_selection_pilot import lexical_order,metrics


def main(archive,parent,output):
    prior=json.loads((parent/'protocol.json').read_text());summary=json.loads((parent/'summary.json').read_text())
    if sha(archive)!=prior['source_sha256'] or sha(parent/'protocol.json')!=summary['protocol_sha256'] or sha(parent/'results.json')!=summary['results_sha256'] or prior['question']!=QUESTION:raise ValueError('Changed previous inputs')
    rows=json.loads((parent/'results.json').read_text());corpus,claims=load(archive);papers={str(p['doc_id']):p for p in corpus};by_claim={c['id']:c for c in claims}
    if [r['claim_id'] for r in rows]!=prior['case_ids'] or len(set(r['claim_id'] for r in rows))!=len(rows):raise ValueError('Changed cohort')
    prepared=[]
    for row in rows:
        c=by_claim[row['claim_id']];p=papers[row['doc_id']];groups=c['evidence'].get(row['doc_id'],[])
        selected=lexical_order(c['claim'],p['abstract'])[:3]
        if selected!=row['selected'] or bool(groups)!=row['known_evidence'] or (groups and (len({g['label'] for g in groups})!=1 or groups[0]['label'].lower()!=row['label'] or metrics(selected,groups)!=row['rationale'])):
            raise ValueError('Changed retrieved fixed view')
        prepared.append((row,c,p))
    output.mkdir(parents=True,exist_ok=False);(output/'attempts').mkdir();load_dotenv('.env')
    frozen={'source_sha256':sha(archive),'parent_protocol_sha256':sha(parent/'protocol.json'),'parent_results_sha256':sha(parent/'results.json'),
            'case_ids':[r['claim_id'] for r in rows],'selection':'Same retrieved abstract and top-three lexical sentences; sent in original abstract order.',
            'question':QUESTION,'model':'gpt-6-luna','workers':3,'retries':0,
            'code_sha256':{str(p):sha(p) for p in (Path(__file__),Path('experiments/scifact/rationale_selection_pilot.py'),Path('src/expert_search/choice_judgments.py'))},
            'scope':'Post-hoc sequential repeat on reused public development cases; same question and title, but shortened source. No label or prior decision is sent. Unannotated top-one sources are unknown, not negatives.'}
    (output/'protocol.json').write_text(json.dumps(frozen,indent=2)+'\n')
    def run(item):
        previous,claim,paper=item;start=time.perf_counter();r={'claim_id':previous['claim_id'],'doc_id':previous['doc_id'],'status':'unresolved',
                'known_evidence':previous['known_evidence'],'label':previous['label'],'prior_status':previous['status'],'full_verdict':previous.get('predicted'),
                'complete_rationale':previous['rationale']['complete_rationale'] if previous['known_evidence'] else None}
        try:
            shown=[paper['abstract'][i] for i in sorted(previous['selected'])]
            answer=decide(LaiwenAI(model='gpt-6-luna',max_retries=0),{'claim':claim['claim'],'title':paper['title'],'abstract':shown},[QUESTION])
            r.update(status='valid',selected_verdict=answer['labels']['verdict'],decision_seconds=answer['seconds'])
        except Exception as error:r.update(error_type=type(error).__name__,http_status=getattr(error,'status_code',None))
        r['attempt_seconds']=time.perf_counter()-start;(output/'attempts'/(str(r['claim_id'])+'.json')).write_text(json.dumps(r,indent=2)+'\n');return r
    with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(run,prepared))
    (output/'results.json').write_text(json.dumps(results,indent=2)+'\n');valid=[r for r in results if r['status']=='valid' and r['prior_status']=='valid'];known=[r for r in valid if r['known_evidence']]
    report={'planned':len(results),'jointly_valid':len(valid),'known_jointly_valid':len(known),'same_verdict':sum(r['selected_verdict']==r['full_verdict'] for r in valid),
            'verdict_changes':[{'full':a,'selected':b,'count':n} for (a,b),n in sorted(Counter((r['full_verdict'],r['selected_verdict']) for r in valid).items())],
            'known_full_correct':sum(r['full_verdict']==r['label'] for r in known),'known_selected_correct':sum(r['selected_verdict']==r['label'] for r in known),
            'known_complete_rationale_selected_correct':sum(r['complete_rationale'] and r['selected_verdict']==r['label'] for r in known),
            'paired_known_correctness':paired_interval([int(r['selected_verdict']==r['label'])-int(r['full_verdict']==r['label']) for r in known]) if len(known)>=10 else None,
            'unannotated_jointly_valid':len(valid)-len(known),'errors':dict(Counter(r.get('error_type') for r in results if r['status']!='valid')),
            'mean_attempt_seconds':statistics.mean(r['attempt_seconds'] for r in results),'protocol_sha256':sha(output/'protocol.json'),'results_sha256':sha(output/'results.json'),
            'scope':frozen['scope']+' Full and shortened calls run at different times; model stochasticity and ordering differences are confounded with context length. Labels describe full abstract, not necessarily sufficiency of selected sentences. Agreement is not entailment proof. Query bootstrap assumes independence, is post-hoc and unadjusted; timings use three concurrent workers and exclude retrieval.'}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.archive,a.parent,a.output)
