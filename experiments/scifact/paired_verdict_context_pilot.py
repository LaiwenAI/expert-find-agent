"""Alternating paired verdict calls on full versus selected source views."""
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
        if lexical_order(c['claim'],p['abstract'])[:3]!=row['selected'] or bool(groups)!=row['known_evidence'] or (groups and (len({g['label'] for g in groups})!=1 or groups[0]['label'].lower()!=row['label'] or metrics(row['selected'],groups)!=row['rationale'])):raise ValueError('Changed fixed source view')
        prepared.append((row,c,p))
    output.mkdir(parents=True,exist_ok=False);(output/'attempts').mkdir();load_dotenv('.env')
    frozen={'source_sha256':sha(archive),'parent_protocol_sha256':sha(parent/'protocol.json'),'parent_results_sha256':sha(parent/'results.json'),
            'case_ids':[r['claim_id'] for r in rows],'arm_order':'alternating by fixed cohort index, one request per arm',
            'views':'Full abstract vs selected top-three sentences in original source order; title and claim identical.',
            'question':QUESTION,'model':'gpt-6-luna','workers':3,'retries':0,
            'code_sha256':{str(p):sha(p) for p in (Path(__file__),Path('experiments/scifact/rationale_selection_pilot.py'),Path('src/expert_search/choice_judgments.py'))},
            'scope':'Post-hoc reused SciFact development queries and retrieved sources. Labels and rationale annotations withheld. Unannotated documents remain unknown.'}
    (output/'protocol.json').write_text(json.dumps(frozen,indent=2)+'\n')
    def run(item):
        index,(old,claim,paper)=item;client=LaiwenAI(model='gpt-6-luna',max_retries=0);arms={}
        for arm in (('full','selected') if index%2==0 else ('selected','full')):
            texts=paper['abstract'] if arm=='full' else [paper['abstract'][i] for i in sorted(old['selected'])]
            start=time.perf_counter();value={'status':'unresolved'}
            try:
                response=decide(client,{'claim':claim['claim'],'title':paper['title'],'abstract':texts},[QUESTION])
                value.update(status='valid',verdict=response['labels']['verdict'],decision_seconds=response['seconds'])
            except Exception as error:value.update(error_type=type(error).__name__,http_status=getattr(error,'status_code',None))
            value['attempt_seconds']=time.perf_counter()-start;arms[arm]=value
        result={'claim_id':old['claim_id'],'doc_id':old['doc_id'],'known_evidence':old['known_evidence'],'label':old['label'],
                'complete_rationale':old['rationale']['complete_rationale'] if old['known_evidence'] else None,'arms':arms}
        (output/'attempts'/(str(old['claim_id'])+'.json')).write_text(json.dumps(result,indent=2)+'\n');return result
    with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(run,enumerate(prepared)))
    (output/'results.json').write_text(json.dumps(results,indent=2)+'\n');valid=[r for r in results if all(r['arms'][a]['status']=='valid' for a in ('full','selected'))];known=[r for r in valid if r['known_evidence']]
    report={'planned':len(results),'jointly_valid':len(valid),'known_jointly_valid':len(known),'unannotated_jointly_valid':len(valid)-len(known),
            'same_verdict':sum(r['arms']['full']['verdict']==r['arms']['selected']['verdict'] for r in valid),
            'transitions':[{'full':a,'selected':b,'count':n} for (a,b),n in sorted(Counter((r['arms']['full']['verdict'],r['arms']['selected']['verdict']) for r in valid).items())],
            'known_full_correct':sum(r['arms']['full']['verdict']==r['label'] for r in known),'known_selected_correct':sum(r['arms']['selected']['verdict']==r['label'] for r in known),
            'known_selected_correct_complete_rationale':sum(r['complete_rationale'] and r['arms']['selected']['verdict']==r['label'] for r in known),
            'paired_known_correctness':paired_interval([int(r['arms']['selected']['verdict']==r['label'])-int(r['arms']['full']['verdict']==r['label']) for r in known]) if len(known)>=10 else None,
            'mean_attempt_seconds':{a:statistics.mean(r['arms'][a]['attempt_seconds'] for r in results) for a in ('full','selected')},
            'errors':{a:dict(Counter(r['arms'][a].get('error_type') for r in results if r['arms'][a]['status']!='valid')) for a in ('full','selected')},
            'protocol_sha256':sha(output/'protocol.json'),'results_sha256':sha(output/'results.json'),
            'scope':frozen['scope']+' Alternating, not randomized; one model call per arm and concurrency remain sensitive to timing and model variability. Abstract labels do not establish selected-view sufficiency. Query bootstrap is post-hoc and assumes independence. Timings exclude retrieval, indexing, initialization and concurrent service load.'}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.archive,a.parent,a.output)
