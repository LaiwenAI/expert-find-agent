"""Evaluate retrieval and paired full/selected verdicts on unused SciFact dev claims."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import time
from dotenv import load_dotenv
from laiwenai import LaiwenAI
from expert_search.candidate_evidence_pipeline import verify_evidence_documents
from expert_search.choice_judgments import decide
from expert_search.fast_retrieval import FastSearch
from expert_search.statistics import paired_interval
from experiments.scifact.decisions_oracle_pilot import QUESTION,load,sha
from experiments.scifact.rationale_selection_pilot import lexical_order,metrics


def main(archive,corpus,database,prior_paths,output):
    source,claims=load(archive);papers={str(p['doc_id']):p for p in source}
    adapted=[json.loads(s) for s in (corpus/'queries.jsonl').read_text().splitlines()];queries={q['id']:q for q in adapted}
    if len(queries)!=len(adapted) or len(papers)!=len(source):raise ValueError('Duplicate input IDs')
    used=set();previous={}
    for folder in prior_paths:
        p=json.loads((folder/'protocol.json').read_text());s=json.loads((folder/'summary.json').read_text())
        if sha(archive)!=p['archive_sha256'] or sha(folder/'protocol.json')!=s['protocol_sha256'] or sha(folder/'results.json')!=s['results_sha256']:raise ValueError('Changed prior cohort')
        used.update(c['claim_id'] for c in p['cases']);previous[str(folder)]=sha(folder/'protocol.json')
    selected=[]
    for claim in sorted(claims,key=lambda x:int(x['id'])):
        if claim['id'] in used:continue
        groups=[g for values in claim['evidence'].values() for g in values];labels={g['label'] for g in groups}
        if len(labels)==1 and labels<=set(('SUPPORT','CONTRADICT')):selected.append(claim)
    if len(selected)!=145 or len({c['id'] for c in selected})!=len(selected):raise ValueError('Unexpected fresh cohort')
    output.mkdir(parents=True,exist_ok=False);(output/'attempts').mkdir();load_dotenv('.env')
    frozen={'source_sha256':sha(archive),'adapted_corpus_sha256':sha(corpus/'evidence.jsonl'),'adapted_experts_sha256':sha(corpus/'experts.jsonl'),
            'adapted_queries_sha256':sha(corpus/'queries.jsonl'),'index_sha256':sha(database),'excluded_protocols_sha256':previous,
            'case_ids':[c['id'] for c in selected],'selection':'All previously unused dev claims with one unique support/contradict label across annotated documents, ordered by numeric claim ID.',
            'retrieval':'BM25 top one from full adapted corpus','sentence_selection':'BM25 top three in source order','arm_order':'alternating by cohort position',
            'question':QUESTION,'model':'gpt-6-luna','workers':3,'retries':0,
            'code_sha256':{str(p):sha(p) for p in (Path(__file__),Path('experiments/scifact/rationale_selection_pilot.py'),Path('src/expert_search/fast_retrieval.py'),Path('src/expert_search/choice_judgments.py'))},
            'scope':'Development claims not used in the two prior SciFact model pilots, but benchmark and protocol were previously inspected. Not a sealed test; document labels may be incomplete. Scientific claims, not enterprise expert fit.'}
    (output/'protocol.json').write_text(json.dumps(frozen,indent=2)+'\n')
    search=FastSearch(corpus,database);verify_evidence_documents(corpus,search);prepared=[]
    for claim in selected:
        qid=str(claim['id'])
        if qid not in queries or queries[qid]['query']!=claim['claim']:raise ValueError('Changed query')
        start=time.perf_counter();hits=search.search(claim['claim'],10,'bm25-max');retrieval=time.perf_counter()-start
        if not hits:raise ValueError('Empty retrieval')
        ids=[str(h['id']) for h in hits];doc=ids[0]
        if doc not in papers:raise ValueError('Unknown document')
        p=papers[doc];expected=(p['title']+'\n'+' '.join(p['abstract'])).strip()
        if hits[0]['evidence'][0]['text'].split()!=expected.split():raise ValueError('Changed source beyond whitespace')
        start=time.perf_counter();chosen=lexical_order(claim['claim'],p['abstract'])[:3];sentence_time=time.perf_counter()-start;groups=claim['evidence'].get(doc,[])
        if groups and len({g['label'] for g in groups})!=1:raise ValueError('Mixed document labels')
        prepared.append({'claim_id':claim['id'],'doc_id':doc,'top10':ids,'selected':chosen,'known_evidence':bool(groups),'label':groups[0]['label'].lower() if groups else None,
                         'rationale':metrics(chosen,groups) if groups else None,'top10_annotated':any(x in claim['evidence'] for x in ids),
                         'retrieval_seconds':retrieval,'sentence_seconds':sentence_time})
    search.db.close();by_claim={c['id']:c for c in selected}
    def run(item):
        i,row=item;row=dict(row);claim=by_claim[row['claim_id']];p=papers[row['doc_id']];client=LaiwenAI(model='gpt-6-luna',max_retries=0);arms={}
        for arm in (('full','selected') if i%2==0 else ('selected','full')):
            abstract=p['abstract'] if arm=='full' else [p['abstract'][j] for j in sorted(row['selected'])]
            start=time.perf_counter();result={'status':'unresolved'}
            try:
                decision=decide(client,{'claim':claim['claim'],'title':p['title'],'abstract':abstract},[QUESTION])
                result.update(status='valid',verdict=decision['labels']['verdict'],decision_seconds=decision['seconds'])
            except Exception as error:result.update(error_type=type(error).__name__,http_status=getattr(error,'status_code',None))
            result['attempt_seconds']=time.perf_counter()-start;arms[arm]=result
        row['arms']=arms;(output/'attempts'/(str(row['claim_id'])+'.json')).write_text(json.dumps(row,indent=2)+'\n');return row
    with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(run,enumerate(prepared)))
    (output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    valid=[r for r in rows if all(r['arms'][a]['status']=='valid' for a in ('full','selected'))];known=[r for r in valid if r['known_evidence']]
    report={'planned':len(rows),'claim_strata':dict(Counter(next(iter({g['label'] for groups in c['evidence'].values() for g in groups})) for c in selected)),
            'top1_annotated':sum(r['known_evidence'] for r in rows),'top10_annotated':sum(r['top10_annotated'] for r in rows),'jointly_valid':len(valid),'known_jointly_valid':len(known),
            'known_full_correct':sum(r['arms']['full']['verdict']==r['label'] for r in known),'known_selected_correct':sum(r['arms']['selected']['verdict']==r['label'] for r in known),
            'known_complete_rationale':sum(r['rationale']['complete_rationale'] for r in known),'known_selected_correct_complete_rationale':sum(r['rationale']['complete_rationale'] and r['arms']['selected']['verdict']==r['label'] for r in known),
            'paired_known_correctness':paired_interval([int(r['arms']['selected']['verdict']==r['label'])-int(r['arms']['full']['verdict']==r['label']) for r in known]) if len(known)>=10 else None,
            'same_verdict':sum(r['arms']['full']['verdict']==r['arms']['selected']['verdict'] for r in valid),
            'errors':{a:dict(Counter(r['arms'][a].get('error_type') for r in rows if r['arms'][a]['status']!='valid')) for a in ('full','selected')},
            'mean_retrieval_seconds':statistics.mean(r['retrieval_seconds'] for r in rows),'mean_sentence_seconds':statistics.mean(r['sentence_seconds'] for r in rows),
            'mean_attempt_seconds':{a:statistics.mean(r['arms'][a]['attempt_seconds'] for r in rows) for a in ('full','selected')},
            'protocol_sha256':sha(output/'protocol.json'),'results_sha256':sha(output/'results.json'),
            'scope':frozen['scope']+' Nonrandom alternating calls, model and service variability remain. Paired intervals assume independent queries and are unadjusted. Results on annotated top-one subset are conditional; unannotated documents are not negatives. Abstract labels do not establish selected-sentence sufficiency. Concurrent model attempts and local retrieval timings are not end-to-end service latency.'}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);p.add_argument('--corpus',type=Path,required=True);p.add_argument('--database',type=Path,required=True);p.add_argument('--prior',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.archive,a.corpus,a.database,a.prior,a.output)
