"""Pilot native choice screening followed by bounded evidence explanations."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

from dotenv import load_dotenv
from laiwenai import LaiwenAI
from expert_search.candidate_evidence_pipeline import CandidateEvidencePipeline
from expert_search.candidate_reasoning_agent import CandidateReasoningAgent
from expert_search.choice_judgments import decide
from experiments.annotate_fresh_evaluation import FIT_CHOICES, FIT_INSTRUCTION


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


class FixedPipeline:
    def __init__(self, rows, timing, snapshot):self.rows=rows;self.timing=timing;self.snapshot=snapshot
    def search_reranked(self,query,k,candidates,evidence_per_expert):
        return {'results':self.rows,'timing':self.timing,'source_snapshot_sha256':self.snapshot}


def main(output):
    prior=Path('artifacts/enterprise-agent-budget-pilot-v1');protocol0=json.loads((prior/'protocol.json').read_text())
    query_file=Path('artifacts/dataset-release/fresh-enterprise-eval-v1/queries.jsonl')
    if sha(query_file)!=protocol0['queries_sha256']:raise ValueError('Changed query source')
    queries={r['id']:r for r in map(json.loads,query_file.read_text().splitlines()) if r['split']=='development'};ids=protocol0['query_ids']
    if len(ids)!=4 or set(ids)-set(queries):raise ValueError('Changed development cohort')
    output.mkdir(parents=True,exist_ok=False)
    code=[Path(__file__),Path('src/expert_search/choice_judgments.py'),Path('src/expert_search/candidate_reasoning_agent.py'),Path('src/expert_search/candidate_evidence_pipeline.py'),Path('experiments/annotate_fresh_evaluation.py')]
    protocol={'query_ids':ids,'prior_protocol_sha256':sha(prior/'protocol.json'),'queries_sha256':sha(query_file),'model':'gpt-6-luna',
              'candidate_pool':50,'decision_candidates':20,'explanation_candidates':5,'explanation_k':3,'explanation_options':{'reasoning_effort':'none','max_completion_tokens':1500},
              'fit_instruction':FIT_INSTRUCTION,'fit_choices':FIT_CHOICES,'workers':2,'automatic_retries':0,'code_sha256':{str(p):sha(p) for p in code},
              'scope':'Four previously inspected development needs. One native Decisions call assesses twenty owned fixed evidence views; only direct or conditional model choices proceed, with up to five views and at most three explanations. Unqualified cases abstain. One bounded chat call validates exact owned quotes; no retries. Shared retrieval measured separately, index initialization excluded. Decisions are uncalibrated, quotes do not establish entailment or personal contribution. Different output budget and function from prior five-result chat; no independent quality claim.'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');start=time.perf_counter();engine=CandidateEvidencePipeline(scoring='resident');initialization=time.perf_counter()-start;prepared={}
    for qid in ids:
        start=time.perf_counter();result=engine.search_reranked(queries[qid]['text'],k=20,candidates=50,evidence_per_expert=1)
        prepared[qid]={'result':result,'seconds':time.perf_counter()-start}
    snapshot=engine.ranker.snapshot;engine.evidence.db.close();load_dotenv('.env')
    def worker(item):
        i,qid=item;client=LaiwenAI(model='gpt-6-luna',max_retries=0);raw=prepared[qid]['result'];views=raw['results']
        records={'demand':queries[qid]['text'],'candidates':{f'c{j}':{'expert_id':r['expert']['id'],'sources':[{'id':e['id'],'text':e['text'][:6000]} for e in r['evidence']]} for j,r in enumerate(views)}}
        questions=[{'type':'choice','name':f'c{j}','instructions':FIT_INSTRUCTION+' Evaluate only candidate '+f'c{j}'+'.','choices':FIT_CHOICES} for j in range(len(views))]
        started=time.perf_counter();row={'query_id':qid,'status':'decision_failed','retrieval_seconds':prepared[qid]['seconds']}
        try:
            choice=decide(client,records,questions);labels=choice['labels'];row.update(decision_seconds=time.perf_counter()-started,decision_labels=labels,decision_response=choice['response'])
            ordinal={'direct_application':0,'transferable_method':1};selected=sorted((j for j in range(len(views)) if labels[f'c{j}'] in ordinal),key=lambda j:(ordinal[labels[f'c{j}']],j))[:5]
            row['selected_candidates']=len(selected)
            if not selected:row.update(status='abstained',results=[],explanation_seconds=0.)
            else:
                picked=[views[j] for j in selected];agent=CandidateReasoningAgent(pipeline=FixedPipeline(picked,raw['timing'],snapshot),client=client)
                agent.helper.request_options=protocol['explanation_options'];step=time.perf_counter()
                try:
                    result=agent.run(queries[qid]['text'],k=min(3,len(selected)),candidates=len(selected),evidence_per_expert=1,retrieval='adaptive_bm25',rerank_pool=len(selected))
                    row.update(status='valid',result=result,returned=len(result['results']),rejected=result['rejected_results'],explanation_seconds=time.perf_counter()-step)
                except Exception as error:row.update(status='explanation_failed',error_type=type(error).__name__,http_status=getattr(error,'status_code',None),explanation_seconds=time.perf_counter()-step)
        except Exception as error:row.update(error_type=type(error).__name__,http_status=getattr(error,'status_code',None),decision_seconds=time.perf_counter()-started)
        row['model_seconds']=time.perf_counter()-started;row['total_seconds']=row['model_seconds']+row['retrieval_seconds']
        (output/(qid+'.json')).write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n');print('Completed query',i+1,row['status'],flush=True);return row
    with ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(worker,enumerate(ids)))
    report={'queries':len(rows),'statuses':{s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})},
            'mean_total_seconds':sum(r['total_seconds'] for r in rows)/len(rows),'mean_decision_seconds':sum(r['decision_seconds'] for r in rows)/len(rows),
            'mean_explanation_seconds':sum(r.get('explanation_seconds',0.) for r in rows)/len(rows),'mean_retrieval_seconds':sum(r['retrieval_seconds'] for r in rows)/len(rows),
            'returned':sum(r.get('returned',0) for r in rows),'selected_candidates':sum(r.get('selected_candidates',0) for r in rows),
            'pipeline_initialization_seconds':initialization,'corpus_snapshot_sha256':snapshot,'protocol_sha256':sha(output/'protocol.json'),'scope':protocol['scope']}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
