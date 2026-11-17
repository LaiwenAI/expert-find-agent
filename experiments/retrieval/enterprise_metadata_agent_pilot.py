"""Private live integration of date-aware evidence with the candidate agent."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import time

from dotenv import load_dotenv
from laiwenai import LaiwenAI
from expert_search.candidate_evidence_pipeline import CandidateEvidencePipeline
from expert_search.candidate_reasoning_agent import CandidateReasoningAgent
from expert_search.metadata_overlay import MetadataOverlayPipeline


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FrozenPipeline:
    def __init__(self,result):self.result=result
    def search_reranked(self,*args,**kwargs):return self.result


def main(output):
    previous=Path('artifacts/enterprise-agent-budget-pilot-v1/protocol.json');prior=json.loads(previous.read_text())
    source=Path('artifacts/dataset-release/fresh-enterprise-eval-v1/queries.jsonl')
    if sha(source)!=prior['queries_sha256']:raise ValueError('Changed query cohort')
    queries={r['id']:r for r in map(json.loads,source.read_text().splitlines()) if r['split']=='development'};ids=prior['query_ids']
    if len(ids)!=4 or len(set(ids))!=4 or any(q not in queries for q in ids):raise ValueError('Invalid development cohort')
    output.mkdir(parents=True,exist_ok=False)
    protocol={'query_ids':ids,'queries_sha256':sha(source),'prior_protocol_sha256':sha(previous),'model':'gpt-6-luna','automatic_retries':0,'workers':2,
              'options':{'reasoning_effort':'none','max_completion_tokens':1500},'candidate_pool':50,'candidate_views':20,'output_k':3,
              'code_sha256':{str(p):sha(p) for p in (Path(__file__),Path('src/expert_search/metadata_overlay.py'),Path('src/expert_search/candidate_reasoning_agent.py'),Path('src/expert_search/candidate_evidence_pipeline.py'))},
              'scope':'Four previously studied private development needs. One frozen retrieval per need, one model request, no retries. No independent labels, date truth, personal contribution, semantic entailment or historical cutoff validation.'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');load_dotenv('.env')
    start=time.perf_counter();pipeline=CandidateEvidencePipeline(scoring='resident-adaptive');init=time.perf_counter()-start
    start=time.perf_counter();overlay=MetadataOverlayPipeline(pipeline,'artifacts/corpus','artifacts/corpus-temporal-source-v1');overlay_init=time.perf_counter()-start
    prepared={};counts=Counter()
    for qid in ids:
        start=time.perf_counter();raw=overlay.search_reranked(queries[qid]['text'],k=20,candidates=50,evidence_per_expert=1);elapsed=time.perf_counter()-start
        expected={e['id']:e for row in raw['results'] for e in row['evidence']}
        if any(e['expert_id']!=row['expert']['id'] for row in raw['results'] for e in row['evidence']):raise ValueError('Foreign evidence')
        counts['input_sources']+=len(expected);counts['input_papers']+=sum(e.get('kind')=='paper' for e in expected.values())
        counts['input_parsed_dates']+=sum(e.get('publication',{}).get('status')=='parsed' for e in expected.values())
        prepared[qid]={'result':raw,'retrieval_seconds':elapsed,'view_sha256':hashlib.sha256(json.dumps(raw['results'],sort_keys=True,ensure_ascii=False).encode()).hexdigest()}
    pipeline.evidence.db.close();(output/'view-hashes.json').write_text(json.dumps({q:v['view_sha256'] for q,v in prepared.items()},indent=2)+'\n')
    def run(qid):
        agent=CandidateReasoningAgent(pipeline=FrozenPipeline(prepared[qid]['result']),client=LaiwenAI(model='gpt-6-luna',max_retries=0));agent.helper.request_options=protocol['options']
        original=agent.helper.ask;captured={}
        def ask(stage,instruction,payload):
            expected={e['id']:e for row in prepared[qid]['result']['results'] for e in row['evidence']}
            items=payload['evidence']
            if set(e['id'] for e in items)!=set(expected):raise ValueError('Changed model evidence set')
            for e in items:
                source=expected[e['id']]
                if e['publication']!=source.get('publication') or e['expert_id']!=source['expert_id'] or e['text']!=source['text'][:6000] or e['contribution_status']!='unverified':
                    raise ValueError('Changed metadata in model payload')
            captured['payload_sha256']=hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
            return original(stage,instruction,payload)
        agent.helper.ask=ask;start=time.perf_counter();row={'query_id':qid,'status':'error'}
        try:
            result=agent.run(queries[qid]['text'],k=3,candidates=20,evidence_per_expert=1,retrieval='adaptive_bm25',rerank_pool=50)
            row.update(status='valid',result=result,completion=result['completion'])
        except Exception as error:row.update(error_type=type(error).__name__,http_status=getattr(error,'status_code',None))
        row.update(model_seconds=time.perf_counter()-start,payload_sha256=captured.get('payload_sha256'))
        path=output/(hashlib.sha256(qid.encode()).hexdigest()+'.json');temp=path.with_suffix('.partial');temp.write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n');temp.replace(path)
        print('Completed query',row['status'],flush=True);return row
    with ThreadPoolExecutor(max_workers=2) as pool:rows=[future.result() for future in as_completed([pool.submit(run,qid) for qid in ids])]
    for row in rows:
        if row['status']!='valid':continue
        for expert in row['result']['results']:
            for e in expert['evidence']:
                counts['returned_citations']+=1;counts['returned_papers']+=e['kind']=='paper';counts['returned_parsed_dates']+=isinstance(e['publication'],dict) and e['publication'].get('status')=='parsed'
                counts['returned_unverified_roles']+=e['contribution_status']=='unverified'
    report={'attempts':len(rows),'valid':sum(r['status']=='valid' for r in rows),'failed':[{'error_type':r.get('error_type'),'http_status':r.get('http_status')} for r in rows if r['status']!='valid'],
            'completions':dict(Counter(r['completion']['status'] for r in rows if r['status']=='valid')),'counts':dict(counts),
            'mean_model_seconds':statistics.mean(r['model_seconds'] for r in rows),'mean_retrieval_seconds':statistics.mean(p['retrieval_seconds'] for p in prepared.values()),
            'pipeline_init_seconds':init,'overlay_init_seconds':overlay_init,'protocol_sha256':sha(output/'protocol.json'),'view_hashes_sha256':sha(output/'view-hashes.json'),'scope':protocol['scope']}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
