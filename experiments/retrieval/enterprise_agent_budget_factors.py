"""Isolate output cap and reasoning-effort factors on fixed agent evidence."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time

from dotenv import load_dotenv
from laiwenai import LaiwenAI
from expert_search.candidate_evidence_pipeline import CandidateEvidencePipeline
from expert_search.candidate_reasoning_agent import CandidateReasoningAgent


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


class FixedPipeline:
    def __init__(self, result): self.result=result
    def search_reranked(self, query, k, candidates, evidence_per_expert): return self.result


class ObservedClient:
    def __init__(self,client,capture):
        self.client=client;self.capture=capture;self.chat=SimpleNamespace(completions=SimpleNamespace(create=self.create))
    def create(self,**kwargs):
        response=self.client.chat.completions.create(**kwargs)
        choice=response.choices[0];usage=getattr(response,'usage',None)
        content=choice.message.content or ''
        self.capture['response']={'finish_reason':getattr(choice,'finish_reason',None),'content_characters':len(content),
                                  'content_sha256':hashlib.sha256(content.encode()).hexdigest(),
                                  'prompt_tokens':getattr(usage,'prompt_tokens',None),'completion_tokens':getattr(usage,'completion_tokens',None)}
        return response


def main(output):
    prior=Path('artifacts/enterprise-agent-budget-pilot-v1');previous=json.loads((prior/'protocol.json').read_text())
    data=Path('artifacts/dataset-release/fresh-enterprise-eval-v1/queries.jsonl')
    if sha(data)!=previous['queries_sha256']:raise ValueError('Changed queries')
    queries={r['id']:r for r in map(json.loads,data.read_text().splitlines()) if r['split']=='development'}
    ids=previous['query_ids']
    if len(ids)!=4 or set(ids)-set(queries):raise ValueError('Changed cohort')
    old={}
    for qid in ids:
        old[qid]={arm:json.loads((prior/(qid+'-'+arm+'.json')).read_text()) for arm in ('default','compact')}
        if old[qid]['default']['prompt_sha256']!=old[qid]['compact']['prompt_sha256'] or old[qid]['default']['prompt_sha256'] is None:raise ValueError('Prior prompts differ')
    output.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),Path('src/expert_search/candidate_reasoning_agent.py'),Path('src/expert_search/candidate_evidence_pipeline.py'),Path('src/expert_search/agent.py')]
    protocol={'query_ids':ids,'previous_protocol_sha256':sha(prior/'protocol.json'), 'previous_attempts_sha256':{qid+'-'+arm:sha(prior/(qid+'-'+arm+'.json')) for qid in ids for arm in ('default','compact')},
              'queries_sha256':sha(data),'model':'gpt-6-luna','arms':{'none_unbounded':{'reasoning_effort':'none'},'default_cap':{'max_completion_tokens':1000}},
              'code_sha256':{str(p):sha(p) for p in files},'workers':2,'automatic_retries':0,
              'scope':'Two new factor arms on four previously examined development needs; same source prompt verified by hash against prior default/compact arms. Arms vary reasoning effort and output cap separately, but are collected later than prior arms. Two concurrent workers, alternating order, one attempt per arm, no retry. Invalid JSON and request failures retained; no independent relevance, semantic correctness or held-out evaluation.'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    start=time.perf_counter();engine=CandidateEvidencePipeline(scoring='resident');initialization=time.perf_counter()-start;prepared={}
    for qid in ids:
        start=time.perf_counter();retrieved=engine.search_reranked(queries[qid]['text'],k=20,candidates=50,evidence_per_expert=1)
        prepared[qid]={'result':retrieved,'retrieval_seconds':time.perf_counter()-start}
    snapshot=engine.ranker.snapshot;engine.evidence.db.close();load_dotenv('.env')
    def worker(item):
        i,qid=item;capture={};client=ObservedClient(LaiwenAI(model='gpt-6-luna',max_retries=0),capture)
        agent=CandidateReasoningAgent(pipeline=FixedPipeline(prepared[qid]['result']),client=client);original=agent.helper.ask
        def ask(stage,instruction,payload):
            capture['prompt_sha256']=hashlib.sha256(json.dumps({'stage':stage,'instruction':instruction,'payload':payload},ensure_ascii=True,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            if capture['prompt_sha256']!=old[qid]['default']['prompt_sha256']:raise ValueError('Changed frozen prompt')
            return original(stage,instruction,payload)
        agent.helper.ask=ask;attempts={}
        for arm in (('none_unbounded','default_cap') if i%2==0 else ('default_cap','none_unbounded')):
            agent.helper.request_options=protocol['arms'][arm];capture.clear();start=time.perf_counter();row={'query_id':qid,'arm':arm,'status':'failed'}
            try:
                result=agent.run(queries[qid]['text'],k=5,candidates=20,evidence_per_expert=1,retrieval='adaptive_bm25',rerank_pool=50)
                row.update(status='valid',returned=len(result['results']),rejected=result['rejected_results'],ranking=[r['expert']['id'] for r in result['results']],result=result)
            except Exception as error:row.update(error_type=type(error).__name__,http_status=getattr(error,'status_code',None))
            row.update(model_stage_seconds=time.perf_counter()-start,prompt_sha256=capture.get('prompt_sha256'),response_meta=capture.get('response'));attempts[arm]=row
            (output/(qid+'-'+arm+'.json')).write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n');print('Completed query',i+1,arm,row['status'],flush=True)
        return attempts
    with ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(worker,enumerate(ids)))
    metrics={}
    for arm in protocol['arms']:
        items=[r[arm] for r in rows];valid=[r for r in items if r['status']=='valid'];observed=[r['response_meta'] for r in items if r['response_meta'] is not None]
        metrics[arm]={'attempts':len(items),'valid':len(valid),'mean_model_stage_seconds':sum(r['model_stage_seconds'] for r in items)/len(items),
                      'returned':sum(r['returned'] for r in valid),'rejected':sum(r['rejected'] for r in valid),
                      'finish_reasons':{str(reason):sum(x['finish_reason']==reason for x in observed) for reason in sorted({x['finish_reason'] for x in observed},key=str)},
                      'usage_records':sum(isinstance(x['completion_tokens'],int) for x in observed),
                      'completion_tokens_observed':sum(x['completion_tokens'] for x in observed if isinstance(x['completion_tokens'],int))}
    report={'queries':len(rows),'arms':metrics,'matching_prior_prompts':sum(r[a]['prompt_sha256']==old[ids[i]]['default']['prompt_sha256'] for i,r in enumerate(rows) for a in protocol['arms']),
            'mean_shared_retrieval_seconds':sum(x['retrieval_seconds'] for x in prepared.values())/len(prepared),'pipeline_initialization_seconds':initialization,
            'corpus_snapshot_sha256':snapshot,'protocol_sha256':sha(output/'protocol.json'),'scope':protocol['scope']}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
