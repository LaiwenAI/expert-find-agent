"""Paired output-budget control using identical frozen candidate evidence."""
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


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


class FixedPipeline:
    def __init__(self, result): self.result = result
    def search_reranked(self, query, k, candidates, evidence_per_expert): return self.result


def main(output):
    data = Path('artifacts/dataset-release/fresh-enterprise-eval-v1'); manifest = json.loads((data/'manifest.json').read_text())
    if sha(data/'queries.jsonl') != manifest['files_sha256']['queries.jsonl']: raise ValueError('Changed query data')
    queries = sorted((r for r in map(json.loads,(data/'queries.jsonl').read_text().splitlines()) if r['split']=='development'),key=lambda r:r['id'])[40:44]
    if len(queries)!=4: raise ValueError('Incomplete cohort')
    output.mkdir(parents=True,exist_ok=False)
    code=[Path(__file__),Path('src/expert_search/candidate_reasoning_agent.py'),Path('src/expert_search/candidate_evidence_pipeline.py'),Path('src/expert_search/agent.py')]
    protocol={'query_ids':[r['id'] for r in queries], 'queries_sha256':manifest['files_sha256']['queries.jsonl'], 'model':'gpt-6-luna',
              'arms':{'default':{},'compact':{'reasoning_effort':'none','max_completion_tokens':1000}}, 'workers':2, 'automatic_retries':0,
              'code_sha256':{str(p):sha(p) for p in code},
              'scope':'Four reused development needs, fixed resident candidate50/return20 with one source each. Arms use the same complete prompt; compact changes reasoning effort and output cap together. One attempt per arm, alternating order with two concurrent workers, no retries. Shared retrieval and initialization recorded separately. No independent expert-fit, citation-entailment or held-out evaluation.'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    start=time.perf_counter();engine=CandidateEvidencePipeline(scoring='resident');initialization=time.perf_counter()-start
    prepared={}
    for query in queries:
        start=time.perf_counter(); result=engine.search_reranked(query['text'],k=20,candidates=50,evidence_per_expert=1)
        prepared[query['id']]={'result':result,'seconds':time.perf_counter()-start}
    snapshot=engine.ranker.snapshot;engine.evidence.db.close();load_dotenv('.env')
    def worker(item):
        i,query=item;agent=CandidateReasoningAgent(pipeline=FixedPipeline(prepared[query['id']]['result']),client=LaiwenAI(model='gpt-6-luna',max_retries=0));original=agent.helper.ask;capture={}
        def ask(stage,instruction,payload):
            capture['prompt_sha256']=hashlib.sha256(json.dumps({'stage':stage,'instruction':instruction,'payload':payload},ensure_ascii=True,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            return original(stage,instruction,payload)
        agent.helper.ask=ask;attempts={}
        for arm in (('default','compact') if i%2==0 else ('compact','default')):
            agent.helper.request_options=protocol['arms'][arm];capture.clear();start=time.perf_counter();row={'query_id':query['id'],'arm':arm,'status':'failed'}
            try:
                result=agent.run(query['text'],k=5,candidates=20,evidence_per_expert=1,retrieval='adaptive_bm25',rerank_pool=50)
                row.update(status='valid',returned=len(result['results']),rejected=result['rejected_results'],
                           ranking=[r['expert']['id'] for r in result['results']],usage=result['usage'],result=result)
            except Exception as error: row.update(error_type=type(error).__name__,http_status=getattr(error,'status_code',None))
            row.update(model_stage_seconds=time.perf_counter()-start,prompt_sha256=capture.get('prompt_sha256'))
            attempts[arm]=row;(output/(query['id']+'-'+arm+'.json')).write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')
            print('Completed query',i+1,arm,row['status'],flush=True)
        return attempts
    with ThreadPoolExecutor(max_workers=2) as pool: rows=list(pool.map(worker,enumerate(queries)))
    arms={}
    for arm in protocol['arms']:
        items=[r[arm] for r in rows];valid=[r for r in items if r['status']=='valid']
        arms[arm]={'attempts':len(items),'valid':len(valid),'mean_model_stage_seconds':sum(r['model_stage_seconds'] for r in items)/len(items),
                   'mean_total_with_shared_retrieval_seconds':sum(r['model_stage_seconds']+prepared[r['query_id']]['seconds'] for r in items)/len(items),
                   'returned':sum(r['returned'] for r in valid),'rejected':sum(r['rejected'] for r in valid),
                   'input_tokens_valid':sum(r['usage']['prompt_tokens'] for r in valid),'output_tokens_valid':sum(r['usage']['completion_tokens'] for r in valid)}
    valid_pairs=[r for r in rows if all(r[a]['status']=='valid' for a in protocol['arms'])]
    report={'queries':len(rows),'arms':arms,'matching_prompt_pairs':sum(r['default']['prompt_sha256'] is not None and r['default']['prompt_sha256']==r['compact']['prompt_sha256'] for r in rows),
            'valid_pairs':len(valid_pairs),'same_ordered_rankings':sum(r['default']['ranking']==r['compact']['ranking'] for r in valid_pairs),
            'mean_shared_retrieval_seconds':sum(v['seconds'] for v in prepared.values())/len(prepared),'pipeline_initialization_seconds':initialization,
            'corpus_snapshot_sha256':snapshot,'protocol_sha256':sha(output/'protocol.json'),'scope':protocol['scope']}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);main(parser.parse_args().output)
