"""Compare free-text and structured reasoning on identical retrieved views."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import statistics
import time

from dotenv import load_dotenv
from laiwenai import LaiwenAI
from expert_search.candidate_evidence_pipeline import CandidateEvidencePipeline
from expert_search.candidate_reasoning_agent import CandidateReasoningAgent
from expert_search.structured_candidate_agent import StructuredCandidateAgent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FrozenPipeline:
    def __init__(self,result):self.result=result
    def search_reranked(self,*args,**kwargs):return self.result


def main(output):
    prior=Path('artifacts/enterprise-agent-budget-pilot-v1/protocol.json');old=json.loads(prior.read_text())
    source=Path('artifacts/dataset-release/fresh-enterprise-eval-v1/queries.jsonl')
    if sha(source)!=old['queries_sha256']:raise ValueError('Changed query source')
    ids=old['query_ids'];queries={r['id']:r for r in map(json.loads,source.read_text().splitlines()) if r['split']=='development'}
    if len(ids)!=4 or len(set(ids))!=4 or any(q not in queries for q in ids):raise ValueError('Changed cohort')
    output.mkdir(parents=True,exist_ok=False)
    code=[Path(__file__),Path('src/expert_search/structured_candidate_agent.py'),Path('src/expert_search/partial_method_match.py'),Path('src/expert_search/candidate_reasoning_agent.py'),Path('src/expert_search/candidate_evidence_pipeline.py')]
    protocol={'query_ids':ids,'queries_sha256':sha(source),'prior_protocol_sha256':sha(prior),'model':'gpt-6-luna','retries':0,'workers':2,
              'candidate_pool':50,'candidates':5,'k':3,'source_chars':1200,'evidence_per_expert':1,'arms':['free_text','structured'],
              'options':{'reasoning_effort':'none','max_completion_tokens':2400},'arm_order':'alternating, serial per query',
              'code_sha256':{str(p):sha(p) for p in code},'scope':'Four previously studied private development needs. Identical candidates and source prefixes; output schemas and prompts differ. One attempt per arm; structural validity is not relevance, semantic support or verified personal contribution.'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');load_dotenv('.env');start=time.perf_counter()
    pipeline=CandidateEvidencePipeline(scoring='resident-adaptive');init=time.perf_counter()-start;prepared={}
    for qid in ids:
        start=time.perf_counter();r=pipeline.search_reranked(queries[qid]['text'],k=5,candidates=50,evidence_per_expert=1);seconds=time.perf_counter()-start
        r=deepcopy(r)
        for row in r['results']:
            for e in row['evidence']:e['text']=e['text'][:1200]
        prepared[qid]={'result':r,'seconds':seconds,'sha256':hashlib.sha256(json.dumps(r['results'],sort_keys=True,ensure_ascii=False).encode()).hexdigest()}
    pipeline.evidence.db.close();(output/'view-hashes.json').write_text(json.dumps({q:v['sha256'] for q,v in prepared.items()},indent=2)+'\n')
    def run(qid,index):
        client=LaiwenAI(model='gpt-6-luna',max_retries=0);attempts={}
        for arm in (protocol['arms'] if index%2==0 else list(reversed(protocol['arms']))):
            frozen=FrozenPipeline(prepared[qid]['result']);start=time.perf_counter();r={'status':'error'}
            try:
                if arm=='free_text':
                    agent=CandidateReasoningAgent(frozen,client);agent.helper.request_options=protocol['options']
                    result=agent.run(queries[qid]['text'],k=3,candidates=5,evidence_per_expert=1,retrieval='adaptive_bm25',rerank_pool=50)
                    r.update(status='valid',result=result,ranking=[x['expert']['id'] for x in result['results']],completion=result['completion'])
                else:
                    result=StructuredCandidateAgent(frozen,client).run(queries[qid]['text'],k=3,candidates=5,rerank_pool=50,evidence_per_expert=1,source_chars=1200)
                    if result['status']=='structurally_valid':r.update(status='valid',result=result,ranking=[x['expert_id'] for x in result['results']],completion=result['completion'])
                    else:r.update(status=result['status'],result=result,error_type=result.get('matcher',{}).get('error_type'))
            except Exception as error:r.update(error_type=type(error).__name__,http_status=getattr(error,'status_code',None))
            r['seconds']=time.perf_counter()-start;attempts[arm]=r;print('Completed',index+1,arm,r['status'],flush=True)
        path=output/(hashlib.sha256(qid.encode()).hexdigest()+'.json');tmp=path.with_suffix('.partial');tmp.write_text(json.dumps({'query_id':qid,'arms':attempts},ensure_ascii=False,indent=2)+'\n');tmp.replace(path)
        return {'query_id':qid,'arms':attempts}
    with ThreadPoolExecutor(max_workers=2) as pool:rows=[f.result() for f in as_completed([pool.submit(run,qid,i) for i,qid in enumerate(ids)])]
    arms={a:[r['arms'][a] for r in rows] for a in protocol['arms']};joint=[r for r in rows if all(r['arms'][a]['status']=='valid' for a in protocol['arms'])]
    levels=Counter(x['level'] for r in arms['structured'] if r['status']=='valid' for x in r['result']['results'])
    report={'queries':len(rows),'valid':{a:sum(r['status']=='valid' for r in arms[a]) for a in arms},
            'statuses':{a:dict(Counter(r['status'] for r in arms[a])) for a in arms},
            'errors':{a:[r.get('error_type') for r in arms[a] if r['status']!='valid'] for a in arms},
            'completion':{a:dict(Counter(r['completion']['status'] for r in arms[a] if r['status']=='valid')) for a in arms},
            'joint_valid':len(joint),'same_ordered_rankings':sum(r['arms']['free_text']['ranking']==r['arms']['structured']['ranking'] for r in joint),
            'mean_top3_overlap':statistics.mean(len(set(r['arms']['free_text']['ranking'])&set(r['arms']['structured']['ranking']))/3 for r in joint) if joint else None,
            'structured_levels':dict(levels),'mean_model_seconds':{a:statistics.mean(r['seconds'] for r in arms[a]) for a in arms},
            'mean_shared_retrieval_seconds':statistics.mean(r['seconds'] for r in prepared.values()),'pipeline_init_seconds':init,
            'protocol_sha256':sha(output/'protocol.json'),'view_hashes_sha256':sha(output/'view-hashes.json'),
            'scope':protocol['scope']+' Timing excludes initialization and shared retrieval; schemas, prompt and token needs differ. Levels and quote offsets are not independently judged relevance or entailment.'}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.output)
