"""Compare date-aware reasoning with and without paper citation context."""
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
from expert_search.metadata_overlay import MetadataOverlayPipeline
from expert_search.citation_overlay import CitationOverlayPipeline


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


class FrozenPipeline:
    def __init__(self,result):self.result=result
    def search_reranked(self,*args,**kwargs):return self.result


def main(output):
    prior=Path('artifacts/enterprise-agent-budget-pilot-v1/protocol.json');old=json.loads(prior.read_text())
    source=Path('artifacts/dataset-release/fresh-enterprise-eval-v1/queries.jsonl')
    if sha(source)!=old['queries_sha256']:raise ValueError('Changed query data')
    queries={r['id']:r for r in map(json.loads,source.read_text().splitlines()) if r['split']=='development'};ids=old['query_ids']
    if len(ids)!=4 or len(set(ids))!=4 or any(q not in queries for q in ids):raise ValueError('Changed cohort')
    output.mkdir(parents=True,exist_ok=False)
    paths=[Path(__file__),Path('src/expert_search/citation_overlay.py'),Path('src/expert_search/metadata_overlay.py'),Path('src/expert_search/candidate_reasoning_agent.py'),Path('src/expert_search/candidate_evidence_pipeline.py')]
    protocol={'query_ids':ids,'queries_sha256':sha(source),'prior_protocol_sha256':sha(prior),'model':'gpt-6-luna','retries':0,'workers':2,
              'arms':['without_citations','with_citations'],'arm_order':'alternating by query, serial within query','options':{'reasoning_effort':'none','max_completion_tokens':1500},
              'code_sha256':{str(p):sha(p) for p in paths},'scope':'Four reused enterprise development needs. Identical retrieved candidates, evidence text and publication metadata; the citation arm adds totals and a corresponding instruction together. No independent relevance, personal contribution, semantic entailment or historical validity labels.'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');load_dotenv('.env')
    start=time.perf_counter();pipeline=CandidateEvidencePipeline(scoring='resident-adaptive');init=time.perf_counter()-start
    date=MetadataOverlayPipeline(pipeline,'artifacts/corpus','artifacts/corpus-temporal-source-v1');citation=CitationOverlayPipeline(date,'artifacts/corpus','artifacts/source-metadata-v1','data/teachers/shenzhen/papers.jsonl')
    prepared={}
    for qid in ids:
        start=time.perf_counter();first=date.search_reranked(queries[qid]['text'],k=20,candidates=50,evidence_per_expert=1);second=citation._attach(first);elapsed=time.perf_counter()-start
        stripped=deepcopy(second['results'])
        for row in stripped:
            for e in row['evidence']:e.pop('citation_context',None)
        if stripped!=first['results']:raise ValueError('Changed source view or ranking')
        prepared[qid]={'without_citations':first,'with_citations':second,'retrieval_seconds':elapsed,'citation_by_expert':{r['expert']['id']:next((e['citation_context']['observed_total'] for e in r['evidence'] if 'citation_context' in e),None) for r in second['results']}}
    pipeline.evidence.db.close()
    (output/'view-hashes.json').write_text(json.dumps({q:{a:digest(prepared[q][a]['results']) for a in protocol['arms']} for q in ids},indent=2)+'\n')
    def run(qid,index):
        row={'query_id':qid,'arms':{}};client=LaiwenAI(model='gpt-6-luna',max_retries=0)
        for arm in (protocol['arms'] if index%2==0 else list(reversed(protocol['arms']))):
            agent=CandidateReasoningAgent(pipeline=FrozenPipeline(prepared[qid][arm]),client=client);agent.helper.request_options=protocol['options'];original=agent.helper.ask;capture={}
            def ask(stage,instruction,payload):
                base=deepcopy(payload)
                for e in base['evidence']:e.pop('citation_context',None)
                capture['common_payload_sha256']=digest(base);capture['full_payload_sha256']=digest(payload)
                return original(stage,instruction,payload)
            agent.helper.ask=ask;start=time.perf_counter();attempt={'status':'error'}
            try:
                result=agent.run(queries[qid]['text'],k=3,candidates=20,evidence_per_expert=1,retrieval='adaptive_bm25',rerank_pool=50)
                ranking=[x['expert']['id'] for x in result['results']];known=[prepared[qid]['citation_by_expert'][e] for e in ranking if prepared[qid]['citation_by_expert'][e] is not None]
                attempt.update(status='valid',result=result,ranking=ranking,completion=result['completion'],selected_known_citation_count=len(known),selected_known_citation_mean=statistics.mean(known) if known else None)
            except Exception as error:attempt.update(error_type=type(error).__name__,http_status=getattr(error,'status_code',None))
            attempt.update(seconds=time.perf_counter()-start,**capture);row['arms'][arm]=attempt
            print('Completed',index+1,arm,attempt['status'],flush=True)
        path=output/(hashlib.sha256(qid.encode()).hexdigest()+'.json');tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n');tmp.replace(path);return row
    with ThreadPoolExecutor(max_workers=2) as pool:rows=[f.result() for f in as_completed([pool.submit(run,qid,i) for i,qid in enumerate(ids)])]
    arms={a:[r['arms'][a] for r in rows] for a in protocol['arms']};joint=[r for r in rows if all(r['arms'][a]['status']=='valid' for a in protocol['arms'])]
    for row in rows:
        first=row['arms']['without_citations'].get('common_payload_sha256');second=row['arms']['with_citations'].get('common_payload_sha256')
        if first is not None and second is not None and first!=second:raise ValueError('Common model inputs differ')
    report={'queries':len(rows),'joint_valid':len(joint),'valid':{a:sum(r['status']=='valid' for r in arms[a]) for a in arms},
            'failures':{a:[{'error_type':r.get('error_type'),'http_status':r.get('http_status')} for r in arms[a] if r['status']!='valid'] for a in arms},
            'completion':{a:dict(Counter(r['completion']['status'] for r in arms[a] if r['status']=='valid')) for a in arms},
            'same_ordered_rankings':sum(r['arms']['without_citations']['ranking']==r['arms']['with_citations']['ranking'] for r in joint),
            'mean_top_three_overlap':statistics.mean(len(set(r['arms']['without_citations']['ranking'])&set(r['arms']['with_citations']['ranking']))/3 for r in joint) if joint else None,
            'selected_known_citation_counts':{a:sum(r['selected_known_citation_count'] for r in arms[a] if r['status']=='valid') for a in arms},
            'mean_attempt_seconds':{a:statistics.mean(r['seconds'] for r in arms[a]) for a in arms},'pipeline_init_seconds':init,
            'protocol_sha256':sha(output/'protocol.json'),'view_hashes_sha256':sha(output/'view-hashes.json'),'scope':protocol['scope']+' Small dependent pilot; prompt instruction and citation fields change together. Rank changes do not demonstrate accuracy, bias or causality. Model times exclude shared retrieval and initialization.'}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.output)
