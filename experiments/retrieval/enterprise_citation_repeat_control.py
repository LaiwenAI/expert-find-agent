"""Repeat the no-citation arm to assess same-input ranking variability."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
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
from experiments.retrieval.enterprise_citation_context_pilot import FrozenPipeline,digest,sha


def main(output):
    previous=Path('artifacts/enterprise-citation-context-pilot-v1');protocol=json.loads((previous/'protocol.json').read_text());summary=json.loads((previous/'summary.json').read_text())
    if sha(previous/'protocol.json')!=summary['protocol_sha256'] or sha(previous/'view-hashes.json')!=summary['view_hashes_sha256']:raise ValueError('Changed previous pilot')
    for name,value in protocol['code_sha256'].items():
        if sha(name)!=value:raise ValueError('Changed previous code')
    source=Path('artifacts/dataset-release/fresh-enterprise-eval-v1/queries.jsonl')
    if sha(source)!=protocol['queries_sha256']:raise ValueError('Changed query data')
    ids=protocol['query_ids'];queries={r['id']:r for r in map(json.loads,source.read_text().splitlines()) if r['split']=='development'}
    if len(ids)!=4 or any(q not in queries for q in ids):raise ValueError('Changed cohort')
    previous_rows={};hashes={}
    for qid in ids:
        path=previous/(hashlib.sha256(qid.encode()).hexdigest()+'.json');row=json.loads(path.read_text());hashes[path.name]=sha(path)
        if row['query_id']!=qid or row['arms']['without_citations']['status']!='valid':raise ValueError('Prior control unavailable')
        previous_rows[qid]=row
    output.mkdir(parents=True,exist_ok=False)
    frozen={'query_ids':ids,'queries_sha256':sha(source),'prior_protocol_sha256':sha(previous/'protocol.json'),'prior_results_sha256':hashes,'model':'gpt-6-luna','retries':0,'workers':2,
            'options':protocol['options'],'code_sha256':{str(p):sha(p) for p in (Path(__file__),Path('src/expert_search/candidate_reasoning_agent.py'),Path('src/expert_search/candidate_evidence_pipeline.py'),Path('src/expert_search/metadata_overlay.py'))},
            'scope':'Same four previously studied development needs and same no-citation model input, reissued later. No independent fit labels, no randomization against the prior paired calls, and no assessment of individual contribution.'}
    (output/'protocol.json').write_text(json.dumps(frozen,indent=2)+'\n');load_dotenv('.env');start=time.perf_counter()
    pipeline=CandidateEvidencePipeline(scoring='resident-adaptive');init=time.perf_counter()-start;overlay=MetadataOverlayPipeline(pipeline,'artifacts/corpus','artifacts/corpus-temporal-source-v1')
    old_hashes=json.loads((previous/'view-hashes.json').read_text());prepared={}
    for qid in ids:
        result=overlay.search_reranked(queries[qid]['text'],k=20,candidates=50,evidence_per_expert=1)
        if digest(result['results'])!=old_hashes[qid]['without_citations']:raise ValueError('Changed candidate views')
        prepared[qid]=result
    pipeline.evidence.db.close()
    def run(qid):
        agent=CandidateReasoningAgent(FrozenPipeline(prepared[qid]),client=LaiwenAI(model='gpt-6-luna',max_retries=0));agent.helper.request_options=frozen['options'];ask=agent.helper.ask;capture={}
        def verify(stage,instruction,payload):
            h=digest(payload);capture['payload_sha256']=h
            if h!=previous_rows[qid]['arms']['without_citations']['full_payload_sha256']:raise ValueError('Changed model payload')
            return ask(stage,instruction,payload)
        agent.helper.ask=verify;start=time.perf_counter();row={'query_id':qid,'status':'error'}
        try:
            result=agent.run(queries[qid]['text'],k=3,candidates=20,evidence_per_expert=1,retrieval='adaptive_bm25',rerank_pool=50)
            row.update(status='valid',result=result,ranking=[x['expert']['id'] for x in result['results']],completion=result['completion'])
        except Exception as error:row.update(error_type=type(error).__name__,http_status=getattr(error,'status_code',None))
        row.update(seconds=time.perf_counter()-start,**capture);path=output/(hashlib.sha256(qid.encode()).hexdigest()+'.json');tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)
        print('Completed repeat',row['status'],flush=True);return row
    with ThreadPoolExecutor(max_workers=2) as pool:rows=[f.result() for f in as_completed([pool.submit(run,q) for q in ids])]
    valid=[r for r in rows if r['status']=='valid'];joint=[]
    for row in valid:
        prior=previous_rows[row['query_id']]['arms']['without_citations']['ranking'];citation=previous_rows[row['query_id']]['arms']['with_citations']['ranking'];repeat=row['ranking']
        joint.append({'repeat_equals_prior':repeat==prior,'citation_equals_prior':citation==prior,
                      'repeat_overlap':len(set(repeat)&set(prior))/3,'citation_overlap':len(set(citation)&set(prior))/3})
    report={'attempts':len(rows),'valid':len(valid),'failures':[{'error_type':r.get('error_type'),'http_status':r.get('http_status')} for r in rows if r['status']!='valid'],
            'completion':dict(Counter(r['completion']['status'] for r in valid)),
            'same_ordered_repeat':sum(r['repeat_equals_prior'] for r in joint),'same_ordered_citation':sum(r['citation_equals_prior'] for r in joint),
            'mean_repeat_overlap':statistics.mean(r['repeat_overlap'] for r in joint) if joint else None,'mean_citation_overlap':statistics.mean(r['citation_overlap'] for r in joint) if joint else None,
            'mean_attempt_seconds':statistics.mean(r['seconds'] for r in rows),'pipeline_init_seconds':init,'protocol_sha256':sha(output/'protocol.json'),
            'scope':frozen['scope']+' Overlap uses fixed denominator three and counts missing outputs as missing slots. Sequential nonrandomized repeats do not isolate citation effects or prove accuracy, equivalence or bias.'}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.output)
