"""Run progressive retrieval and explanations on fixed enterprise needs."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

from dotenv import load_dotenv
from laiwenai import LaiwenAI
from expert_search.candidate_evidence_pipeline import CandidateEvidencePipeline
from expert_search.progressive_candidate_agent import ProgressiveCandidateAgent


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main(output):
    prior=Path('artifacts/enterprise-agent-budget-pilot-v1/protocol.json');protocol0=json.loads(prior.read_text())
    query_file=Path('artifacts/dataset-release/fresh-enterprise-eval-v1/queries.jsonl')
    if sha(query_file)!=protocol0['queries_sha256']:raise ValueError('Changed query data')
    queries={r['id']:r for r in map(json.loads,query_file.read_text().splitlines()) if r['split']=='development'};ids=protocol0['query_ids']
    if len(ids)!=4 or len(set(ids))!=4 or set(ids)-set(queries):raise ValueError('Changed cohort')
    output.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),Path('src/expert_search/progressive_candidate_agent.py'),Path('src/expert_search/candidate_reasoning_agent.py'),Path('src/expert_search/candidate_evidence_pipeline.py')]
    protocol={'query_ids':ids,'query_sha256':sha(query_file),'prior_protocol_sha256':sha(prior),'model':'gpt-6-luna','no_automatic_retries':True,
              'candidate_pool':50,'explanation_candidates':20,'k':5,'execution':'Sequential; one first-result event followed by one model call per query.',
              'code_sha256':{str(p):sha(p) for p in files},
              'scope':'Four reused development needs, one attempt per query. Initial candidates are unverified; later explanation quote ownership is structurally checked, not semantic support. No independent quality judgments; initialization and index building measured separately.'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');load_dotenv('.env');start=time.perf_counter()
    pipeline=CandidateEvidencePipeline(scoring='resident');init=time.perf_counter()-start
    agent=ProgressiveCandidateAgent(pipeline,client=LaiwenAI(model='gpt-6-luna',max_retries=0));rows=[]
    for i,qid in enumerate(ids):
        events=agent.events(queries[qid]['text'],k=5,candidates=20,rerank_pool=50,evidence_per_expert=1)
        first=next(events);final=next(events)
        if first['stage']!='retrieval' or final['stage'] not in ('explanation','explanation_failed','explanation_unavailable'):raise ValueError('Invalid event sequence')
        initial=[r['expert']['id'] for r in first['results']];later=[r['expert']['id'] for r in final.get('results',[])]
        row={'query_id':qid,'first':first,'final':final,'first_ids':initial,'final_ids':later,
             'ordered_equal':initial==later if final['stage']=='explanation' else None,
             'set_overlap':len(set(initial)&set(later)) if final['stage']=='explanation' else None}
        (output/(qid+'.json')).write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n');rows.append(row)
        print('Completed query',i+1,final['stage'],flush=True)
    pipeline.evidence.db.close();valid=[r for r in rows if r['final']['stage']=='explanation']
    report={'queries':len(rows),'first_events':len(rows),'final_states':{s:sum(r['final']['stage']==s for r in rows) for s in sorted({r['final']['stage'] for r in rows})},
            'final_completion':{s:sum(r['final']['completion']['status']==s for r in valid) for s in ('complete','partial','empty')},
            'mean_first_seconds':statistics.mean(r['first']['seconds'] for r in rows),
            'mean_total_seconds':statistics.mean(r['final']['seconds'] for r in rows),
            'same_ordered_rankings':sum(r['ordered_equal'] for r in valid),'mean_top5_set_overlap':statistics.mean(r['set_overlap'] for r in valid) if valid else None,
            'pipeline_initialization_seconds':init,'resident_build_seconds':pipeline.resident.initialization_seconds,
            'resident_array_bytes':pipeline.resident.array_bytes,'corpus_snapshot_sha256':pipeline.ranker.snapshot,'protocol_sha256':sha(output/'protocol.json'),'scope':protocol['scope']}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
