"""One-query smoke test of progressive native screening and explanation."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from dotenv import load_dotenv
from laiwenai import LaiwenAI
from expert_search.candidate_evidence_pipeline import CandidateEvidencePipeline
from expert_search.progressive_decision_agent import ProgressiveDecisionAgent


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main(output):
    prior=Path('artifacts/enterprise-agent-budget-pilot-v1/protocol.json');p=json.loads(prior.read_text())
    source=Path('artifacts/dataset-release/fresh-enterprise-eval-v1/queries.jsonl')
    if sha(source)!=p['queries_sha256']:raise ValueError('Changed need source')
    ids=p['query_ids'];queries={r['id']:r for r in map(json.loads,source.read_text().splitlines()) if r['split']=='development'}
    if len(ids)!=4 or ids[0] not in queries:raise ValueError('Changed smoke cohort')
    output.mkdir(parents=True,exist_ok=False)
    code=[Path(__file__),Path('src/expert_search/progressive_decision_agent.py'),Path('src/expert_search/candidate_reasoning_agent.py'),Path('src/expert_search/choice_judgments.py'),Path('src/expert_search/candidate_evidence_pipeline.py')]
    protocol={'query_id':ids[0],'queries_sha256':sha(source),'prior_protocol_sha256':sha(prior),'model':'gpt-6-luna','automatic_retries':0,
              'options':{'reasoning_effort':'none','max_completion_tokens':1500},'code_sha256':{str(p):sha(p) for p in code},
              'scope':'One previously inspected development need, not a quality benchmark. Native Decisions screening and at most three quote-anchored explanations. Events produced on demand; no independent relevance, entailment or personal contribution labels.'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');load_dotenv('.env');start=time.perf_counter()
    pipeline=CandidateEvidencePipeline(scoring='resident');init=time.perf_counter()-start
    agent=ProgressiveDecisionAgent(pipeline=pipeline,client=LaiwenAI(model='gpt-6-luna',max_retries=0),explanation_options=protocol['options'])
    events=[]
    for event in agent.events(queries[ids[0]]['text'],k=3,candidates=20,rerank_pool=50,explanation_candidates=5,evidence_per_expert=1):
        events.append(event);print('Completed stage',event['stage'],flush=True)
    pipeline.evidence.db.close();(output/'private-events.json').write_text(json.dumps(events,ensure_ascii=False,indent=2)+'\n')
    report={'stages':[r['stage'] for r in events],'seconds':{r['stage']:r['seconds'] for r in events},'initial_candidates':len(events[0]['results']),
            'selected':next((r['selected'] for r in events if r['stage']=='screen'),None),
            'explanation_completion':next((r['completion'] for r in events if r['stage']=='explanation'),None),
            'failure_types':[r['error_type'] for r in events if 'error_type' in r],
            'pipeline_initialization_seconds':init,'corpus_snapshot_sha256':pipeline.ranker.snapshot,
            'protocol_sha256':sha(output/'protocol.json'),'events_sha256':sha(output/'private-events.json'),'scope':protocol['scope']}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
