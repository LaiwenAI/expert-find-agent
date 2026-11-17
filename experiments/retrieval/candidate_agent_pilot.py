"""Compare two one-call full-search paths on development needs."""
import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from expert_search.agent import ExpertAgent
from expert_search.candidate_reasoning_agent import CandidateReasoningAgent
from expert_search.single_pass_agent import SinglePassAgent
from .enterprise_end_to_end_pilot import sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    qpath=Path('artifacts/dataset-release/fresh-enterprise-eval-v1/queries.jsonl')
    queries=sorted((r for r in map(json.loads,qpath.read_text().splitlines()) if r['split']=='development'),key=lambda r:r['id'])[8:12]
    if len(queries)!=4:raise ValueError('Insufficient development queries')
    inputs=['artifacts/search.sqlite','artifacts/fast-search.sqlite','artifacts/fast-search-selective-v1.sqlite','artifacts/corpus/evidence.jsonl','artifacts/corpus/experts.jsonl']
    code=['experiments/retrieval/candidate_agent_pilot.py','src/expert_search/candidate_reasoning_agent.py','src/expert_search/candidate_evidence_pipeline.py',
          'src/expert_search/agent.py','src/expert_search/single_pass_agent.py','src/expert_search/fast_retrieval.py','src/expert_search/compiled_expert_search.py','src/expert_search/certified_topk.py']
    model_dir=Path('artifacts/teacher-compiled-lm');args.output.mkdir(parents=True,exist_ok=False)
    protocol={'query_ids':[q['id'] for q in queries],'query_sha256':sha(qpath),'inputs_sha256':{p:sha(Path(p)) for p in inputs},
              'code_sha256':{p:sha(Path(p)) for p in code},'model_files_sha256':{p.name:sha(p) for p in model_dir.iterdir() if p.is_file()},
              'arms':['candidate_first','single_pass'],'order':'alternating','model':'gpt-6-luna',
              'scope':'Four development needs; different shortlist ranking, evidence budgets and payloads. One attempt per arm; includes retrieval and reasoning, excludes initialization and index builds. No judgments read.'}
    load_dotenv('.env');baseline=ExpertAgent(compiled_database='artifacts/fast-search.sqlite');candidate=CandidateReasoningAgent(client=baseline.client)
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for i,q in enumerate(queries):
        for arm in (protocol['arms'] if i%2==0 else list(reversed(protocol['arms']))):
            start=time.perf_counter()
            try:
                result=candidate.run(q['text'],k=5,candidates=20) if arm=='candidate_first' else SinglePassAgent.run(baseline,q['text'],k=5,candidates=20)
                row={'status':'ok','result':result}
            except Exception as error:row={'status':'error','error_type':type(error).__name__}
            row.update(query_id=q['id'],arm=arm,attempt_seconds=time.perf_counter()-start)
            (args.output/(q['id']+'-'+arm+'.json')).write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')
            print(q['id'],arm,row['status'],flush=True)


if __name__=='__main__':main()
