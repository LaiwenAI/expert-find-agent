"""Repeat frozen development queries to inspect within-query timing variation."""
import argparse
import json
import random
import time
from pathlib import Path

from dotenv import load_dotenv
from expert_search.agent import ExpertAgent
from expert_search.candidate_reasoning_agent import CandidateReasoningAgent
from expert_search.single_pass_agent import SinglePassAgent
from .enterprise_end_to_end_pilot import sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    original=Path('artifacts/candidate-agent-pilot-v1/protocol.json');p=json.loads(original.read_text())
    qpath=Path('artifacts/dataset-release/fresh-enterprise-eval-v1/queries.jsonl')
    if sha(qpath)!=p['query_sha256']:raise ValueError('Changed queries')
    for path,digest in {**p['inputs_sha256'],**p['code_sha256']}.items():
        if sha(Path(path))!=digest:raise ValueError('Changed prior input or code: '+path)
    for name,digest in p['model_files_sha256'].items():
        if sha(Path('artifacts/teacher-compiled-lm')/name)!=digest:raise ValueError('Changed candidate index')
    queries={r['id']:r for r in map(json.loads,qpath.read_text().splitlines()) if r['split']=='development'}
    if any(qid not in queries for qid in p['query_ids']):raise ValueError('Changed query split')
    rng=random.Random(20261123);schedule=[]
    for repeat in range(2):
        for qid in p['query_ids']:
            arms=list(p['arms']);rng.shuffle(arms)
            schedule.extend({'query_id':qid,'arm':arm,'repeat':repeat} for arm in arms)
    args.output.mkdir(parents=True,exist_ok=False)
    protocol={'prior_protocol_sha256':sha(original),'code_sha256':sha(Path(__file__)),'seed':20261123,'schedule':schedule,
              'scope':'Two additional attempts per arm and development query, randomized order within each pair; sequential requests. Same reused needs and different candidate sets, no independent relevance labels. Excludes initialization.'}
    load_dotenv('.env');baseline=ExpertAgent(compiled_database='artifacts/fast-search.sqlite');candidate=CandidateReasoningAgent(client=baseline.client)
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for item in schedule:
        qid,arm,repeat=item['query_id'],item['arm'],item['repeat'];start=time.perf_counter()
        try:
            result=candidate.run(queries[qid]['text'],k=5,candidates=20) if arm=='candidate_first' else SinglePassAgent.run(baseline,queries[qid]['text'],k=5,candidates=20)
            row={'status':'ok','result':result}
        except Exception as error:row={'status':'error','error_type':type(error).__name__}
        row.update(item,attempt_seconds=time.perf_counter()-start)
        (args.output/(qid+'-'+arm+'-'+str(repeat)+'.json')).write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')
        print(qid,arm,repeat,row['status'],flush=True)


if __name__=='__main__':main()
