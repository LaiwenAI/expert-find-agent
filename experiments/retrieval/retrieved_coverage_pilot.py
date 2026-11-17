"""Compare evidence budgets on shared live lexical candidate pools, development only."""
import argparse
import hashlib
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from expert_search.agent import ExpertAgent
from .enterprise_end_to_end_pilot import sha
from .fixed_evidence_routing import run_case


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--offset',type=int,default=4);parser.add_argument('--count',type=int,default=4)
    args=parser.parse_args()
    if args.offset<0 or args.count<1:raise ValueError('Invalid sample range')
    query_file=Path('artifacts/dataset-release/fresh-enterprise-eval-v1/queries.jsonl')
    all_queries=sorted((r for r in map(json.loads,query_file.read_text().splitlines()) if r['split']=='development'),key=lambda r:r['id'])
    queries=all_queries[args.offset:args.offset+args.count]
    if len(queries)!=args.count:raise ValueError('Insufficient queries')
    paths=['experiments/retrieval/retrieved_coverage_pilot.py','experiments/retrieval/fixed_evidence_routing.py',
           'experiments/retrieval/enterprise_end_to_end_pilot.py','src/expert_search/agent.py',
           'src/expert_search/fast_retrieval.py','src/expert_search/retrieval.py']
    args.output.mkdir(parents=True,exist_ok=False)
    protocol={'query_ids':[r['id'] for r in queries],'query_file_sha256':sha(query_file),
              'code_sha256':{p:sha(Path(p)) for p in paths},
              'input_sha256':{p:sha(Path(p)) for p in ['artifacts/search.sqlite','artifacts/fast-search.sqlite',
                                'artifacts/corpus/evidence.jsonl','artifacts/corpus/experts.jsonl']},
              'arms':['direct','coverage_direct'],'order':'alternating','model':'gpt-6-luna','k':5,'candidates':20,
              'scope':'Development only; one retrieved pool per query shared by both arms, but displayed evidence differs. Timing excludes retrieval, initialization and index build; no relevance labels read.'}
    load_dotenv('.env');agent=ExpertAgent(compiled_database='artifacts/fast-search.sqlite')
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for i,q in enumerate(queries):
        started=time.perf_counter();hits=agent.searcher.search(q['text'],20);retrieval_seconds=time.perf_counter()-started
        frozen=json.dumps(hits,sort_keys=True,ensure_ascii=False).encode();digest=hashlib.sha256(frozen).hexdigest()
        (args.output/(q['id']+'-pool.json')).write_bytes(frozen)
        for arm in (protocol['arms'] if i%2==0 else list(reversed(protocol['arms']))):
            started=time.perf_counter()
            try:row=run_case(agent,q['text'],hits,arm,k=5);row['status']='ok'
            except Exception as error:row={'arm':arm,'status':'error','error_type':type(error).__name__,'seconds':time.perf_counter()-started}
            row.update(query_id=q['id'],input_sha256=digest,retrieval_seconds=retrieval_seconds)
            (args.output/(q['id']+'-'+arm+'.json')).write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')
            print(q['id'],arm,row['status'],flush=True)


if __name__=='__main__':main()
