"""Run a small alternating-order fixed-candidate evidence coverage pilot."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from expert_search.agent import ExpertAgent
from .fixed_evidence_routing import run_case


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--start',type=int,default=4);parser.add_argument('--count',type=int,default=4);args=parser.parse_args()
    if args.start<0 or args.count<1:raise ValueError('Invalid query slice')
    root=args.output;root.mkdir(parents=True,exist_ok=False)
    queries=[]
    for line in Path('artifacts/benchmark/runs.jsonl').read_text().splitlines():
        row=json.loads(line)
        if row['query_id'] not in {q['query_id'] for q in queries}:queries.append(row)
        if len(queries)==args.start+args.count:break
    if len(queries)!=args.start+args.count:raise ValueError('Insufficient distinct benchmark queries')
    queries=queries[args.start:]
    protocol={'query_ids':[q['query_id'] for q in queries], 'model':'gpt-6-luna',
              'arms':['direct','coverage_direct'], 'order':'alternating by query',
              'files_sha256':{p:sha(p) for p in ['artifacts/benchmark/runs.jsonl','artifacts/benchmark/manifest.json','artifacts/fast-search.sqlite','src/expert_search/agent.py','experiments/retrieval/fixed_evidence_routing.py']},
              'scope':'Proxy queries; rankings use different evidence subsets; excludes retrieval and initialization.'}
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    load_dotenv('.env');agent=ExpertAgent(compiled_database='artifacts/fast-search.sqlite')
    excluded=json.loads(Path('artifacts/benchmark/manifest.json').read_text())['excluded_evidence']
    for i,q in enumerate(queries):
        hits=agent.searcher.search(q['query'],20,exclude=excluded)
        digest=hashlib.sha256(json.dumps(hits,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        for arm in (['direct','coverage_direct'] if i%2==0 else ['coverage_direct','direct']):
            start=time.perf_counter()
            try:row=run_case(agent,q['query'],hits,arm);row['status']='ok'
            except Exception as error:row={'arm':arm,'status':'error','error_type':type(error).__name__,'seconds':time.perf_counter()-start}
            row.update(query_id=q['query_id'],input_sha256=digest)
            (root/(q['query_id']+'-'+arm+'.json')).write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
