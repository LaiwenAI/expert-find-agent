"""Run a development-only paired full-search pilot without reading judgments."""
import argparse
import hashlib
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from expert_search.agent import ExpertAgent
from expert_search.single_pass_agent import SinglePassAgent


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--count',type=int,default=4);args=parser.parse_args()
    if args.count<1:raise ValueError('Count must be positive')
    query_file=Path('artifacts/dataset-release/fresh-enterprise-eval-v1/queries.jsonl')
    queries=sorted((r for r in map(json.loads,query_file.read_text().splitlines()) if r['split']=='development'),key=lambda r:r['id'])[:args.count]
    if len(queries)!=args.count:raise ValueError('Insufficient development queries')
    paths=['experiments/retrieval/enterprise_end_to_end_pilot.py','src/expert_search/agent.py',
           'src/expert_search/single_pass_agent.py','src/expert_search/fast_retrieval.py',
           'src/expert_search/retrieval.py','src/expert_search/expert_catalog.py']
    args.output.mkdir(parents=True,exist_ok=False)
    protocol={'query_ids':[r['id'] for r in queries],'query_file_sha256':sha(query_file),
              'code_sha256':{p:sha(Path(p)) for p in paths},'model':'gpt-6-luna',
              'input_sha256':{p:sha(Path(p)) for p in ['artifacts/search.sqlite','artifacts/fast-search.sqlite',
                                'artifacts/corpus/evidence.jsonl','artifacts/corpus/experts.jsonl']},
              'arms':['single_pass','tree_agent'],'arm_order':'alternating',
              'k':5,'candidates':20,'scope':'Development only; one attempt per arm and query, excludes agent initialization and index construction. Candidate pools differ. No judgments read.'}
    load_dotenv('.env');agent=ExpertAgent(compiled_database='artifacts/fast-search.sqlite')
    protocol['catalog_sha256']=sha(agent.catalog.database)
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for i,q in enumerate(queries):
        order=protocol['arms'] if i%2==0 else list(reversed(protocol['arms']))
        for arm in order:
            start=time.perf_counter()
            try:
                result=(SinglePassAgent.run(agent,q['text'],k=5,candidates=20) if arm=='single_pass' else agent.run(q['text'],k=5,candidates=20))
                record={'status':'ok','result':result}
            except Exception as error:
                record={'status':'error','error_type':type(error).__name__,'trace':agent.trace,'usage':agent.usage}
            record.update(query_id=q['id'],arm=arm,attempt_seconds=time.perf_counter()-start)
            (args.output/(q['id']+'-'+arm+'.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
            print(q['id'],arm,record['status'],flush=True)


if __name__=='__main__':main()
