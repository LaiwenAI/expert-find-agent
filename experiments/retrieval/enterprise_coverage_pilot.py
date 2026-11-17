"""Development-only fixed-snapshot coverage pilot; never reads judgments."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from expert_search.agent import ExpertAgent
from .fixed_evidence_routing import run_case


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--count',type=int,default=8);args=parser.parse_args()
    if args.count<1:raise ValueError('Count must be positive')
    data=Path('artifacts/dataset-release/fresh-enterprise-eval-v1')
    names=['queries.jsonl','pools.jsonl','snapshots.jsonl','corpus.jsonl']
    hashes={name:hashlib.sha256((data/name).read_bytes()).hexdigest() for name in names}
    queries=sorted((q for q in read(data/'queries.jsonl') if q['split']=='development'),key=lambda q:q['id'])[:args.count]
    if len(queries)!=args.count:raise ValueError('Insufficient development queries')
    pools={x['query_id']:x for x in read(data/'pools.jsonl') if x['split']=='development'}
    snapshots={(x['query_id'],x['expert_id']):x for x in read(data/'snapshots.jsonl') if x['split']=='development'}
    corpus={x['id']:x for x in read(data/'corpus.jsonl')}
    args.output.mkdir(parents=True,exist_ok=False)
    protocol={'query_ids':[q['id'] for q in queries],'inputs_sha256':hashes,'arms':['direct','coverage_direct'],
              'order':'alternating','model':'gpt-6-luna',
              'code_sha256':{name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in [__file__,'experiments/retrieval/fixed_evidence_routing.py','src/expert_search/agent.py']},
              'scope':'Development only; frozen snapshot prefixes; no judgments read; excludes retrieval and initialization.'}
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    load_dotenv('.env');agent=ExpertAgent(compiled_database='artifacts/fast-search.sqlite')
    for i,q in enumerate(queries):
        hits=[]
        for eid in pools[q['id']]['candidates']:
            snap=snapshots[q['id'],eid];evidence=[]
            for span in snap['sources']:
                src=corpus[span['source_id']];text=src['text'][span['start']:span['end']]
                if src['expert_id']!=eid or hashlib.sha256(text.encode()).hexdigest()!=span['sha256']:raise ValueError('Invalid source ownership or span hash')
                evidence.append({'id':src['id'],'expert_id':eid,'text':text})
            hits.append({'id':eid,'evidence':evidence})
        digest=hashlib.sha256(json.dumps(hits,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        for arm in (['direct','coverage_direct'] if i%2==0 else ['coverage_direct','direct']):
            start=time.perf_counter()
            try:row=run_case(agent,q['text'],hits,arm,k=7);row['status']='ok'
            except Exception as error:row={'arm':arm,'status':'error','error_type':type(error).__name__,'seconds':time.perf_counter()-start}
            row.update(query_id=q['id'],input_sha256=digest)
            (args.output/(q['id']+'-'+arm+'.json')).write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
