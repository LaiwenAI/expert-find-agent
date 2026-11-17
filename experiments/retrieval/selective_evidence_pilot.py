"""Compare exact evidence lookup paths on shared real-query candidate pools."""
import argparse
import json
import statistics
import time
from pathlib import Path

from expert_search.fast_retrieval import FastSearch
from .enterprise_end_to_end_pilot import sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    data=Path('artifacts/dataset-release/fresh-enterprise-eval-v1/queries.jsonl')
    queries=sorted((r for r in map(json.loads,data.read_text().splitlines()) if r['split']=='development'),key=lambda r:r['id'])[:20]
    if len(queries)!=20:raise ValueError('Expected 20 development queries')
    old=FastSearch('artifacts/corpus','artifacts/fast-search.sqlite')
    new=FastSearch('artifacts/corpus','artifacts/fast-search-selective-v1.sqlite')
    args.output.mkdir(parents=True,exist_ok=False)
    protocol={'query_file_sha256':sha(data),'baseline_db_sha256':sha(Path('artifacts/fast-search.sqlite')),
              'selective_db_sha256':sha(Path('artifacts/fast-search-selective-v1.sqlite')),
              'code_sha256':sha(Path('src/expert_search/fast_retrieval.py')),
              'scope':'20 lexicographic development queries, one call per arm with alternating order. Both use the same lexical candidate pool; retrieval, index construction and startup excluded.'}
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    rows=[]
    for i,q in enumerate(queries):
        ids=[r['id'] for r in old.search(q['text'],20)];result={};times={}
        for name,obj in ([('baseline',old),('selective',new)] if i%2==0 else [('selective',new),('baseline',old)]):
            start=time.perf_counter();result[name]=(obj.evidence_for(q['text'],ids) if name=='baseline' else obj.evidence_for_selective(q['text'],ids));times[name]=time.perf_counter()-start
        rows.append({'query_id':q['id'],'candidate_count':len(ids),'identical':result['baseline']==result['selective'],'seconds':times})
    (args.output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps({'queries':len(rows),'identical':sum(r['identical'] for r in rows),
                      'mean_baseline_seconds':statistics.mean(r['seconds']['baseline'] for r in rows),
                      'mean_selective_seconds':statistics.mean(r['seconds']['selective'] for r in rows),'scope':protocol['scope']},indent=2))


if __name__=='__main__':main()
