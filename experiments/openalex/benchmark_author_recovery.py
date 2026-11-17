"""Run lexical baselines against the frozen OpenAlex author-recovery proxy."""
import hashlib
import argparse
import json
import statistics
import time
from pathlib import Path

from expert_search.retrieval import Search,index
from .verify_author_recovery import verify


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('artifacts/openalex-author-recovery-benchmark-v3'));args=parser.parse_args()
    data=Path('benchmarks/openalex-author-recovery-v1');verify(data)
    output=args.output;output.mkdir(parents=True,exist_ok=False)
    database=output/'index.sqlite';start=time.perf_counter();index(data,database);build_seconds=time.perf_counter()-start
    engine=Search(data,database);queries=[json.loads(line) for line in (data/'queries.jsonl').read_text().splitlines()]
    labels={r['query_id']:set(r['known_author_ids']) for r in map(json.loads,(data/'qrels.jsonl').read_text().splitlines())}
    methods=('bm25-max','bm25-vote','tfidf-max');rows=[]
    for i,q in enumerate(queries):
        run={}
        for method in methods[i%3:]+methods[:i%3]:
            started=time.perf_counter();hits=engine.search(q['text'],10,method);seconds=time.perf_counter()-started
            rank=[r['id'] for r in hits];truth=labels[q['id']];matched=[j for j,eid in enumerate(rank,1) if eid in truth]
            run[method]={'seconds':seconds,'returned':len(rank),'hit_at_10':int(bool(matched)),
                         'mrr_at_10':1/min(matched) if matched else 0.,'known_author_recall_at_10':len(set(rank)&truth)/len(truth)}
        rows.append({'query_id':q['id'],'methods':run})
    summary={'queries':len(rows),'index_build_seconds':build_seconds,
             'methods':{m:{metric:statistics.mean(r['methods'][m][metric] for r in rows) for metric in ('seconds','returned','hit_at_10','mrr_at_10','known_author_recall_at_10')} for m in methods},
             'dataset_manifest_sha256':hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest(),
             'code_sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (__file__,'src/expert_search/retrieval.py','experiments/openalex/verify_author_recovery.py')},
             'scope':'Author-recovery proxy; positives are OpenAlex target-paper authors with historical records. Non-authors are unjudged, not negatives. Overlapping-topic API-selected queries, one rotating-order timing per method; excludes initialization, semantic encoding and model reasoning. Residual semantic leakage and OpenAlex author disambiguation not audited.'}
    (output/'results.json').write_text(json.dumps(rows,indent=2)+'\n');(output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
