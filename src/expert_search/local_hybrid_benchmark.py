"""Frozen local query encoding plus fixed RRF, full online-stage timing."""
import json
import time
from pathlib import Path
import numpy as np
from fastembed import TextEmbedding
from .corpus import records
from .benchmark import metrics
from .fast_retrieval import FastSearch
from .progressive_dense import top_indices
from .hybrid_benchmark import rrf


def run(output='artifacts/public/scifact-local-hybrid'):
    root=Path('artifacts/public');out=Path(output);out.mkdir(parents=True,exist_ok=True)
    matrix=np.load(root/'scifact-local-bge/vectors.npy')
    corpus=[r for _,r in records(root/'scifact-adapted/evidence.jsonl')];ids=[r['id'] for r in corpus]
    model=TextEmbedding(model_name='BAAI/bge-small-en-v1.5',cache_dir='artifacts/models',threads=1)
    lexical=FastSearch(root/'scifact-adapted',root/'scifact-results/fast.sqlite');rows=[]
    for _,q in records(root/'scifact-adapted/queries.jsonl'):
        p=out/f"query-{q['id']}.json"
        if p.exists():rows.append(json.loads(p.read_text()));continue
        start=time.perf_counter();v=np.asarray(list(model.query_embed(q['query']))[0],dtype=np.float32);v/=np.linalg.norm(v)
        scores=matrix@v;dense=[ids[i] for i in top_indices(scores,ids,100)]
        sparse=[r['id'] for r in lexical.search(q['query'],100)]
        ranked=rrf([dense,sparse]);elapsed=time.perf_counter()-start
        row=dict(query_id=q['id'],ranking=ranked,metrics=metrics(ranked,q['qrels']),seconds=elapsed)
        p.write_text(json.dumps(row,indent=2));rows.append(row)
    report=dict(model='BAAI/bge-small-en-v1.5 + BM25 RRF',queries=len(rows),depth=100,constant=60,
        metrics={key:float(np.mean([r['metrics'][key] for r in rows])) for key in rows[0]['metrics']},
        p50_seconds=float(np.median([r['seconds'] for r in rows])),p95_seconds=float(np.percentile([r['seconds'] for r in rows],95)),
        warning='Includes local query encoding, dense scoring, lexical retrieval and fusion; excludes model/index load. Fixed parameters, single sequential run, no training.')
    (out/'summary.json').write_text(json.dumps(report,indent=2));return report

if __name__=='__main__':print(json.dumps(run(),indent=2))
