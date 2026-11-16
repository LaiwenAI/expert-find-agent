"""Offline-query-vector component evaluation; does not measure online embedding latency."""
import json
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
from .benchmark import metrics
from .corpus import records
from .fast_retrieval import FastSearch
from .progressive_dense import progressive_search


def rrf(rankings,k=10,constant=60):
    scores=defaultdict(float)
    for ranking in rankings:
        if len(ranking)!=len(set(ranking)):raise ValueError('Duplicate IDs in ranking')
        for rank,eid in enumerate(ranking,1):scores[eid]+=1/(constant+rank)
    return sorted(scores,key=lambda eid:(-scores[eid],eid))[:k]


def run(root='artifacts/public',output='artifacts/public/scifact-hybrid-results'):
    root=Path(root);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    dm=json.loads((root/'scifact-dense/manifest.json').read_text())
    qm=json.loads((root/'scifact-query-dense/manifest.json').read_text())
    if dm['model']!=qm['model']:raise ValueError('Embedding models differ')
    docs=np.load(root/'scifact-dense/vectors.npy');queries=np.load(root/'scifact-query-dense/vectors.npy')
    query_vectors={qid:queries[i] for i,qid in enumerate(qm['ids'])}
    lexical=FastSearch(root/'scifact-adapted',root/'scifact-results/fast.sqlite');rows=[]
    for _,q in records(root/'scifact-adapted/queries.jsonl'):
        start=time.perf_counter();scores=docs@query_vectors[q['id']]
        dense=[dm['ids'][i] for i in sorted(range(len(scores)),key=lambda i:(-float(scores[i]),dm['ids'][i]))[:100]]
        lexical_ids=[r['id'] for r in lexical.search(q['query'],100)]
        elapsed=time.perf_counter()-start
        progressive_start=time.perf_counter()
        progressive=progressive_search(docs,query_vectors[q['id']],dm['ids'],10,prefix=256,depth=100)
        progressive_seconds=time.perf_counter()-progressive_start
        for method,ranking in [('dense',dense[:10]),('rrf',rrf([dense,lexical_ids])),('prefix256-refine100',progressive)]:
            rows.append(dict(query_id=q['id'],method=method,ranking=ranking,metrics=metrics(ranking,q['qrels']),combined_local_seconds=elapsed,progressive_local_seconds=progressive_seconds if method=='prefix256-refine100' else None))
    (out/'runs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    summary={method:{key:sum(r['metrics'][key] for r in rows if r['method']==method)/len(query_vectors) for key in rows[0]['metrics']} for method in ['dense','rrf','prefix256-refine100']}
    (out/'summary.json').write_text(json.dumps(dict(results=summary,model=dm['model'],rrf_constant=60,depth=100,
        latency_warning='Query embeddings precomputed. No online latency or end-to-end speedup claim.'),indent=2))
    return summary
