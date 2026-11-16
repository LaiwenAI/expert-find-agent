"""Exploratory fixed conjunction slice; frozen encoders and no label tuning."""
import hashlib
import json
import statistics
import time
from pathlib import Path
import numpy as np
from fastembed import TextEmbedding
from .corpus import records
from .fast_retrieval import FastSearch
from .hybrid_benchmark import rrf
from .progressive_dense import top_indices
from .public_benchmark import public_metrics


def run(output='artifacts/public/comlq/conjunction-results'):
    root=Path('artifacts/public/comlq');plans=root/'conjunction-plans';out=Path(output);out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((plans/'protocol.json').read_text());ids=protocol['query_ids']
    queries={q['id']:q for _,q in records(root/'adapted/queries.jsonl')}
    plan_rows={qid:json.loads((plans/(qid+'.json')).read_text()) for qid in ids}
    documents=[r for _,r in records(root/'adapted/evidence.jsonl')];docids=[r['id'] for r in documents]
    matrix=np.load(root/'local-bge/vectors.npy')
    if matrix.shape[0]!=len(docids):raise ValueError('Vector/document count mismatch')
    fingerprint=hashlib.sha256(json.dumps(dict(protocol=protocol,plans=plan_rows,
        vector_sha256=hashlib.sha256((root/'local-bge/vectors.npy').read_bytes()).hexdigest(),
        lexical_depth=100,rrf_constant=60,version='conjunction-v1'),sort_keys=True).encode()).hexdigest()
    marker=out/'input.sha256'
    if marker.exists() and marker.read_text()!=fingerprint:raise ValueError('Conjunction benchmark inputs changed')
    marker.write_text(fingerprint)
    encoder=TextEmbedding(model_name='BAAI/bge-small-en-v1.5',cache_dir='artifacts/models',threads=1)
    sparse=FastSearch(root/'adapted',root/'results/fast.sqlite');rows=[]
    for qid in ids:
        path=out/(qid+'.json')
        if path.exists():rows.append(json.loads(path.read_text()));continue
        query=queries[qid];plan=plan_rows[qid]
        if plan['query']!=query['query']:raise ValueError('Planner query changed')
        facets=plan.get('plan',{}).get('facets',[]) if plan['status']=='ok' else []
        use_facets=bool(plan.get('plan',{}).get('decomposable')) and 2<=len(facets)<=4
        start=time.perf_counter()
        texts=[query['query']]+(facets if use_facets else [])
        vectors=np.asarray(list(encoder.query_embed(texts)),dtype=np.float32)
        vectors/=np.maximum(np.linalg.norm(vectors,axis=1,keepdims=True),1e-30)
        scores=matrix@vectors.T
        dense=[docids[i] for i in top_indices(scores[:,0],docids,100)]
        lexical_rows=sparse.search(query['query'],100)
        lexical=[r['id'] for r in lexical_rows]
        rankings={'dense-joint':dense,'lexical-joint':lexical,'hybrid-joint':rrf([dense,lexical])}
        if use_facets:
            # Fixed aggregation functions, not calibrated semantic probabilities.
            positive=np.maximum(scores[:,1:],1e-8)
            aggregates={'mean':positive.mean(axis=1),'min':positive.min(axis=1),
                        'hmean':positive.shape[1]/(1/positive).sum(axis=1)}
            for name,values in aggregates.items():
                rankings['dense-facet-'+name]=[docids[i] for i in top_indices(values,docids,100)]
            per_facet=[]
            for facet in facets:
                hits=sparse.search(facet,100);maximum=max((r['score'] for r in hits),default=1.)
                per_facet.append({r['id']:r['score']/max(maximum,1e-30) for r in hits})
            candidates=set().union(*(set(x) for x in per_facet))
            conjunctive={eid:min(f.get(eid,0.) for f in per_facet) for eid in candidates}
            rankings['lexical-facet-min']=sorted((eid for eid in candidates if conjunctive[eid]>0),key=lambda eid:(-conjunctive[eid],eid))[:100]
            rankings['fixed-facet-fusion']=rrf([lexical,rankings['lexical-facet-min'],rankings['dense-facet-hmean']])
        else:
            for name in ('mean','min','hmean'):rankings['dense-facet-'+name]=dense
            rankings['lexical-facet-min']=lexical
            rankings['fixed-facet-fusion']=rankings['hybrid-joint']
        row=dict(query_id=qid,query_type=query.get('query_type'),used_facets=use_facets,facets=facets,
            metrics={method:public_metrics(ranking,query['qrels']) for method,ranking in rankings.items()},
            rankings={method:ranking[:10] for method,ranking in rankings.items()},
            combined_experiment_seconds=time.perf_counter()-start,planner_seconds=plan['seconds'])
        path.write_text(json.dumps(row,indent=2));rows.append(row)
    summary=dict(queries=len(rows),planned=sum(r['used_facets'] for r in rows),
        metrics={method:{key:statistics.mean(r['metrics'][method][key] for r in rows)
                        for key in rows[0]['metrics'][method]} for method in rows[0]['metrics']},
        mean_planner_seconds=statistics.mean(r['planner_seconds'] for r in rows),
        warning='Exploratory fixed 100-query pure-conjunction slice, not all ComLQ or expert relevance. No labels in planning or parameter tuning; semantic decomposition not independently audited. Failed/rejected plans fall back and remain in denominator. Combined experiment timing runs all variants together, so is not any method latency; planner calls were separate and concurrent. Existing decomposition/fusion components, not a novel-method claim.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2));return summary


if __name__=='__main__':print(json.dumps(run(),indent=2))
