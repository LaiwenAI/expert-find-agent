"""Independent-label BEIR-format lexical benchmark, with resumable per-query output."""
import json
import math
import statistics
import time
from pathlib import Path
from .benchmark import metrics
from .corpus import records
from .fast_retrieval import FastSearch,compile_index
from .retrieval import index


def public_metrics(ranking,qrels,k=10):
    """TREC/BEIR linear graded gain; internal proxy metrics use exponential gain."""
    result=metrics(ranking,qrels,k)
    def dcg(values):return sum(g/math.log2(i+2) for i,g in enumerate(values))
    ideal=dcg(sorted(qrels.values(),reverse=True)[:k])
    result['ndcg@10']=dcg([qrels.get(x,0) for x in ranking[:k]])/ideal if ideal else 0.
    return result


def run(corpus='artifacts/public/scifact-adapted', output='artifacts/public/scifact-results',dataset_name='SciFact'):
    corpus=Path(corpus);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    db=out/'index.sqlite';fast=out/'fast.sqlite'
    if not db.exists():index(corpus,db)
    if not fast.exists():compile_index(db,fast)
    engine=FastSearch(corpus,fast);rows=[]
    for _,q in records(corpus/'queries.jsonl'):
        for method in ['bm25-max','tfidf-max']:
            target=out/f"{q['id']}-{method}.json"
            if target.exists():row=json.loads(target.read_text())
            else:
                start=time.perf_counter();result=engine.search(q['query'],10,method)
                ids=[r['id'] for r in result]
                row=dict(query_id=q['id'],query_type=q.get('query_type'),method=method,ranking=ids,seconds=time.perf_counter()-start)
            # Re-score cached rankings too: changing metric conventions must not
            # reuse stale values, and never requires repeating timed retrieval.
            row['metrics']=public_metrics(row['ranking'],q['qrels'])
            row['query_type']=q.get('query_type');row['ndcg_gain']='linear'
            target.write_text(json.dumps(row,indent=2))
            rows.append(row)
    summary={}
    for method in ['bm25-max','tfidf-max']:
        selected=[r for r in rows if r['method']==method];times=sorted(r['seconds'] for r in selected)
        summary[method]=dict(queries=len(selected),metrics={k:statistics.mean(r['metrics'][k] for r in selected) for k in selected[0]['metrics']},
            p50_seconds=statistics.median(times),p95_seconds=times[max(0,int(.95*len(times))-1)])
    by_type={}
    for query_type in sorted({r.get('query_type') for r in rows if r.get('query_type')}):
        by_type[query_type]={}
        for method in ['bm25-max','tfidf-max']:
            selected=[r for r in rows if r['method']==method and r.get('query_type')==query_type]
            by_type[query_type][method]=dict(queries=len(selected),metrics={key:statistics.mean(r['metrics'][key] for r in selected) for key in selected[0]['metrics']})
    (out/'summary.json').write_text(json.dumps(dict(task=dataset_name+' document retrieval; not expert matching',ndcg_gain='linear (TREC/BEIR)',results=summary,by_query_type=by_type),indent=2))
    return summary
