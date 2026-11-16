"""Fixed-budget HNSW source routing plus complete candidate-source refinement."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import faiss
import numpy as np
from expert_search.cof_benchmark import load,rank_judged,precision_metrics


def run(dataset):
    faiss.omp_set_num_threads(1)
    root,_,reviewers,labels=load(dataset);source=root/'qwen-embedding-torch-v1'/'vectors.npy'
    matrix=np.load(source,allow_pickle=False);pids=sorted({p for r in reviewers.values() for p in r['papers']});qids=sorted(labels['soft'])
    if matrix.shape!=(len(pids)+len(qids),1024):raise ValueError('Changed vector inventory')
    x=np.ascontiguousarray(matrix[:len(pids)],dtype=np.float32);position={p:i for i,p in enumerate(pids)}
    owners={e:np.array([position[p] for p in sorted(set(r['papers']))],dtype=np.int64) for e,r in reviewers.items()}
    reverse=[[] for _ in pids]
    for e,ids in owners.items():
        for i in ids:reverse[i].append(e)
    flat=faiss.IndexFlatIP(1024);flat.add(x)
    start=time.perf_counter();hnsw=faiss.IndexHNSWFlat(1024,32,faiss.METRIC_INNER_PRODUCT);hnsw.hnsw.efConstruction=128;hnsw.add(x);build=time.perf_counter()-start
    rows=[];full_times=[];full_metrics=[]
    for qi,qid in enumerate(qids):
        q=np.ascontiguousarray(matrix[len(pids)+qi:len(pids)+qi+1]);reference_times=[]
        for repeat in range(3):
            start=time.perf_counter();all_scores=x@q[0]
            full={e:float(np.max(all_scores[ids])) if len(ids) else 0. for e,ids in owners.items()};full_order=sorted(full,key=lambda e:(-full[e],e));reference_times.append(time.perf_counter()-start)
        full_times.append(float(np.median(reference_times)))
        full_metrics.append({t:precision_metrics(rank_judged(full,truth[qid]),truth[qid]) for t,truth in labels.items()})
        exact_source_top10=set(map(int,flat.search(q,10)[1][0]))
        for depth in (32,128,512):
            for mode,index in (('flat',flat),('hnsw',hnsw)):
                if mode=='hnsw':index.hnsw.efSearch=max(64,depth)
                times=[]
                for repeat in range(3):
                    start=time.perf_counter();_,indices=index.search(q,min(depth,len(pids)))
                    candidates=sorted({e for i in indices[0] if i>=0 for e in reverse[i]})
                    selected=np.array(sorted({int(i) for e in candidates for i in owners[e]}),dtype=np.int64)
                    refined=x[selected]@q[0];lookup={int(i):float(s) for i,s in zip(selected,refined)}
                    scores={e:max(lookup[int(i)] for i in owners[e]) for e in candidates}
                    ranking=sorted(scores,key=lambda e:(-scores[e],e));times.append(time.perf_counter()-start)
                rows.append(dict(query_id=qid,mode=mode,depth=depth,seconds=float(np.median(times)),candidate_experts=len(candidates),refined_unique_sources=len(selected),
                    full_source_top10_retention=len(exact_source_top10 & set(map(int,indices[0])))/len(exact_source_top10),full_top5_retention=len(set(full_order[:5])&set(candidates))/5,ordered_top5_equal=ranking[:5]==full_order[:5],
                    metrics={t:precision_metrics([e for e in ranking if e in truth[qid]],truth[qid]) for t,truth in labels.items()}))
        print(dataset,qid,flush=True)
    aggregates=[]
    for mode in ('flat','hnsw'):
        for depth in (32,128,512):
            group=[r for r in rows if r['mode']==mode and r['depth']==depth]
            aggregates.append(dict(mode=mode,depth=depth,mean_seconds=float(np.mean([r['seconds'] for r in group])),mean_candidates=float(np.mean([r['candidate_experts'] for r in group])),mean_refined_sources=float(np.mean([r['refined_unique_sources'] for r in group])),mean_full_source_top10_retention=float(np.mean([r['full_source_top10_retention'] for r in group])),mean_full_top5_retention=float(np.mean([r['full_top5_retention'] for r in group])),ordered_top5_equal=sum(r['ordered_top5_equal'] for r in group),metrics={t:{m:float(np.mean([r['metrics'][t][m] for r in group])) for m in ('p@5','ndcg@10')} for t in labels}))
    report=dict(dataset=dataset,queries=len(qids),faiss_version=faiss.__version__,vectors_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),full_metrics={t:{m:float(np.mean([r[t][m] for r in full_metrics])) for m in ('p@5','ndcg@10')} for t in labels},build_seconds=build,mean_full_seconds=float(np.mean(full_times)),index_bytes=len(faiss.serialize_index(hnsw)),M=32,efConstruction=128,efSearch='max(64,source_depth)',threads=1,repeats=3,aggregates=aggregates,rows=rows,warning='Mature external baseline, not own method. Source routing approximates expert max; all sources of recovered experts are refined. Full-top5 retention is fidelity to frozen similarity, not relevance recall. Cached query scoring only, no encoder or evidence verification. Fixed budgets, no test-label selection; floating ties may change order. Public metrics condense to judged retrieved experts; unreturned experts are omitted rather than appended with arbitrary tied scores.')
    out=Path('artifacts')/('cof-hnsw-'+dataset.lower()+'-v1');out.mkdir(exist_ok=True);(out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',default='SIGIR',choices=['SIGIR','NIPS']);run(parser.parse_args().dataset)
