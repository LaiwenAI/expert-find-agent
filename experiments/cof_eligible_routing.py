"""Candidate-ID-aware full scan and adaptive ANN, without relevance values."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import faiss
import numpy as np
from expert_search.cof_benchmark import load,precision_metrics
from expert_search.source_routing import expand_sources


def run(dataset):
    faiss.omp_set_num_threads(1);root,_,reviewers,labels=load(dataset)
    pids=sorted({p for r in reviewers.values() for p in r['papers']});qids=sorted(labels['soft']);source=root/'qwen-embedding-torch-v1'/'vectors.npy'
    matrix=np.load(source,allow_pickle=False);x=np.ascontiguousarray(matrix[:len(pids)]);position={p:i for i,p in enumerate(pids)}
    if matrix.shape!=(len(pids)+len(qids),1024):raise ValueError('Invalid vectors')
    owners={e:np.array([position[p] for p in sorted(set(r['papers']))],dtype=np.int64) for e,r in reviewers.items()};reverse=[[] for _ in pids]
    for e,ids in owners.items():
        for i in ids:reverse[i].append(e)
    start=time.perf_counter();index=faiss.IndexHNSWFlat(1024,32,faiss.METRIC_INNER_PRODUCT);index.hnsw.efConstruction=128;index.add(x);build=time.perf_counter()-start
    def score(q,candidates,selected):
        values=np.einsum('ij,j->i',x if len(selected)==len(x) else x[selected],q,optimize=False);lookup=np.empty(len(x),dtype=np.float32);lookup[selected]=values
        scores={e:float(np.max(lookup[owners[e]])) if len(owners[e]) else 0. for e in candidates}
        return sorted(scores,key=lambda e:(-scores[e],e))
    rows=[]
    for qi,qid in enumerate(qids):
        prepare_start=time.perf_counter();eligible=set(labels['soft'][qid])
        if eligible!=set(labels['hard'][qid]):raise ValueError('Different evaluation pools')
        q=matrix[len(pids)+qi];eligible_sources=np.array(sorted({int(i) for e in eligible for i in owners[e]}),dtype=np.int64)
        preparation=time.perf_counter()-prepare_start
        modes=('global_full','eligible_full','adaptive','selectivity_switch');results={};timings={m:[] for m in modes}
        for repeat in range(3):
            for mode in (modes if repeat%2==0 else modes[::-1]):
                start=time.perf_counter();route=None
                if mode=='global_full':candidates=sorted(eligible);selected=np.arange(len(pids));branch='full'
                elif mode=='eligible_full' or (mode=='selectivity_switch' and len(eligible_sources)<=1024):
                    candidates=sorted(eligible);selected=np.array(sorted({int(i) for e in eligible for i in owners[e]}),dtype=np.int64);branch='eligible_full'
                else:
                    route=expand_sources(index,q,reverse,eligible,target=10,initial=32,limit=1024);candidates=route['candidates'];branch='ann'
                    if not route['target_met']:candidates=sorted(eligible);branch='fallback_full'
                    selected=np.array(sorted({int(i) for e in candidates for i in owners[e]}),dtype=np.int64)
                ranking=score(q,candidates,selected);timings[mode].append(time.perf_counter()-start)
                results[mode]=dict(ranking=ranking,candidates=len(candidates),sources=len(selected),branch=branch,depths=route['depths'] if route else [],metrics={t:precision_metrics(ranking,truth[qid]) for t,truth in labels.items()})
        for mode,result in results.items():
            result['seconds']=float(np.median(timings[mode]));result['ordered_top5_equal']=result['ranking'][:5]==results['global_full']['ranking'][:5]
        rows.append(dict(query_id=qid,eligibility_preparation_seconds=preparation,eligible_experts=len(eligible),eligible_sources=len(eligible_sources),results=results));print(dataset,qid,flush=True)
    aggregates={m:dict(mean_seconds=float(np.mean([r['results'][m]['seconds'] for r in rows])),ordered_top5_equal=sum(r['results'][m]['ordered_top5_equal'] for r in rows),mean_sources=float(np.mean([r['results'][m]['sources'] for r in rows])),fallbacks=sum(r['results'][m]['branch']=='fallback_full' for r in rows),metrics={t:{k:float(np.mean([r['results'][m]['metrics'][t][k] for r in rows])) for k in ('p@5','ndcg@10')} for t in labels}) for m in rows[0]['results']}
    report=dict(dataset=dataset,queries=len(rows),vectors_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),faiss_version=faiss.__version__,build_seconds=build,mean_shared_preparation_seconds=float(np.mean([r['eligibility_preparation_seconds'] for r in rows])),parameters=dict(M=32,efConstruction=128,initial_sources=32,max_sources=1024,target_experts=10,switch_sources=1024,repeats=3,threads=1,timing_order='alternating forward/reverse each repeat'),aggregates=aggregates,rows=rows,warning='Only benchmark candidate IDs supplied before retrieval, no relevance values. Candidate availability is an explicit protocol assumption, not general enterprise-search ground truth. Stable per-row FP32 summation shared by all modes; previous BLAS baselines may differ on ties. Shared candidate/source-ID preparation reported separately; per-path times exclude it. Query cache excludes encoding, planning and evidence review. Heuristic switch and expansion are mature engineering, not a new optimality guarantee; no test-label parameter selection.')
    out=Path('artifacts')/('cof-eligible-routing-'+dataset.lower()+'-v1');out.mkdir(exist_ok=True);(out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',default='SIGIR',choices=['SIGIR','NIPS']);run(parser.parse_args().dataset)
