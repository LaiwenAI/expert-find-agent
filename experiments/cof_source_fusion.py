"""Fixed semantic/lexical fusion at expert versus owned-source levels."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from urllib.parse import quote
import numpy as np
from expert_search.array_bm25 import ArrayBM25
from expert_search.cof_benchmark import load,precision_metrics
from expert_search.statistics import paired_interval
from expert_search.source_fusion import reciprocal_scores,source_fusion


def run(dataset,lexical_snapshot=None):
    root,papers,reviewers,labels=load(dataset);pids=sorted({p for r in reviewers.values() for p in r['papers']});qids=sorted(labels['soft'])
    vp=root/'qwen-embedding-torch-v1/vectors.npy';vectors=np.load(vp,allow_pickle=False)
    if vectors.shape!=(len(pids)+len(qids),1024):raise ValueError('Changed vector inventory')
    position={p:i for i,p in enumerate(pids)}
    if lexical_snapshot is None:engine=ArrayBM25(root/'unique-profile-corpus/fast.sqlite');lexical_matrix=None
    else:
        with np.load(lexical_snapshot,allow_pickle=False) as data:
            if list(data['paper_ids'])!=pids or list(data['query_ids'])!=qids:raise ValueError('Changed lexical snapshot order')
            lexical_matrix=data['scores']
        if lexical_matrix.shape!=(len(qids),len(pids)) or not np.isfinite(lexical_matrix).all():raise ValueError('Invalid lexical scores')
    out=Path('artifacts')/('cof-source-fusion-'+dataset.lower()+'-v1');out.mkdir(exist_ok=True)
    protocol=dict(version=1,dataset=dataset,constant=60,weights=[1,1],methods=['qwen_max','bm25_top3','expert_rrf','source_max','source_top3'],
        vectors_sha256=hashlib.sha256(vp.read_bytes()).hexdigest(),
        input_sha256=hashlib.sha256(json.dumps([reviewers,labels,{q:papers[q] for q in qids},pids],sort_keys=True).encode()).hexdigest(),
        fusion_code_sha256=hashlib.sha256(Path('src/expert_search/source_fusion.py').read_bytes()).hexdigest(),
        scope='Same benchmark candidate IDs for all methods; no relevance values used in ranking. Source universe is unique papers of eligible experts. Equal reciprocal ranks; tied scores share competition rank, zero lexical hits contribute nothing. Frozen models, no own training.',
        analysis='Exploratory after inspecting prior baselines. All fixed variants reported, no test-label parameter selection. Paired query bootstrap intervals unadjusted for multiple comparisons; not independent confirmation.')
    if lexical_snapshot is not None:protocol['lexical_snapshot_sha256']=hashlib.sha256(Path(lexical_snapshot).read_bytes()).hexdigest()
    marker=out/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=protocol:raise ValueError('Changed fusion experiment inputs')
    marker.write_text(json.dumps(protocol,indent=2));rows=[]
    for qi,qid in enumerate(qids):
        eligible=set(labels['soft'][qid])
        if eligible!=set(labels['hard'][qid]):raise ValueError('Inconsistent candidate pools')
        profiles={e:reviewers[e]['papers'] for e in sorted(eligible)};ids=sorted({p for ps in profiles.values() for p in ps})
        query=(papers[qid].get('title','')+'\n'+papers[qid].get('abstract','')).strip();start=time.perf_counter()
        lexical=engine.document_scores(query) if lexical_matrix is None else {p:float(v) for p,v in zip(pids,lexical_matrix[qi]) if v>0};values=vectors[[position[p] for p in ids]]@vectors[len(pids)+qi];semantic=dict(zip(ids,map(float,values)))
        dense={e:max((semantic[p] for p in set(ps)),default=0.) for e,ps in profiles.items()}
        sparse={e:sum(sorted((lexical.get(p,0.) for p in set(ps)),reverse=True)[:3])/3 for e,ps in profiles.items()}
        a,b=reciprocal_scores(dense),reciprocal_scores(sparse,positive_only=True)
        fused=source_fusion(semantic,lexical,profiles)
        scores=dict(qwen_max=dense,bm25_top3=sparse,expert_rrf={e:a[e]+b.get(e,0.) for e in profiles},source_max={e:r['source_max'] for e,r in fused.items()},source_top3={e:r['source_top3'] for e,r in fused.items()})
        rankings={m:sorted(s,key=lambda e:(-s[e],e)) for m,s in scores.items()};seconds=time.perf_counter()-start
        metrics={m:{t:precision_metrics(rank,truth[qid]) for t,truth in labels.items()} for m,rank in rankings.items()}
        row=dict(query_id=qid,rankings=rankings,metrics=metrics,shared_all_methods_seconds=seconds,source_witnesses={e:fused[e]['witnesses'] for e in set(rankings['source_max'][:10]+rankings['source_top3'][:10])})
        rows.append(row);(out/(quote(qid,safe='')+'.json')).write_text(json.dumps(row,indent=2))
    methods=protocol['methods'];aggregates={m:{t:{k:float(np.mean([r['metrics'][m][t][k] for r in rows])) for k in ('p@5','ndcg@10')} for t in labels} for m in methods}
    contrasts={m:{baseline:{t:paired_interval([r['metrics'][m][t]['p@5']-r['metrics'][baseline][t]['p@5'] for r in rows]) for t in labels} for baseline in ('qwen_max','bm25_top3')} for m in ('expert_rrf','source_max','source_top3')}
    report=dict(protocol=protocol,queries=len(rows),metrics=aggregates,contrasts=contrasts,warning='Common candidate protocol and fixed frozen vectors. Source witnesses indicate retrieval contributions, not factual support. Cached shared-scoring timings do not measure independent online methods. No industrial-fit, significance-after-selection or SOTA claim.')
    (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(dataset=dataset,queries=len(rows),metrics=aggregates),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=['SIGIR','NIPS','KDD','SciRepEval']);parser.add_argument('--lexical-snapshot');args=parser.parse_args()
    if args.lexical_snapshot and not args.dataset:parser.error('Lexical snapshot requires a dataset')
    for dataset in ([args.dataset] if args.dataset else ['SIGIR','NIPS']):run(dataset,args.lexical_snapshot)
