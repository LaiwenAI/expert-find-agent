"""Frozen Qwen3 embedding baselines with resumable, hashed materialization."""
import hashlib
import inspect
import json
import time
from pathlib import Path
from urllib.parse import quote
import numpy as np
from expert_search.cof_benchmark import load,rank_judged,precision_metrics
from expert_search.qwen_embedding import QwenEmbedding,TASK


def run(dataset='SIGIR',backend='mlx'):
    encoder=QwenEmbedding
    if backend=='torch':
        from expert_search.qwen_embedding_torch import QwenTorchEmbedding
        encoder=QwenTorchEmbedding
    root,papers,reviewers,labels=load(dataset);out=root/('qwen-embedding-torch-v1' if backend=='torch' else 'qwen-embedding-v1');out.mkdir(exist_ok=True)
    profile_ids=sorted({pid for r in reviewers.values() for pid in r['papers']})
    qids=sorted(labels['soft']);items=[dict(id='doc:'+pid,text=(papers[pid].get('title','')+'\n'+papers[pid].get('abstract','')).strip(),query=False) for pid in profile_ids]
    items += [dict(id='query:'+qid,text=(papers[qid].get('title','')+'\n'+papers[qid].get('abstract','')).strip(),query=True) for qid in qids]
    checkpoint=Path('artifacts/models/qwen3-embedding-0.6b/checkpoint-provenance.json')
    protocol=dict(version=1,dataset=dataset,encoder_sha256=hashlib.sha256(inspect.getsource(encoder).encode()).hexdigest(),input_sha256=hashlib.sha256(json.dumps(items,sort_keys=True).encode()).hexdigest(),
        checkpoint=json.loads(checkpoint.read_text()),max_length=512,batch_size=8,task=TASK,chunk_size=32,
        scoring='Unique owned paper cosine max and normalized mean, fixed equal mean/max mixture. Full 1024 dims, BF16 backbone and FP32 normalized pooling. No training.')
    marker=out/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=protocol:raise ValueError('Changed embedding protocol')
    marker.write_text(json.dumps(protocol,indent=2));model=encoder(max_length=512,batch_size=8);vectors=[];encoding_seconds=0.
    for start in range(0,len(items),32):
        group=items[start:start+32];target=out/f'batch-{start}.npz'
        fingerprint=hashlib.sha256(json.dumps(group,sort_keys=True).encode()).hexdigest()
        if target.exists():
            with np.load(target,allow_pickle=False) as data:
                if data['fingerprint'].item()!=fingerprint:raise ValueError('Changed embedding chunk')
                array=data['vectors'];elapsed=float(data['seconds'])
        else:
            begin=time.perf_counter();array=np.empty((len(group),1024),dtype=np.float32)
            for mode in (False,True):
                positions=[i for i,item in enumerate(group) if item['query']==mode]
                if positions:array[positions]=model.encode([group[i]['text'] for i in positions],query=mode)
            elapsed=time.perf_counter()-begin;temporary=out/f'batch-{start}.partial.npz'
            np.savez(temporary,vectors=array,fingerprint=fingerprint,seconds=elapsed);temporary.replace(target)
        if array.shape!=(len(group),1024) or not np.isfinite(array).all() or not np.allclose(np.linalg.norm(array,axis=1),1,atol=1e-5):raise ValueError('Invalid cached vectors')
        vectors.append(array);encoding_seconds+=elapsed;print('encoded',start+len(group),'/',len(items),flush=True)
    matrix=np.concatenate(vectors);np.save(out/'vectors.npy',matrix)
    position={item['id']:i for i,item in enumerate(items)}
    means={eid:matrix[[position['doc:'+pid] for pid in sorted(set(r['papers']))]].mean(axis=0) for eid,r in reviewers.items() if r['papers']}
    means={eid:v/max(float(np.linalg.norm(v)),1e-30) for eid,v in means.items()};rows=[]
    for qid in qids:
        start=time.perf_counter();q=matrix[position['query:'+qid]];similarities=matrix[:len(profile_ids)]@q;scores={}
        for eid,r in reviewers.items():
            maximum=max((float(similarities[position['doc:'+pid]]) for pid in set(r['papers'])),default=0.)
            mean=float(means[eid]@q) if eid in means else 0.
            scores[eid]={'max':maximum,'mean':mean,'fixed_mix':.5*(maximum+mean)}
        results={task:{method:dict(ranking=(ranking:=rank_judged({eid:s[method] for eid,s in scores.items()},truth[qid])),metrics=precision_metrics(ranking,truth[qid])) for method in ('max','mean','fixed_mix')} for task,truth in labels.items()}
        row=dict(query_id=qid,scoring_seconds=time.perf_counter()-start,results=results);rows.append(row)
        (out/(quote(qid,safe='')+'.json')).write_text(json.dumps(row,indent=2))
    report=dict(dataset=dataset,queries=len(qids),papers=len(profile_ids),encoding_seconds=encoding_seconds,
        vectors_sha256=hashlib.sha256((out/'vectors.npy').read_bytes()).hexdigest(),
        metrics={task:{method:{metric:sum(r['results'][task][method]['metrics'][metric] for r in rows)/len(rows) for metric in ('p@5','ndcg@10')} for method in ('max','mean','fixed_mix')} for task in labels},
        mean_cached_scoring_seconds=sum(r['scoring_seconds'] for r in rows)/len(rows),
        warning='Cached queries exclude online embedding latency. Single scoring repeat. Full released candidate sets and original labels; duplicate sensitivity not yet recomputed. Frozen model baseline, no own method-quality claim.')
    (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',default='SIGIR');parser.add_argument('--backend',choices=['mlx','torch'],default='mlx');args=parser.parse_args();run(args.dataset,args.backend)
