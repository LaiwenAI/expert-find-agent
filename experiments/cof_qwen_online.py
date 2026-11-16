"""Online query encoding plus complete expert scoring, frozen document vectors."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from expert_search.cof_benchmark import load,rank_judged,precision_metrics
from expert_search.qwen_embedding import QwenEmbedding


def run():
    root,papers,reviewers,labels=load('SIGIR');parent=root/'qwen-embedding-v1';out=root/'qwen-embedding-online-v1';out.mkdir(exist_ok=True)
    spec=json.loads((parent/'protocol.json').read_text());profile_ids=sorted({pid for r in reviewers.values() for pid in r['papers']});qids=sorted(labels['soft'])
    matrix=np.load(parent/'vectors.npy',allow_pickle=False)
    if hashlib.sha256((parent/'vectors.npy').read_bytes()).hexdigest()!=json.loads((parent/'summary.json').read_text())['vectors_sha256']:raise ValueError('Changed vectors')
    protocol=dict(version=1,parent=spec,document_vectors_sha256=hashlib.sha256((parent/'vectors.npy').read_bytes()).hexdigest(),measurement='Single-query encode plus all expert scoring and both-task ranking; one repeat, loaded model; initialization separate. No query cache.')
    marker=out/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=protocol:raise ValueError('Changed online protocol')
    marker.write_text(json.dumps(protocol,indent=2));start=time.perf_counter();model=QwenEmbedding(max_length=512,batch_size=8);initialization=time.perf_counter()-start
    position={pid:i for i,pid in enumerate(profile_ids)};document_matrix=matrix[:len(profile_ids)]
    means={eid:document_matrix[[position[pid] for pid in sorted(set(r['papers']))]].mean(axis=0) for eid,r in reviewers.items() if r['papers']};means={eid:v/max(float(np.linalg.norm(v)),1e-30) for eid,v in means.items()}
    rows=[]
    for index,qid in enumerate(qids):
        target=out/(qid+'.json')
        if target.exists():row=json.loads(target.read_text())
        else:
            paper=papers[qid];query=(paper.get('title','')+'\n'+paper.get('abstract','')).strip()
            start=time.perf_counter();q=model.encode([query],query=True)[0];encoding=time.perf_counter()-start
            similarities=document_matrix@q;scores={}
            for eid,r in reviewers.items():
                maximum=max((float(similarities[position[pid]]) for pid in set(r['papers'])),default=0.);mean=float(means[eid]@q) if eid in means else 0.
                scores[eid]={'max':maximum,'mean':mean,'fixed_mix':.5*(maximum+mean)}
            rankings={method:rank_judged({eid:s[method] for eid,s in scores.items()},labels['soft'][qid]) for method in ('max','mean','fixed_mix')}
            seconds=time.perf_counter()-start
            cached=json.loads((parent/(qid+'.json')).read_text())
            row=dict(query_id=qid,seconds=seconds,encoding_seconds=encoding,
                maximum_query_vector_difference=float(np.max(np.abs(q-matrix[len(profile_ids)+index]))),
                results={method:dict(ranking=ranking,top5_equal_cached=ranking[:5]==cached['results']['soft'][method]['ranking'][:5],metrics={task:precision_metrics(ranking,truth[qid]) for task,truth in labels.items()}) for method,ranking in rankings.items()})
            temporary=target.with_suffix('.partial');temporary.write_text(json.dumps(row,indent=2));temporary.replace(target)
        rows.append(row);print(qid,round(row['seconds'],4),flush=True)
    report=dict(queries=len(rows),initialization_seconds=initialization,mean_seconds=sum(r['seconds'] for r in rows)/len(rows),median_seconds=float(np.median([r['seconds'] for r in rows])),p95_seconds=float(np.percentile([r['seconds'] for r in rows],95)),
        mean_encoding_seconds=sum(r['encoding_seconds'] for r in rows)/len(rows),
        top5_equal_cached={m:sum(r['results'][m]['top5_equal_cached'] for r in rows) for m in ('max','mean','fixed_mix')},
        metrics={method:{task:{metric:sum(r['results'][method]['metrics'][task][metric] for r in rows)/len(rows) for metric in ('p@5','ndcg@10')} for task in labels} for method in ('max','mean','fixed_mix')},
        warning='One repeat, loaded model, frozen document vectors. Ranking three methods together. No generation, constraints or evidence-verification stage included. Batch versus single-query numerical changes reported explicitly.')
    (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':run()
