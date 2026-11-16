"""Query-to-reviewer baselines with original judged candidate sets and no training."""
import hashlib
import json
import statistics
import time
from pathlib import Path
from urllib.parse import quote
import numpy as np
from .cof_data import prepare,unique_records
from .corpus import records
from .dense_retrieval import encode
from .fast_retrieval import FastSearch,compile_index
from .public_benchmark import public_metrics
from .retrieval import index


def load(dataset):
    prepare(dataset);root=Path('artifacts/public/cof')/dataset
    papers=unique_records([r for _,r in records(root/f'{dataset}_papers_test.json')],'paper')
    reviewers=unique_records([r for _,r in records(root/f'{dataset}_reviewers_test.json')],'reviewer')
    labels={task:{r['query_id']:r['score'] for _,r in records(root/f'{dataset}_queries_test_{task}.json')} for task in ('soft','hard')}
    return root,papers,reviewers,labels


def encode_papers(dataset='SIGIR'):
    from laiwenai import LaiwenAI
    root,papers,_,_=load(dataset)
    texts=[dict(id=pid,text=(p.get('title','')+'\n'+p.get('abstract','')).strip()) for pid,p in sorted(papers.items())]
    encode(texts,root/'api-small',LaiwenAI(embedding_model='text-embedding-3-small'),batch_size=64)
    return dict(dataset=dataset,papers=len(texts),encoding='Frozen API embeddings; includes cached query vectors, no online latency claim')


def precision_metrics(ranking,labels):
    if set(ranking)-set(labels):raise ValueError('Ranking contains unjudged candidates')
    result=public_metrics(ranking,labels)
    for k in (5,10):result[f'p@{k}']=sum(labels.get(eid,0)>0 for eid in ranking[:k])/k
    return result


def rank_judged(scores,labels):
    if set(labels)-set(scores):raise ValueError('Missing judged candidate scores')
    return sorted(labels,key=lambda eid:(-scores[eid],eid))


def run(dataset='SIGIR',use_dense=False):
    root,papers,reviewers,labels=load(dataset)
    out=root/('baseline-api' if use_dense else 'baseline-lexical');out.mkdir(exist_ok=True)
    profile_ids=sorted({pid for r in reviewers.values() for pid in r['papers']})
    corpus=root/'unique-profile-corpus';corpus.mkdir(exist_ok=True)
    spec=hashlib.sha256(json.dumps([[(pid,papers[pid]) for pid in profile_ids],reviewers,labels],sort_keys=True).encode()).hexdigest()
    marker=out/'input.sha256'
    if marker.exists() and marker.read_text()!=spec:raise ValueError('Changed source or labels')
    marker.write_text(spec)
    (corpus/'experts.jsonl').write_text(''.join(json.dumps(dict(id=pid))+'\n' for pid in profile_ids))
    (corpus/'evidence.jsonl').write_text(''.join(json.dumps(dict(id=pid,expert_id=pid,kind='paper',
        text=(papers[pid].get('title','')+'\n'+papers[pid].get('abstract','')).strip(),source=str(root/f'{dataset}_papers_test.json')))+'\n' for pid in profile_ids))
    db=corpus/'index.sqlite';fast=corpus/'fast.sqlite'
    if not db.exists():index(corpus,db)
    if not fast.exists():compile_index(db,fast)
    lexical=FastSearch(corpus,fast)
    if use_dense:
        meta=json.loads((root/'api-small/manifest.json').read_text());vectors=np.load(root/'api-small/vectors.npy')
        position={pid:i for i,pid in enumerate(meta['ids'])}
        means={eid:np.mean(vectors[[position[pid] for pid in sorted(set(r['papers']))]],axis=0)
               for eid,r in reviewers.items() if r['papers']}
        means={eid:v/max(np.linalg.norm(v),1e-30) for eid,v in means.items()}
    rows=[]
    for qid in sorted(labels['soft']):
        path=out/(quote(qid,safe='')+'.json')
        if path.exists():rows.append(json.loads(path.read_text()));continue
        query=(papers[qid].get('title','')+'\n'+papers[qid].get('abstract','')).strip()
        start=time.perf_counter();scores=lexical.document_scores(query)
        per_reviewer={}
        for eid,r in reviewers.items():
            values=sorted((scores.get(pid,0.) for pid in set(r['papers'])),reverse=True)
            per_reviewer[eid]={'bm25-max':values[0] if values else 0.,'bm25-top3':sum(values[:3])/3}
        lexical_seconds=time.perf_counter()-start
        dense_seconds=None
        if use_dense:
            start=time.perf_counter();q=vectors[position[qid]];similarities=vectors@q
            for eid,r in reviewers.items():
                values=sorted((float(similarities[position[pid]]) for pid in set(r['papers'])),reverse=True)
                maximum=values[0] if values else 0.;mean=float(means[eid]@q) if eid in means else 0.
                per_reviewer[eid].update(dense_max=maximum,dense_top3=sum(values[:3])/3,
                                        dense_mean=mean,fixed_mix=.5*maximum+.5*mean)
            dense_seconds=time.perf_counter()-start
        results={}
        for task in ('soft','hard'):
            truth=labels[task][qid]
            results[task]={}
            for method in next(iter(per_reviewer.values())):
                ranking=rank_judged({eid:values[method] for eid,values in per_reviewer.items()},truth)
                results[task][method]=dict(ranking=ranking[:10],metrics=precision_metrics(ranking,truth))
        row=dict(query_id=qid,results=results,lexical_seconds=lexical_seconds,dense_scoring_seconds=dense_seconds)
        path.write_text(json.dumps(row,indent=2));rows.append(row)
    report=dict(dataset=dataset,queries=len(rows),results={task:{method:{key:statistics.mean(r['results'][task][method]['metrics'][key] for r in rows)
        for key in rows[0]['results'][task][method]['metrics']} for method in rows[0]['results'][task]} for task in ('soft','hard')},
        median_lexical_seconds=statistics.median(r['lexical_seconds'] for r in rows),
        median_dense_scoring_seconds=statistics.median(r['dense_scoring_seconds'] for r in rows) if use_dense else None,
        warning='Original soft/hard labels; rank only explicitly judged candidates. No test-label parameter tuning; fixed 0.5 mixture carried from prior CMU exploration. Cached query vectors exclude online encoding; dense timing combines all dense variants, not a standalone method. Profiles use unique paper links; top3 sum divides by 3. Original source data retained; check manifest for text duplicates before leakage-controlled claims.')
    (out/'summary.json').write_text(json.dumps(report,indent=2));return report


if __name__=='__main__':print(json.dumps(run(),indent=2))
