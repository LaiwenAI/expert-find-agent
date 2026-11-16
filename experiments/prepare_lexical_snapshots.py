"""Freeze full BM25 document scores for remote semantic-fusion experiments."""
import hashlib
import json
from pathlib import Path
import numpy as np
from expert_search.array_bm25 import ArrayBM25
from expert_search.cof_benchmark import load


def run(dataset):
    root,papers,reviewers,labels=load(dataset);pids=sorted({p for r in reviewers.values() for p in r['papers']});qids=sorted(labels['soft'])
    database=root/'unique-profile-corpus/fast.sqlite';engine=ArrayBM25(database)
    texts=[(papers[q].get('title','')+'\n'+papers[q].get('abstract','')).strip() for q in qids]
    source_sha=hashlib.sha256(database.read_bytes()).hexdigest();out=Path('artifacts/lexical-snapshots-v1');out.mkdir(exist_ok=True)
    target=out/(dataset+'.npz');metadata=out/(dataset+'.json')
    spec=dict(dataset=dataset,database_sha256=source_sha,queries_sha256=hashlib.sha256(json.dumps(texts).encode()).hexdigest(),scoring='Unchanged ArrayBM25 document scores, float64; full original corpus statistics, unique query tokens. No labels used in scores.')
    if target.exists():
        old=json.loads(metadata.read_text())
        if any(old.get(k)!=v for k,v in spec.items()) or hashlib.sha256(target.read_bytes()).hexdigest()!=old['snapshot_sha256']:raise ValueError('Changed lexical snapshot')
    else:
        matrix=np.zeros((len(qids),len(pids)),dtype=np.float64);position={p:i for i,p in enumerate(pids)}
        for i,text in enumerate(texts):
            for pid,score in engine.document_scores(text).items():matrix[i,position[pid]]=score
        np.savez_compressed(target,scores=matrix,paper_ids=np.array(pids),query_ids=np.array(qids))
        metadata.write_text(json.dumps(dict(spec,snapshot_sha256=hashlib.sha256(target.read_bytes()).hexdigest()),indent=2))
    print(dataset,'snapshot bytes',target.stat().st_size,flush=True)


if __name__=='__main__':
    for dataset in ('KDD','SciRepEval'):run(dataset)
