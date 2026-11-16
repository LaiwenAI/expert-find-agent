"""Frozen Chinese research-area index with deduplicated text and exact provenance."""
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
from fastembed import TextEmbedding
from .corpus import records
from .progressive_dense import top_indices


def build(output='artifacts/chinese-concepts'):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);concepts=defaultdict(list)
    for _,r in records('artifacts/corpus/evidence.jsonl'):
        if r.get('field')!='研究领域_CN':continue
        for phrase in r['text'].splitlines():
            phrase=phrase.strip()
            if 3<=len(phrase)<=40:
                concepts[phrase].append(dict(expert_id=r['expert_id'],evidence_id=r['id'],source=r['source']))
    terms=sorted(concepts,key=lambda t:(len(t),t))
    fingerprint=hashlib.sha256(json.dumps([terms,dict(concepts)],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    marker=out/'input.sha256'
    if marker.exists() and marker.read_text()!=fingerprint:raise ValueError('Concept inputs changed')
    marker.write_text(fingerprint)
    model=TextEmbedding(model_name='BAAI/bge-small-zh-v1.5',cache_dir='artifacts/models',threads=1)
    start=time.perf_counter()
    for i in range(0,len(terms),128):
        p=out/f'batch-{i}.npy'
        if p.exists():continue
        x=np.asarray(list(model.passage_embed(terms[i:i+128],batch_size=32)),dtype=np.float32)
        x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-30);np.save(p,x)
    matrix=np.concatenate([np.load(out/f'batch-{i}.npy') for i in range(0,len(terms),128)])
    np.save(out/'vectors.npy',matrix)
    metadata=dict(model='BAAI/bge-small-zh-v1.5',terms=terms,postings=dict(concepts),input_sha256=fingerprint,
        build_call_seconds=time.perf_counter()-start,warning='Research-area phrases only; does not index complete papers or establish actual capabilities.')
    (out/'manifest.json').write_text(json.dumps(metadata,ensure_ascii=False));return dict(concepts=len(terms),seconds=metadata['build_call_seconds'])


class ConceptSearch:
    def __init__(self,path='artifacts/chinese-concepts'):
        p=Path(path);self.meta=json.loads((p/'manifest.json').read_text());self.x=np.load(p/'vectors.npy')
        self.encoder=TextEmbedding(model_name=self.meta['model'],cache_dir='artifacts/models',threads=1)

    def search(self,query,k=10,depth=100):
        start=time.perf_counter();q=np.asarray(list(self.encoder.query_embed(query))[0],dtype=np.float32);q/=np.linalg.norm(q)
        scores=self.x@q;terms=self.meta['terms'];top=top_indices(scores,terms,depth);experts=defaultdict(list)
        for i in top:
            for source in self.meta['postings'][terms[i]]:experts[source['expert_id']].append(dict(concept=terms[i],score=float(scores[i]),**source))
        ranked=[]
        for eid,hits in experts.items():
            hits.sort(key=lambda r:(-r['score'],r['concept']))
            ranked.append(dict(id=eid,score=hits[0]['score'],evidence=hits[:3]))
        ranked.sort(key=lambda r:(-r['score'],r['id']))
        return dict(results=ranked[:k],seconds=time.perf_counter()-start,concept_depth=depth,
                    warning='Semantic concept similarity, not verified constraint entailment')

if __name__=='__main__':print(build())
