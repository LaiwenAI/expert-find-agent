"""Frozen embedding baseline with checkpointed batches; no model training."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .corpus import records


def encode(texts,output,client,batch_size=32):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    spec=[{'id':r['id'],'text':r['text']} for r in texts]
    if not spec or batch_size<1 or len({r['id'] for r in spec})!=len(spec):raise ValueError('Require nonempty unique IDs and positive batch size')
    digest=hashlib.sha256(json.dumps(spec,ensure_ascii=False).encode()).hexdigest()
    marker=out/'input.sha256'
    if marker.exists() and marker.read_text()!=digest:raise ValueError('Changed embedding corpus; use new output')
    marker.write_text(digest);models=set()
    for i in range(0,len(spec),batch_size):
        path=out/f'batch-{i}.json'
        if path.exists():
            saved=json.loads(path.read_text())
            if saved['ids']!=[r['id'] for r in spec[i:i+batch_size]]:raise ValueError('Cached batch boundaries changed')
            models.add(saved['model']);continue
        result=client.embeddings.create(input=[r['text'] for r in spec[i:i+batch_size]])
        data=sorted(result.data,key=lambda x:x.index)
        if [x.index for x in data]!=list(range(len(spec[i:i+batch_size]))):raise ValueError('Embedding indices mismatch')
        row=dict(model=result.model,ids=[r['id'] for r in spec[i:i+batch_size]],vectors=[x.embedding for x in data])
        path.write_text(json.dumps(row));models.add(result.model)
    if len(models)!=1:raise ValueError('Mixed embedding models')
    vectors=[];ids=[]
    for i in range(0,len(spec),batch_size):
        row=json.loads((out/f'batch-{i}.json').read_text());ids.extend(row['ids']);vectors.extend(row['vectors'])
    matrix=np.asarray(vectors,dtype=np.float32)
    if not np.isfinite(matrix).all():raise ValueError('Invalid embedding values')
    norm=np.linalg.norm(matrix,axis=1,keepdims=True)
    if (norm==0).any():raise ValueError('Zero embeddings')
    np.save(out/'vectors.npy',matrix/norm)
    (out/'manifest.json').write_text(json.dumps(dict(ids=ids,model=next(iter(models)),input_sha256=digest,dimensions=matrix.shape[1]),indent=2))


class DenseSearch:
    def __init__(self,path,client):
        self.path=Path(path);self.meta=json.loads((self.path/'manifest.json').read_text())
        self.matrix=np.load(self.path/'vectors.npy');self.client=client

    def search(self,query,k=10):
        if k<1:raise ValueError('Invalid k')
        response=self.client.embeddings.create(input=query)
        if response.model!=self.meta['model']:raise ValueError('Query/index embedding model mismatch')
        q=np.asarray(response.data[0].embedding,dtype=np.float32)
        if not np.isfinite(q).all() or np.linalg.norm(q)==0:raise ValueError('Invalid query vector')
        scores=self.matrix@(q/np.linalg.norm(q))
        order=sorted(range(len(scores)),key=lambda i:(-float(scores[i]),self.meta['ids'][i]))[:k]
        return [{'id':self.meta['ids'][i],'score':float(scores[i])} for i in order]
