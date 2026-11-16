"""Matched source-selection change for the frozen demand-first audit cohort."""
import hashlib
import json
from pathlib import Path
import numpy as np
from fastembed import TextEmbedding
from expert_search.corpus import records


def run():
    parent=Path('artifacts/demand-first-pool-v1/jobs.jsonl')
    spec=json.loads(Path('artifacts/demand-first-audit-v1/protocol.json').read_text())
    jobs=[j for _,j in records(parent) if j['id'] in spec['ids']]
    if len(jobs)!=len(spec['ids']):raise ValueError('Changed matched cohort')
    wanted={j['expert_id'] for j in jobs}
    documents={s['id']:s for _,s in records('artifacts/corpus/evidence.jsonl') if s['expert_id'] in wanted and s['kind']=='paper' and s['text'].strip()}
    ids=sorted(documents);texts=[documents[e]['text'][:1200] for e in ids]+[j['facet']['search_query'] for j in jobs]
    out=Path('artifacts/demand-first-semantic-pool-v1');out.mkdir(exist_ok=True)
    model=TextEmbedding(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',cache_dir='artifacts/models',threads=1,max_length=128)
    model_dir=model.model._model_dir
    protocol=dict(version=1,parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),ids=spec['ids'],
        input_sha256=hashlib.sha256(json.dumps(texts,ensure_ascii=False).encode()).hexdigest(),
        snapshot=model_dir.name,weights={str(p.relative_to(model_dir)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(model_dir.rglob('*')) if p.is_file()},
        policy='Same demand, condition and expert; rank all owned nonempty paper prefixes by frozen multilingual cosine; retain the same number of sources (up to three). Original full source text supplied to audit.')
    marker=out/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=protocol:raise ValueError('Changed semantic source protocol')
    marker.write_text(json.dumps(protocol,indent=2))
    vector_path=out/'vectors.npy'
    if vector_path.exists():vectors=np.load(vector_path,allow_pickle=False)
    else:
        vectors=np.asarray(list(model.embed(texts,batch_size=32)),dtype=np.float32);vectors/=np.linalg.norm(vectors,axis=1,keepdims=True);np.save(vector_path,vectors)
    if vectors.shape!=(len(texts),384) or not np.isfinite(vectors).all():raise ValueError('Invalid vectors')
    rows=[]
    for i,job in enumerate(jobs):
        similarities=vectors[:len(ids)]@vectors[len(ids)+i]
        available=[k for k,eid in enumerate(ids) if documents[eid]['expert_id']==job['expert_id']]
        chosen=sorted(available,key=lambda k:(-float(similarities[k]),ids[k]))[:len(job['sources'])]
        sources={ids[k]:dict(documents[ids[k]],score=float(similarities[k])) for k in chosen}
        rows.append(dict(job,sources=sources))
    (out/'jobs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    report=dict(jobs=len(rows),papers=len(ids),input_sha256=protocol['input_sha256'],vectors_sha256=hashlib.sha256(vector_path.read_bytes()).hexdigest(),
        changed_sources=sum(set(a['sources'])!=set(b['sources']) for a,b in zip(jobs,rows)),accepted_dataset_labels=0)
    (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':run()
