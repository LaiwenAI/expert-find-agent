"""Frozen local ONNX encoder benchmark. Inference only, no training."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from .corpus import records
from .public_benchmark import public_metrics
from .progressive_dense import top_indices


def run(model_name='BAAI/bge-small-en-v1.5',output='artifacts/public/scifact-local-bge',corpus='artifacts/public/scifact-adapted'):
    from fastembed import TextEmbedding
    root=Path(corpus);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    docs=[r for _,r in records(root/'evidence.jsonl')];queries=[r for _,r in records(root/'queries.jsonl')]
    start=time.perf_counter();model=TextEmbedding(model_name=model_name,cache_dir='artifacts/models',threads=1);load=time.perf_counter()-start
    spec=dict(model=model_name,threads=1,corpus_sha256=hashlib.sha256((root/'evidence.jsonl').read_bytes()).hexdigest())
    marker=out/'config.json'
    if marker.exists() and json.loads(marker.read_text())!=spec:raise ValueError('Changed local encoder configuration')
    marker.write_text(json.dumps(spec,indent=2))
    query_spec=dict(queries_sha256=hashlib.sha256((root/'queries.jsonl').read_bytes()).hexdigest())
    query_marker=out/'query-config.json'
    if query_marker.exists() and json.loads(query_marker.read_text())!=query_spec:raise ValueError('Changed query set')
    query_marker.write_text(json.dumps(query_spec,indent=2))
    vectors=out/'vectors.npy'
    if not vectors.exists():
        start=time.perf_counter();parts=out/'parts';parts.mkdir(exist_ok=True);blocks=[]
        for offset in range(0,len(docs),512):
            part=parts/(str(offset)+'.npy');batch=docs[offset:offset+512]
            if part.exists():block=np.load(part)
            else:
                block=np.asarray(list(model.passage_embed([r['text'] for r in batch],batch_size=32)),dtype=np.float32)
                block/=np.maximum(np.linalg.norm(block,axis=1,keepdims=True),1e-30)
                partial=part.with_suffix('.partial.npy');np.save(partial,block);partial.replace(part)
            if block.ndim!=2 or len(block)!=len(batch) or not np.isfinite(block).all():raise ValueError('Invalid embedding checkpoint')
            blocks.append(block)
        x=np.concatenate(blocks);partial=vectors.with_suffix('.partial.npy');np.save(partial,x);partial.replace(vectors)
        (out/'index-cost.json').write_text(json.dumps({'encoding_seconds':time.perf_counter()-start,'documents':len(docs)},indent=2))
    matrix=np.load(vectors);ids=[r['id'] for r in docs]
    for q in queries:
        target=out/f"query-{q['id']}.json"
        if target.exists():continue
        start=time.perf_counter();v=np.asarray(list(model.query_embed(q['query']))[0],dtype=np.float32);v/=np.linalg.norm(v);encoded=time.perf_counter()
        scores=matrix@v;ranking=[ids[i] for i in top_indices(scores,ids,10)];done=time.perf_counter()
        target.write_text(json.dumps(dict(query_id=q['id'],query_type=q.get('query_type'),ranking=ranking,metrics=public_metrics(ranking,q['qrels']),encoding_seconds=encoded-start,retrieval_seconds=done-encoded,total_seconds=done-start),indent=2))
    rows=[json.loads((out/f"query-{q['id']}.json").read_text()) for q in queries]
    times=[r['total_seconds'] for r in rows]
    report=dict(model=model_name,corpus=str(root),queries=len(rows),model_load_seconds=load,ndcg_gain='linear (TREC/BEIR)',metrics={key:float(np.mean([r['metrics'][key] for r in rows])) for key in rows[0]['metrics']},
        p50_seconds=float(np.median(times)),p95_seconds=float(np.percentile(times,95)),warning='Sequential local encoding plus exact retrieval, one ONNX thread; model load and offline corpus encoding separate. English model; no inference about Chinese expert data.')
    (out/'summary.json').write_text(json.dumps(report,indent=2));return report

if __name__=='__main__':print(json.dumps(run(),indent=2))
