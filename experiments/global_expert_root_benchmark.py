"""Full-corpus frozen root encoding and known-positive coarse-routing audit."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.progressive_dense import top_indices

TASK='Find researchers whose documented research background could help address the technical requirement.'


def run():
    parent=Path('artifacts/global-expert-roots-v1');roots=[json.loads(l) for l in (parent/'roots.jsonl').read_text().splitlines()]
    queries=[json.loads(l) for l in (parent/'queries.jsonl').read_text().splitlines()]
    manifest=json.loads((parent/'manifest.json').read_text());out=Path('artifacts/global-expert-root-benchmark-v1');out.mkdir(exist_ok=True)
    checkpoint=json.loads(Path('artifacts/models/qwen3-embedding-0.6b/checkpoint-provenance.json').read_text())
    protocol=dict(version=1,roots_sha256=hashlib.sha256((parent/'roots.jsonl').read_bytes()).hexdigest(),queries_sha256=hashlib.sha256((parent/'queries.jsonl').read_bytes()).hexdigest(),checkpoint=checkpoint,
        root_max_length=1024,query_max_length=4096,batch_size=8,chunk_size=64,task=TASK,budgets=manifest['planned_candidate_budgets'],
        scope='Full 9861 available roots, explicit missing-data bucket. Two reused known silver cases; full-demand and condition diagnostics separate. No precision or global relevance recall claim.')
    if protocol['roots_sha256']!=manifest['roots_sha256']:raise ValueError('Changed root corpus')
    marker=out/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=protocol:raise ValueError('Changed global routing protocol')
    marker.write_text(json.dumps(protocol,indent=2));start=time.perf_counter();model=QwenTorchEmbedding(max_length=1024,batch_size=8);initialization=time.perf_counter()-start
    chunks=[];total=0.
    for start in range(0,len(roots),64):
        group=roots[start:start+64];target=out/f'batch-{start}.npz';fingerprint=hashlib.sha256(json.dumps(group,sort_keys=True).encode()).hexdigest()
        if target.exists():
            with np.load(target,allow_pickle=False) as data:
                if data['fingerprint'].item()!=fingerprint:raise ValueError('Changed root chunk')
                vectors=data['vectors'];elapsed=float(data['seconds'])
        else:
            begin=time.perf_counter();vectors=model.encode([r['text'] for r in group]);elapsed=time.perf_counter()-begin
            temporary=out/f'batch-{start}.partial.npz';np.savez(temporary,vectors=vectors,fingerprint=fingerprint,seconds=elapsed);temporary.replace(target)
        if vectors.shape!=(len(group),1024) or not np.isfinite(vectors).all():raise ValueError('Invalid root vectors')
        chunks.append(vectors);total+=elapsed;print('encoded',start+len(group),'/',len(roots),flush=True)
    matrix=np.concatenate(chunks);np.save(out/'vectors.npy',matrix);ids=[r['expert_id'] for r in roots];model.max_length=4096;rows=[]
    for query in queries:
        target=out/(query['id']+'.json')
        if target.exists():row=json.loads(target.read_text())
        else:
            begin=time.perf_counter();q=model.encode(['Instruct: '+TASK+'\nQuery:'+query['text']],query=False)[0];encoding=time.perf_counter()-begin
            begin=time.perf_counter();scores=matrix@q;chosen=top_indices(scores,ids,max(protocol['budgets']));retrieval=time.perf_counter()-begin
            full=sorted(range(len(ids)),key=lambda i:(-float(scores[i]),ids[i]));known_position=ids.index(query['known_expert_id'])
            row=dict(id=query['id'],parent=query['parent'],kind=query['kind'],known_expert_rank=full.index(known_position)+1,
                retention={str(k):known_position in chosen[:k] for k in protocol['budgets']},query_encoding_seconds=encoding,retrieval_seconds=retrieval,
                ranking=[ids[i] for i in chosen],scores=[float(scores[i]) for i in chosen])
            temporary=target.with_suffix('.partial');temporary.write_text(json.dumps(row,indent=2));temporary.replace(target)
        rows.append(row);print(query['id'],row['known_expert_rank'],flush=True)
    report=dict(experts=len(roots),total_experts=manifest['total_experts'],unavailable_experts=len(manifest['unavailable_experts']),initialization_seconds=initialization,root_encoding_seconds=total,
        vectors_sha256=hashlib.sha256((out/'vectors.npy').read_bytes()).hexdigest(),rows=[{k:v for k,v in r.items() if k not in ('ranking','scores')} for r in rows],
        warning='Top500 stored per query. Known silver positives are incomplete and reused. Requirement planning latency excluded and not measured in earlier v1. Root scoring does not verify ability or read source evidence. Missing roots must not be treated as negative experts.')
    (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__':run()
