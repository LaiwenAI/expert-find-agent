"""Whole-demand paper-only retrieval over the frozen full source matrix."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from expert_search.qwen_embedding_torch import QwenTorchEmbedding

TASK='Find researchers whose documented research background could help address the technical requirement.'


def run():
    parent=Path('artifacts/demand-first-paper-retrieval-v1');queries=[json.loads(l) for l in (parent/'queries.jsonl').read_text().splitlines()]
    source=Path('artifacts/global-expert-branches-v1/branches.jsonl');records=[json.loads(l) for l in source.read_text().splitlines()]
    index=Path('artifacts/global-expert-branch-benchmark-v1');spec=json.loads((index/'protocol.json').read_text());summary=json.loads((index/'summary.json').read_text())
    if hashlib.sha256(source.read_bytes()).hexdigest()!=spec['roots_sha256']:raise ValueError('Changed source order')
    with (index/'vectors.npy').open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=summary['vectors_sha256']:raise ValueError('Changed source vectors')
    full=np.load(index/'vectors.npy',mmap_mode='r',allow_pickle=False);positions=[i for i,r in enumerate(records) if r['kind']=='paper'];papers=[records[i] for i in positions];matrix=np.ascontiguousarray(full[positions])
    out=parent/'results';out.mkdir(exist_ok=True)
    protocol=dict(version=1,queries_sha256=hashlib.sha256((parent/'queries.jsonl').read_bytes()).hexdigest(),branches_sha256=spec['roots_sha256'],vectors_sha256=summary['vectors_sha256'],task=TASK,query_max_length=4096,paper_max_length=512,source_prefix_chars=1200,experts_per_demand=1,papers_per_expert=3,
                  selection='First expert by maximum paper cosine, deterministic expert/source ID ties. All fixed 20 demands, paper sources only, no labels or topic-filtered teacher pool.')
    marker=parent/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=protocol:raise ValueError('Changed retrieval protocol')
    marker.write_text(json.dumps(protocol,indent=2));model=QwenTorchEmbedding(max_length=4096,batch_size=8);rows=[]
    for query in queries:
        target=out/(query['id']+'.json')
        if target.exists():row=json.loads(target.read_text())
        else:
            start=time.perf_counter();q=model.encode(['Instruct: '+TASK+'\nQuery:'+query['demand']],query=False)[0];encoding=time.perf_counter()-start
            start=time.perf_counter();scores=matrix@q
            best=min(range(len(papers)),key=lambda i:(-float(scores[i]),papers[i]['expert_id'],papers[i]['id']));eid=papers[best]['expert_id']
            chosen=sorted((i for i,p in enumerate(papers) if p['expert_id']==eid),key=lambda i:(-float(scores[i]),papers[i]['id']))[:3]
            row=dict(id=query['id'],demand=query['demand'],expert_id=eid,sources={papers[i]['id']:papers[i] for i in chosen},scores={papers[i]['id']:float(scores[i]) for i in chosen},encoding_seconds=encoding,retrieval_seconds=time.perf_counter()-start)
            temp=target.with_suffix('.partial');temp.write_text(json.dumps(row,ensure_ascii=False,indent=2));temp.replace(target)
        rows.append(row);print(query['id'],'retrieved',len(row['sources']),flush=True)
    report=dict(queries=len(rows),paper_sources=len(papers),distinct_experts=len({r['expert_id'] for r in rows}),mean_encoding_seconds=float(np.mean([r['encoding_seconds'] for r in rows])),mean_retrieval_seconds=float(np.mean([r['retrieval_seconds'] for r in rows])),warning='Candidate construction inputs only, no capability labels or expertise precision. Single whole-demand embedding may miss useful components.')
    (parent/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':run()
