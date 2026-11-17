"""Measure within-parent retrieval overlap for all frozen bilingual intents."""
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
import os
import platform
import statistics
import time
from pathlib import Path
import numpy as np
import torch
from expert_search.gpu_owned_vector_index import GpuOwnedVectorIndex
from expert_search.owned_vector_index import ExactOwnedVectorIndex
from expert_search.qwen_embedding_torch import QwenTorchEmbedding

TASK='Find researchers whose documented research background could help address the technical requirement.'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main(output):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Require allocated CUDA node')
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    root=Path('artifacts/bilingual-intent-retrieval-v1');selection=json.loads((root/'selection.json').read_text())
    if sha(root/'queries.jsonl')!=selection['query_sha256']:raise ValueError('Changed frozen query set')
    rows=[json.loads(x) for x in (root/'queries.jsonl').read_text().splitlines()]
    if len(rows)!=5000 or len({r['id'] for r in rows})!=5000 or len({r['parent_id'] for r in rows})!=1000:raise ValueError('Changed inventory')
    prior=Path('artifacts/bilingual-retrieval-v1');spec=json.loads((prior/'protocol.json').read_text())
    source=Path('artifacts/global-expert-branches-v1/branches.jsonl');vector_path=Path('artifacts/global-expert-branch-benchmark-v1/vectors.npy')
    if sha(source)!=spec['source_sha256'] or sha(vector_path)!=spec['source_vectors_sha256']:raise ValueError('Changed source corpus or vectors')
    checkpoint=Path('artifacts/models/qwen3-embedding-0.6b')
    for name,digest in spec['checkpoint']['files'].items():
        if sha(checkpoint/name)!=digest:raise ValueError('Changed model checkpoint')
    output.mkdir(parents=True,exist_ok=False)
    protocol=dict(selection_sha256=sha(root/'selection.json'),query_sha256=sha(root/'queries.jsonl'),source_sha256=sha(source),source_vectors_sha256=sha(vector_path),checkpoint=spec['checkpoint'],task=TASK,model='Qwen3-Embedding-0.6B',max_length=512,batch_size=16,k=10,
        code_sha256={str(p):sha(p) for p in (Path(__file__),Path('src/expert_search/qwen_embedding_torch.py'),Path('src/expert_search/owned_vector_index.py'),Path('src/expert_search/gpu_owned_vector_index.py'))},
        scope='All five construction intents per parent; bilingual paired retrieval overlap is not semantic intent correctness, independent relevance, expert capability or end-to-end agent latency. No model training or independent human judgments.')
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    start=time.perf_counter();model=QwenTorchEmbedding(max_length=512,batch_size=16);init=time.perf_counter()-start
    texts=['Instruct: '+TASK+'\nQuery:'+r[lang] for r in rows for lang in ('zh','en')]
    lengths=[len(model.tokenizer(x,truncation=False)['input_ids']) for x in texts]
    if max(lengths)>512:raise ValueError('Some queries exceed 512 tokens; refuse silent truncation')
    start=time.perf_counter();queries=model.encode(texts);encoding_seconds=time.perf_counter()-start
    if queries.shape!=(10000,1024) or not np.isfinite(queries).all() or not np.allclose(np.linalg.norm(queries,axis=1),1,atol=2e-5):raise ValueError('Invalid query embeddings')
    np.save(output/'query-vectors.npy',queries);del model;torch.cuda.empty_cache()
    records=[json.loads(line) for line in source.read_text().splitlines()];matrix=np.load(vector_path,mmap_mode='r',allow_pickle=False)
    if matrix.shape!=(len(records),1024):raise ValueError('Changed source dimensions')
    ref=ExactOwnedVectorIndex(matrix,[r['id'] for r in records],[[r['expert_id']] for r in records]);gpu=GpuOwnedVectorIndex(ref);gpu.search_vector(queries[0],k=10)
    results=[];start=time.perf_counter()
    for i,row in enumerate(rows):
        ranks={}
        for j,lang in enumerate(('zh','en')):
            hits=gpu.search_vector(queries[2*i+j],k=10)['hits']
            if len(hits)!=10:raise ValueError('Incomplete result')
            ranks[lang]=[h['id'] for h in hits]
        results.append(dict(id=row['id'],parent_id=row['parent_id'],group_id=row['group_id'],intent_type=row['intent_type'],status=row['status'],ranks=ranks))
        if (i+1)%1000==0:print('Completed',i+1,flush=True)
    retrieval_seconds=time.perf_counter()-start
    (output/'private-results.json').write_text(json.dumps(results,indent=2)+'\n')
    groups=defaultdict(list)
    for r in results:groups[r['parent_id']].append(r)
    if len(groups)!=1000 or any(len(v)!=5 or len({x['intent_type'] for x in v})!=5 for v in groups.values()):raise ValueError('Invalid parent groups')
    comparisons=[]
    for parent,items in groups.items():
        for a,b in itertools.combinations(sorted(items,key=lambda x:x['intent_type']),2):
            comparisons.append(dict(parent_id=parent,intents=[a['intent_type'],b['intent_type']],both_retained=all(x['status']=='retained_model_screened_pair' for x in (a,b)),
                overlap={lang:len(set(a['ranks'][lang])&set(b['ranks'][lang]))/10 for lang in ('zh','en')}))
    def aggregate(items):
        return {lang:dict(mean_top10_overlap=statistics.mean(x['overlap'][lang] for x in items),identical_sets=sum(x['overlap'][lang]==1 for x in items),disjoint_sets=sum(x['overlap'][lang]==0 for x in items)) for lang in ('zh','en')}
    summary=dict(parents=len(groups),queries=len(rows),query_views=len(queries),intent_pairs=len(comparisons),source_records=len(records),experts=len(ref.owners),results=aggregate(comparisons),
        by_intent_pair={' / '.join(k):aggregate([x for x in comparisons if tuple(x['intents'])==k]) for k in sorted({tuple(x['intents']) for x in comparisons})},
        retained_pair_count=sum(x['both_retained'] for x in comparisons),retained_results=aggregate([x for x in comparisons if x['both_retained']]),
        max_query_tokens=max(lengths),model_init_seconds=init,batch_encoding_seconds=encoding_seconds,gpu_scoring_seconds=retrieval_seconds,gpu=torch.cuda.get_device_name(),node=platform.node(),slurm_job_id=os.environ['SLURM_JOB_ID'],
        protocol_sha256=sha(output/'protocol.json'),private_results_sha256=sha(output/'private-results.json'),query_vectors_sha256=sha(output/'query-vectors.npy'),scope=protocol['scope']+' Timings are batch processing, exclude source encoding, loading and model reasoning, and are not single-request or service latencies. All pairs are dependent within parent; no relevance labels.')
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='by_intent_pair'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
