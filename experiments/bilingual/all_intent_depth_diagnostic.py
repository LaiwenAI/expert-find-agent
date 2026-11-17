"""Audit cross-language exact top-k containment at several candidate depths."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import time
import numpy as np
import torch
from expert_search.gpu_owned_vector_index import GpuOwnedVectorIndex
from expert_search.owned_vector_index import ExactOwnedVectorIndex
from expert_search.statistics import clustered_paired_interval


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main(output):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Require allocated CUDA node')
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    root=Path('artifacts/bilingual-intent-retrieval-results-v1');summary=json.loads((root/'summary.json').read_text());spec=json.loads((root/'protocol.json').read_text())
    for name,key in (('protocol.json','protocol_sha256'),('private-results.json','private_results_sha256'),('query-vectors.npy','query_vectors_sha256')):
        if sha(root/name)!=summary[key]:raise ValueError('Changed frozen query input')
    source=Path('artifacts/global-expert-branches-v1/branches.jsonl');vp=Path('artifacts/global-expert-branch-benchmark-v1/vectors.npy')
    if sha(source)!=spec['source_sha256'] or sha(vp)!=spec['source_vectors_sha256']:raise ValueError('Changed source inputs')
    records=[json.loads(x) for x in source.read_text().splitlines()];matrix=np.load(vp,mmap_mode='r');queries=np.load(root/'query-vectors.npy',mmap_mode='r');previous=json.loads((root/'private-results.json').read_text())
    if matrix.shape!=(len(records),1024) or queries.shape!=(10000,1024) or len(previous)!=5000:raise ValueError('Changed source or query dimensions')
    ref=ExactOwnedVectorIndex(matrix,[r['id'] for r in records],[[r['expert_id']] for r in records]);gpu=GpuOwnedVectorIndex(ref);gpu.search_vector(queries[0],k=100)
    output.mkdir(parents=True,exist_ok=False)
    protocol=dict(prior_summary_sha256=sha(root/'summary.json'),prior_protocol_sha256=sha(root/'protocol.json'),prior_results_sha256=sha(root/'private-results.json'),query_vectors_sha256=sha(root/'query-vectors.npy'),source_sha256=sha(source),source_vectors_sha256=sha(vp),
        script_sha256=sha(__file__),gpu_index_sha256=sha('src/expert_search/gpu_owned_vector_index.py'),cpu_index_sha256=sha('src/expert_search/owned_vector_index.py'),depths=[10,20,50,100],k=100,precision='FP32, TF32 disabled',
        scope='Frozen query vectors, previously explored construction intents; full source scan, exact with respect to GPU FP32 operations. Containment is ranker agreement, not relevance recall, semantic equivalence, or an end-to-end search speed result.')
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');rows=[];start=time.perf_counter()
    for i,r in enumerate(previous):
        rankings={}
        for j,lang in enumerate(('zh','en')):
            hits=gpu.search_vector(queries[2*i+j],k=100)['hits'];rankings[lang]=[h['id'] for h in hits]
            if len(rankings[lang])!=100 or len(set(rankings[lang]))!=100 or rankings[lang][:10]!=r['ranks'][lang]:raise ValueError('Changed ordered top-ten or incomplete ranking')
        outcomes={}
        for k in (10,20,50,100):
            zh,en=rankings['zh'],rankings['en'];outcomes[str(k)]=dict(zh_top10_in_en_depth=len(set(zh[:10])&set(en[:k]))/10,
                en_top10_in_zh_depth=len(set(en[:10])&set(zh[:k]))/10,pool_overlap=len(set(zh[:k])&set(en[:k]))/k,union_size=len(set(zh[:k])|set(en[:k])))
        rows.append(dict(id=r['id'],parent_id=r['parent_id'],group_id=r['group_id'],intent_type=r['intent_type'],status=r['status'],outcomes=outcomes))
        if (i+1)%1000==0:print('Completed',i+1,flush=True)
    elapsed=time.perf_counter()-start;(output/'private-results.json').write_text(json.dumps(rows,indent=2)+'\n')
    result={}
    for k in (10,20,50,100):
        x=[r['outcomes'][str(k)] for r in rows]
        result[str(k)]={name:statistics.mean(a[name] for a in x) for name in ('zh_top10_in_en_depth','en_top10_in_zh_depth','pool_overlap','union_size')}
        if k>10:
            interval=clustered_paired_interval([r['outcomes'][str(k)]['zh_top10_in_en_depth']-r['outcomes']['10']['zh_top10_in_en_depth'] for r in rows],[r['group_id'] for r in rows]);interval['sampling_unit']='whole company group; query-weighted mean';result[str(k)]['zh_containment_gain_vs_10']=interval
    report=dict(queries=len(rows),matched_ordered_top10_views=len(rows)*2,depths=result,retrieval_seconds=elapsed,gpu=torch.cuda.get_device_name(),node=platform.node(),slurm_job_id=os.environ['SLURM_JOB_ID'],
        protocol_sha256=sha(output/'protocol.json'),results_sha256=sha(output/'private-results.json'),scope=protocol['scope']+' Group bootstrap is query-weighted, unadjusted, and conditional on company-group resampling. Source aggregation and rendering 100 results per view are included in batch time; query encoding, index creation, reasoning and concurrency are excluded. This does not implement an adaptive candidate policy.')
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
