"""Reproduce cached GPU ranks and audit source-kind and expert concentration."""
import argparse
from collections import Counter
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
    root=Path('artifacts/bilingual-intent-retrieval-results-v1');old=json.loads((root/'summary.json').read_text());protocol=json.loads((root/'protocol.json').read_text())
    for name,key in (('protocol.json','protocol_sha256'),('private-results.json','private_results_sha256'),('query-vectors.npy','query_vectors_sha256')):
        if sha(root/name)!=old[key]:raise ValueError('Changed frozen retrieval input')
    source=Path('artifacts/global-expert-branches-v1/branches.jsonl');vector_path=Path('artifacts/global-expert-branch-benchmark-v1/vectors.npy')
    if sha(source)!=protocol['source_sha256'] or sha(vector_path)!=protocol['source_vectors_sha256']:raise ValueError('Changed source data')
    records=[json.loads(line) for line in source.read_text().splitlines()];ids=[r['id'] for r in records];owners=[[r['expert_id']] for r in records]
    if len(set(ids))!=len(ids) or any(r['kind'] not in ('paper','profile') for r in records):raise ValueError('Invalid source kinds or IDs')
    kinds={r['id']:r['kind'] for r in records};matrix=np.load(vector_path,mmap_mode='r');queries=np.load(root/'query-vectors.npy',mmap_mode='r');prior=json.loads((root/'private-results.json').read_text())
    if matrix.shape!=(len(records),1024) or queries.shape!=(10000,1024) or len(prior)!=5000:raise ValueError('Invalid input shapes')
    ref=ExactOwnedVectorIndex(matrix,ids,owners);gpu=GpuOwnedVectorIndex(ref);gpu.search_vector(queries[0],k=10)
    output.mkdir(parents=True,exist_ok=False)
    spec=dict(prior_summary_sha256=sha(root/'summary.json'),prior_protocol_sha256=sha(root/'protocol.json'),prior_results_sha256=sha(root/'private-results.json'),query_vectors_sha256=sha(root/'query-vectors.npy'),source_sha256=sha(source),source_vectors_sha256=sha(vector_path),
        script_sha256=sha(__file__),gpu_index_sha256=sha('src/expert_search/gpu_owned_vector_index.py'),cpu_index_sha256=sha('src/expert_search/owned_vector_index.py'),precision='FP32, TF32 disabled',k=10,
        scope='Reused construction pairs; winner source kind and concentration describe supplied records, not verified research contribution, expert relevance, or model bias. Score and rank can vary by hardware; exact prior rank required for this replay.')
    (output/'protocol.json').write_text(json.dumps(spec,indent=2)+'\n');rows=[];start=time.perf_counter()
    for i,r in enumerate(prior):
        views={}
        for j,lang in enumerate(('zh','en')):
            hits=gpu.search_vector(queries[2*i+j],k=10)['hits'];ranking=[h['id'] for h in hits]
            if ranking!=r['ranks'][lang]:raise ValueError('Changed GPU ranking: '+str(i))
            winner=[h['evidence'][0]['id'] for h in hits]
            views[lang]=dict(ranking=ranking,paper_count=sum(kinds[s]=='paper' for s in winner),source_kinds=[kinds[s] for s in winner])
        rows.append(dict(id=r['id'],parent_id=r['parent_id'],group_id=r['group_id'],intent_type=r['intent_type'],status=r['status'],views=views))
        if (i+1)%1000==0:print('Completed',i+1,flush=True)
    seconds=time.perf_counter()-start;(output/'private-results.json').write_text(json.dumps(rows,indent=2)+'\n')
    def aggregate(items):
        z=[r['views']['zh']['paper_count']/10 for r in items];e=[r['views']['en']['paper_count']/10 for r in items]
        interval=clustered_paired_interval([b-a for a,b in zip(z,e)],[r['group_id'] for r in items])
        interval['sampling_unit']='whole company group; query-weighted mean'
        return dict(queries=len(items),zh_mean_paper_fraction=statistics.mean(z),en_mean_paper_fraction=statistics.mean(e),
                    paired_difference=interval,
                    zh_paper_free=sum(a==0 for a in z),en_paper_free=sum(a==0 for a in e))
    concentration={}
    for lang in ('zh','en'):
        counts=Counter(e for r in rows for e in r['views'][lang]['ranking']);values=sorted(counts.values(),reverse=True);total=sum(values);n=max(1,(len(ref.owner_ids)+99)//100)
        concentration[lang]=dict(distinct_returned_experts=len(counts),universe_experts=len(ref.owner_ids),top_one_percent_universe_count=n,top_one_percent_universe_share=sum(values[:n])/total,
            maximum_expert_frequency=values[0],total_ranked_slots=total)
    report=dict(queries=len(rows),matched_ordered_views=len(rows)*2,overall=aggregate(rows),by_intent={t:aggregate([r for r in rows if r['intent_type']==t]) for t in sorted({r['intent_type'] for r in rows})},
        by_status={t:aggregate([r for r in rows if r['status']==t]) for t in sorted({r['status'] for r in rows})},concentration=concentration,retrieval_seconds=seconds,gpu=torch.cuda.get_device_name(),node=platform.node(),slurm_job_id=os.environ['SLURM_JOB_ID'],
        protocol_sha256=sha(output/'protocol.json'),private_results_sha256=sha(output/'private-results.json'),scope=spec['scope']+' Bootstrap resamples company groups and weights query views; estimates are descriptive and unadjusted. Concentration uses repeated queries, not independent users; source labels and individual credit are unverified. Times exclude encoding, initialization and model reasoning.')
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('by_intent','by_status')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
