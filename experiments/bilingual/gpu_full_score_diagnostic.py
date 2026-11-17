"""Compare full-source CUDA scoring with CPU exact aggregation on frozen queries."""
import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import time
import numpy as np
from expert_search.gpu_owned_vector_index import GpuOwnedVectorIndex
from expert_search.owned_vector_index import ExactOwnedVectorIndex
from expert_search.statistics import clustered_paired_interval
from experiments.bilingual.adaptive_effort_diagnostic import sha


def main(output):
    import torch
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Run on a Slurm-allocated CUDA compute node')
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    root=Path('artifacts/bilingual-retrieval-v1');selection=json.loads((root/'selection.json').read_text());encoding=json.loads((root/'encoding.json').read_text());prior=json.loads((root/'summary.json').read_text());spec=json.loads((root/'protocol.json').read_text())
    source=Path('artifacts/global-expert-branches-v1/branches.jsonl');vp=Path('artifacts/global-expert-branch-benchmark-v1/vectors.npy')
    if sha(root/'queries.jsonl')!=selection['queries_sha256'] or sha(root/'query-vectors.npz')!=encoding['query_vectors_sha256'] or sha(root/'protocol.json')!=prior['protocol_sha256'] or sha(source)!=spec['source_sha256'] or sha(vp)!=spec['source_vectors_sha256']:raise ValueError('Changed frozen inputs')
    queries=[json.loads(line) for line in (root/'queries.jsonl').read_text().splitlines()]
    with np.load(root/'query-vectors.npz',allow_pickle=False) as data:q=data['vectors'].copy();qids=data['ids'].tolist()
    if len(queries)!=1000 or qids!=[r['id'] for r in queries] or q.shape!=(2000,1024) or not np.isfinite(q).all():raise ValueError('Changed query inventory')
    records=[json.loads(line) for line in source.read_text().splitlines()];ids=[r['id'] for r in records];owners=[[r['expert_id']] for r in records];matrix=np.load(vp,mmap_mode='r',allow_pickle=False)
    if matrix.shape!=(len(records),1024):raise ValueError('Changed source inventory')
    output.mkdir(parents=True,exist_ok=False)
    protocol={'pairs':1000,'views':2000,'query_selection':'all frozen construction pairs','source_sha256':sha(source),'source_vectors_sha256':sha(vp),
              'queries_sha256':sha(root/'queries.jsonl'),'query_vectors_sha256':sha(root/'query-vectors.npz'),'prior_protocol_sha256':sha(root/'protocol.json'),
              'methods':['cpu_full','cuda_full'],'k':10,'precision':'FP32; CUDA TF32 disabled','timing':'One alternating-order end-to-end vector scoring and CPU aggregation per language view, after warmup. CUDA includes per-query upload, score download and synchronization.',
              'code_sha256':{str(p):sha(p) for p in (Path(__file__),Path('src/expert_search/gpu_owned_vector_index.py'),Path('src/expert_search/owned_vector_index.py'),Path('src/expert_search/statistics.py'))},
              'scope':'Reused construction pairs and source vectors. CUDA and CPU floating-point reductions may differ. Fidelity to CPU scores is not expert relevance; no model inference or semantic verification.'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    start=time.perf_counter();reference=ExactOwnedVectorIndex(matrix,ids,owners);cpu_init=time.perf_counter()-start
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();before=torch.cuda.memory_allocated();start=time.perf_counter();gpu=GpuOwnedVectorIndex(reference);gpu_init=time.perf_counter()-start
    resident=torch.cuda.memory_allocated()-before
    reference.search_vector(q[0],k=10);gpu.search_vector(q[0],k=10)
    rows=[]
    for i,row in enumerate(queries):
        for j,lang in enumerate(('zh','en')):
            order=('cpu','gpu') if (2*i+j)%2==0 else ('gpu','cpu');results={};times={}
            for method in order:
                torch.cuda.synchronize();start=time.perf_counter();results[method]=(reference if method=='cpu' else gpu).search_vector(q[2*i+j],k=10);times[method]=time.perf_counter()-start
            a,b=results['cpu']['hits'],results['gpu']['hits'];cpuids=[h['id'] for h in a];gpuids=[h['id'] for h in b]
            if len(cpuids)!=10 or len(gpuids)!=10:raise ValueError('Incomplete ranking')
            ac={h['id']:h for h in a};bc={h['id']:h for h in b};shared=set(ac)&set(bc)
            rows.append({'id':row['id'],'group_id':row['group_id'],'language':lang,'seconds':times,
                'ordered_equal':cpuids==gpuids,'top10_retained':len(shared)/10,
                'ordered_top_source_equal':cpuids==gpuids and all(x['evidence'][0]['id']==y['evidence'][0]['id'] for x,y in zip(a,b)),
                'max_shared_score_difference':max((abs(ac[e]['score']-bc[e]['score']) for e in shared),default=0.)})
        if (i+1)%100==0:print('Completed pairs',i+1,flush=True)
    (output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    interval=clustered_paired_interval([r['seconds']['gpu']-r['seconds']['cpu'] for r in rows],[r['group_id'] for r in rows]);interval['sampling_unit']='whole company group; language-view-weighted mean'
    report={'pairs':len(queries),'views':len(rows),'ordered_equal':sum(r['ordered_equal'] for r in rows),'ordered_top_source_equal':sum(r['ordered_top_source_equal'] for r in rows),
            'mean_top10_retention':statistics.mean(r['top10_retained'] for r in rows),'max_shared_score_difference':max(r['max_shared_score_difference'] for r in rows),
            'mean_seconds':{a:statistics.mean(r['seconds'][a] for r in rows) for a in ('cpu','gpu')},'paired_gpu_minus_cpu_seconds':interval,
            'cpu_initialization_seconds':cpu_init,'gpu_initialization_seconds':gpu_init,'gpu_resident_allocated_bytes':resident,'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'node':platform.node(),'slurm_job_id':os.environ['SLURM_JOB_ID'],
            'protocol_sha256':sha(output/'protocol.json'),'results_sha256':sha(output/'results.json'),
            'scope':protocol['scope']+' One observation per view, one allocated GPU and one CPU thread, no competing-load experiment. Mean timing includes GPU transfer and output aggregation but excludes initialization, query encoding and index/model construction. GPU precision is not an exact CPU-score guarantee; metadata and process/device reserved memory exceed measured tensor allocation.'}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
