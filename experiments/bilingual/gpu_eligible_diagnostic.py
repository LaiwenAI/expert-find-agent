"""Compare candidate-scoped CPU and GPU scoring with full-source retrieval."""
import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import time
import numpy as np
from expert_search.compiled_expert_search import CompiledExpertSearch
from expert_search.gpu_owned_vector_index import GpuOwnedVectorIndex
from expert_search.owned_vector_index import ExactOwnedVectorIndex
from expert_search.statistics import clustered_paired_interval
from experiments.bilingual.adaptive_effort_diagnostic import sha


def main(output):
    import torch
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Run on a Slurm-allocated CUDA compute node')
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    root=Path('artifacts/bilingual-retrieval-v1');selection=json.loads((root/'selection.json').read_text());encoding=json.loads((root/'encoding.json').read_text());spec=json.loads((root/'protocol.json').read_text())
    candidate_root=Path('artifacts/bilingual-eligible-candidates-v1');candidate_protocol=json.loads((candidate_root/'protocol.json').read_text());candidate_summary=json.loads((candidate_root/'summary.json').read_text())
    source=Path('artifacts/global-expert-branches-v1/branches.jsonl');vp=Path('artifacts/global-expert-branch-benchmark-v1/vectors.npy')
    if sha(root/'queries.jsonl')!=selection['queries_sha256'] or sha(root/'query-vectors.npz')!=encoding['query_vectors_sha256'] or sha(root/'queries.jsonl')!=candidate_protocol['queries_sha256'] or sha(candidate_root/'protocol.json')!=candidate_summary['protocol_sha256'] or sha(candidate_root/'candidates.json')!=candidate_summary['candidates_sha256'] or sha(source)!=spec['source_sha256'] or sha(vp)!=spec['source_vectors_sha256'] or sha('artifacts/teacher-compiled-lm/metadata.json')!=candidate_protocol['index_metadata_sha256'] or sha('artifacts/teacher-compiled-lm/arrays.npz')!=candidate_protocol['index_arrays_sha256']:
        raise ValueError('Changed frozen inputs')
    queries=[json.loads(line) for line in (root/'queries.jsonl').read_text().splitlines()];candidates=json.loads((candidate_root/'candidates.json').read_text())
    with np.load(root/'query-vectors.npz',allow_pickle=False) as data:q=data['vectors'].copy();qids=data['ids'].tolist()
    if len(queries)!=1000 or len(candidates)!=2000 or qids!=[r['id'] for r in queries] or q.shape!=(2000,1024):raise ValueError('Changed inventory')
    if any(c['id']!=queries[i//2]['id'] or c['language']!=('zh','en')[i%2] or c['group_id']!=queries[i//2]['group_id'] or len(c['eligible'])!=50 or len(set(c['eligible']))!=50 for i,c in enumerate(candidates)):raise ValueError('Changed candidates')
    records=[json.loads(line) for line in source.read_text().splitlines()];ids=[r['id'] for r in records];owners=[[r['expert_id']] for r in records];matrix=np.load(vp,mmap_mode='r',allow_pickle=False)
    if matrix.shape!=(len(records),1024):raise ValueError('Changed source inventory')
    output.mkdir(parents=True,exist_ok=False)
    protocol={'pairs':1000,'views':2000,'query_vectors_sha256':sha(root/'query-vectors.npz'),'source_sha256':sha(source),'source_vectors_sha256':sha(vp),'candidates_sha256':sha(candidate_root/'candidates.json'),
              'candidate_protocol_sha256':sha(candidate_root/'protocol.json'),'candidate_selection':'candidate-LM top 50, recomputed and compared with frozen list per view',
              'methods':['cpu_full','gpu_full','cpu_selected','gpu_selected','candidate_lm_plus_gpu'],'k':10,'precision':'FP32, CUDA TF32 disabled',
              'timing':'One rotating-order measurement per view after warmup. GPU scoring includes per-query upload, selected-row gather, score download and CPU aggregation. Candidate pipeline measured as a separate sequential call.',
              'code_sha256':{str(p):sha(p) for p in (Path(__file__),Path('src/expert_search/gpu_owned_vector_index.py'),Path('src/expert_search/owned_vector_index.py'),Path('src/expert_search/compiled_expert_search.py'),Path('src/expert_search/compiled_language_model.py'),Path('src/expert_search/certified_topk.py'),Path('src/expert_search/retrieval.py'),Path('src/expert_search/statistics.py'))},
              'scope':'Reused bilingual construction queries; selected candidates have no relevance labels. Agreement with full dense ranking is fidelity, not expert relevance or evidence support. GPU and CPU FP32 reductions may differ.'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    start=time.perf_counter();ranker=CompiledExpertSearch();ranker_init=time.perf_counter()-start
    if ranker.snapshot!=candidate_protocol['corpus_snapshot_sha256']:raise ValueError('Changed candidate corpus')
    start=time.perf_counter();reference=ExactOwnedVectorIndex(matrix,ids,owners);reference_init=time.perf_counter()-start
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();before=torch.cuda.memory_allocated();start=time.perf_counter();gpu=GpuOwnedVectorIndex(reference);gpu_init=time.perf_counter()-start;resident=torch.cuda.memory_allocated()-before
    if any(set(c['eligible'])-reference.owners.keys() for c in candidates):raise ValueError('Candidate without vector sources')
    first=candidates[0]['eligible'];reference.search_vector(q[0],k=10,eligible=first);gpu.search_vector(q[0],k=10,eligible=first);gpu.search_vector(q[0],k=10)
    names=('cpu_full','gpu_full','cpu_selected','gpu_selected');rows=[]
    for i,c in enumerate(candidates):
        query=queries[i//2][c['language']];start=time.perf_counter();ranked=ranker.search(query,50);rank_seconds=time.perf_counter()-start;eligible=[x['expert']['id'] for x in ranked['results']]
        if eligible!=c['eligible']:raise ValueError('Candidate list differs from frozen local construction')
        order=names[i%4:]+names[:i%4];results={};timings={}
        for arm in order:
            torch.cuda.synchronize();started=time.perf_counter()
            results[arm]=(reference if arm.startswith('cpu') else gpu).search_vector(q[i],k=10,eligible=eligible if arm.endswith('selected') else None)
            timings[arm]=time.perf_counter()-started
        # Actual combined sequence, not the sum of separately interleaved timings.
        torch.cuda.synchronize();started=time.perf_counter();again=ranker.search(query,50);again_ids=[x['expert']['id'] for x in again['results']];pipeline=gpu.search_vector(q[i],k=10,eligible=again_ids);timings['candidate_lm_plus_gpu']=time.perf_counter()-started
        if again_ids!=eligible or pipeline['hits']!=results['gpu_selected']['hits']:raise ValueError('Combined path differs')
        extract=lambda hits:[h['id'] for h in hits];source_ids=lambda hits:[[x['id'] for x in h['evidence']] for h in hits]
        cpu_full=results['cpu_full']['hits'];cpu_sel=results['cpu_selected']['hits'];gpu_sel=results['gpu_selected']['hits'];gpu_full=results['gpu_full']['hits']
        if any(len(h)!=10 for h in (cpu_full,cpu_sel,gpu_sel,gpu_full)):raise ValueError('Incomplete result')
        rows.append({'id':c['id'],'group_id':c['group_id'],'language':c['language'],'rank_seconds':rank_seconds,'seconds':timings,
                     'source_rows':results['gpu_selected']['source_rows'],
                     'gpu_selected_order_equal':extract(cpu_sel)==extract(gpu_sel),
                     'gpu_selected_evidence_equal':extract(cpu_sel)==extract(gpu_sel) and source_ids(cpu_sel)==source_ids(gpu_sel),
                     'gpu_full_order_equal':extract(cpu_full)==extract(gpu_full),
                     'candidate_top10_full_recall':len(set(eligible)&set(extract(cpu_full)))/10,
                     'selected_top10_full_recall':len(set(extract(cpu_sel))&set(extract(cpu_full)))/10})
        if (i+1)%200==0:print('Completed views',i+1,flush=True)
    (output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    delta=clustered_paired_interval([r['seconds']['gpu_selected']-r['seconds']['cpu_selected'] for r in rows],[r['group_id'] for r in rows]);delta['sampling_unit']='whole company group; language-view-weighted mean'
    report={'views':len(rows),'pairs':len(queries),'selected_gpu_order_equal':sum(r['gpu_selected_order_equal'] for r in rows),'selected_gpu_evidence_equal':sum(r['gpu_selected_evidence_equal'] for r in rows),'full_gpu_order_equal':sum(r['gpu_full_order_equal'] for r in rows),
            'mean_candidate_top10_full_recall':statistics.mean(r['candidate_top10_full_recall'] for r in rows),'mean_selected_top10_full_recall':statistics.mean(r['selected_top10_full_recall'] for r in rows),
            'by_language':{lang:{'views':sum(r['language']==lang for r in rows),'mean_candidate_top10_full_recall':statistics.mean(r['candidate_top10_full_recall'] for r in rows if r['language']==lang),'mean_selected_top10_full_recall':statistics.mean(r['selected_top10_full_recall'] for r in rows if r['language']==lang)} for lang in ('zh','en')},
            'mean_selected_source_rows':statistics.mean(r['source_rows'] for r in rows),'mean_rank_seconds':statistics.mean(r['rank_seconds'] for r in rows),
            'mean_seconds':{a:statistics.mean(r['seconds'][a] for r in rows) for a in names+('candidate_lm_plus_gpu',)},'paired_gpu_minus_cpu_selected_seconds':delta,
            'ranker_init_seconds':ranker_init,'reference_init_seconds':reference_init,'gpu_init_seconds':gpu_init,'gpu_resident_allocated_bytes':resident,'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'node':platform.node(),'slurm_job_id':os.environ['SLURM_JOB_ID'],'protocol_sha256':sha(output/'protocol.json'),'results_sha256':sha(output/'results.json'),
            'scope':protocol['scope']+' Timings exclude encoding, initialization and concurrent service load. Candidate pipeline uses a cached embedding rather than measuring query encoding. One observation per view on one GPU and CPU thread. Candidate/full overlap is not relevance recall; numerical agreement is not guaranteed for other inputs.'}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
