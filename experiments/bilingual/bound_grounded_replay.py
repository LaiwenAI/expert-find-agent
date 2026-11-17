"""Replay cached vector retrieval with and without strict source binding."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import time
import numpy as np
import torch
from expert_search.expert_catalog import ExpertCatalog
from expert_search.grounded_pipeline import GroundedPipeline
from expert_search.gpu_owned_vector_index import GpuOwnedVectorIndex
from expert_search.owned_vector_index import ExactOwnedVectorIndex
from expert_search.statistics import clustered_paired_interval
from expert_search.vector_catalog_binding import VectorCatalogBinding


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


class CachedEncoder:
    def __init__(self):self.vector=None
    def encode(self,texts,query=False):return np.asarray([self.vector],dtype=np.float32)


class Planner:
    def plan(self,demand):return {'requirements':{'r':{'text':demand}}}


class Assessor:
    def assess(self,demand,eid,requirements,sources):
        if not sources:raise ValueError('No evidence')
        return dict(status='structurally_valid',result=dict(expert_id=eid,actionable_coverage=0,direct_coverage=0,topic_coverage=1))


def main(output):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Require allocated CUDA node')
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    roots=Path('artifacts/global-expert-branches-v1');parent=Path('artifacts/global-expert-branch-benchmark-v1');prior=Path('artifacts/bilingual-retrieval-v1')
    root_manifest=json.loads((roots/'manifest.json').read_text());parent_summary=json.loads((parent/'summary.json').read_text());selection=json.loads((prior/'selection.json').read_text());encoding=json.loads((prior/'encoding.json').read_text())
    branch=roots/'branches.jsonl';vp=parent/'vectors.npy'
    if sha(branch)!=root_manifest['branches_sha256'] or sha(vp)!=parent_summary['vectors_sha256'] or sha(prior/'query-vectors.npz')!=encoding['query_vectors_sha256'] or sha(prior/'queries.jsonl')!=selection['queries_sha256'] or sha('artifacts/corpus/evidence.jsonl')!=root_manifest['source_sha256']:
        raise ValueError('Changed frozen replay inputs')
    rows=[json.loads(x) for x in (prior/'queries.jsonl').read_text().splitlines()];records=[json.loads(x) for x in branch.read_text().splitlines()]
    with np.load(prior/'query-vectors.npz',allow_pickle=False) as data:queries=data['vectors'].copy();ids=data['ids'].tolist()
    if len(rows)!=1000 or ids!=[r['id'] for r in rows] or queries.shape!=(2000,1024):raise ValueError('Changed paired cohort')
    output.mkdir(parents=True,exist_ok=False)
    protocol=dict(selection='Fixed-seed sample of 200 prior Chinese construction views',seed=20261125,queries=200,source_sha256=root_manifest['source_sha256'],branches_sha256=sha(branch),vectors_sha256=sha(vp),query_vectors_sha256=sha(prior/'query-vectors.npz'),queries_sha256=sha(prior/'queries.jsonl'),
        code_sha256={str(p):sha(p) for p in (Path(__file__),Path('src/expert_search/vector_catalog_binding.py'),Path('src/expert_search/expert_catalog.py'),Path('src/expert_search/grounded_pipeline.py'),Path('src/expert_search/gpu_owned_vector_index.py'),Path('src/expert_search/owned_vector_index.py'))},
        scope='Cached query vectors with deterministic mock planner and assessor. Compares strict binding and unbound execution for complete result equivalence. No live reasoning, independently judged relevance, encoding or production latency.')
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    start=time.perf_counter();catalog=ExpertCatalog();catalog_init=time.perf_counter()-start
    matrix=np.load(vp,mmap_mode='r',allow_pickle=False);start=time.perf_counter();ref=ExactOwnedVectorIndex(matrix,[r['id'] for r in records],[[r['expert_id']] for r in records]);gpu=GpuOwnedVectorIndex(ref);index_init=time.perf_counter()-start
    start=time.perf_counter();binding=VectorCatalogBinding(gpu,catalog,branch,vp,branches_sha256=root_manifest['branches_sha256'],vectors_sha256=parent_summary['vectors_sha256']);binding_init=time.perf_counter()-start
    encoder=CachedEncoder();planner=Planner();assessor=Assessor();pipelines={'bound':GroundedPipeline(encoder,gpu,catalog,planner,assessor,binding=binding,require_binding=True),'unbound':GroundedPipeline(encoder,gpu,catalog,planner,assessor)}
    chosen=sorted(random.Random(20261125).sample(range(len(rows)),200));out=[]
    for i,idx in enumerate(chosen):
        encoder.vector=queries[2*idx];times={'bound':[],'unbound':[]};reference=None
        for repeat in range(-1,3):
            for arm in (('bound','unbound') if (i+repeat)%2==0 else ('unbound','bound')):
                torch.cuda.synchronize();start=time.perf_counter();events=pipelines[arm].run('Cached retrieval replay',k=5,candidates=10);elapsed=time.perf_counter()-start
                result=events[-1];retrieval=next(e for e in events if e['stage']=='retrieval')
                if result['stage']!='result' or result['status']!='provisional' or retrieval['index_catalog_binding']!=('validated_at_initialization' if arm=='bound' else 'unverified'):raise ValueError('Unexpected pipeline result')
                signature=(retrieval['route']['hits'],result['ranking'],result['unresolved_experts'])
                if reference is None:reference=signature
                elif signature!=reference:raise ValueError('Changed complete result')
                if repeat>=0:times[arm].append(elapsed)
        out.append(dict(id=rows[idx]['id'],group_id=rows[idx]['group_id'],seconds=times))
        if (i+1)%50==0:print('Completed',i+1,flush=True)
    catalog.db.close();(output/'private-results.json').write_text(json.dumps(out,indent=2)+'\n')
    means={a:[statistics.mean(r['seconds'][a]) for r in out] for a in ('bound','unbound')};interval=clustered_paired_interval([a-b for a,b in zip(means['bound'],means['unbound'])],[r['group_id'] for r in out]);interval['sampling_unit']='whole company group; query-weighted mean'
    report=dict(queries=len(out),matched_complete_results=len(out)*8,catalog_init_seconds=catalog_init,index_init_seconds=index_init,binding_init_seconds=binding_init,bound_sources=binding.sources,
        mean_seconds={a:statistics.mean(means[a]) for a in means},bound_minus_unbound=interval,protocol_sha256=sha(output/'protocol.json'),results_sha256=sha(output/'private-results.json'),
        slurm_job_id=os.environ['SLURM_JOB_ID'],gpu=torch.cuda.get_device_name(),scope=protocol['scope']+' Alternating cache-warm three dependent repeats, one allocated GPU, CPU source reads and mock assessment. Initialization and input generation are excluded. Source file, model provenance and vectors are not independently authenticated; index internals must remain immutable. No end-to-end agent or production-performance claim.')
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
