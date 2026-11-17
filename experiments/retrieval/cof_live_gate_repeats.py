"""Live encoding, dense candidate, lexical gate and cached source-view read."""
import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
import numpy as np
import torch
from expert_search.cof_data import unique_records
from expert_search.corpus import records
from expert_search.fast_retrieval import FastSearch
from expert_search.resident_candidate_bm25 import ResidentCandidateBM25
from expert_search.shared_owner_gate import SharedOwnerGate
from expert_search.gpu_owned_vector_index import GpuOwnedVectorIndex
from expert_search.owned_vector_index import ExactOwnedVectorIndex
from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import paired_interval


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main(output):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Require allocated CUDA compute node')
    torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    provenance_path=Path('artifacts/cof-gpu-public-input-v1/input-provenance.json')
    provenance=json.loads(provenance_path.read_text())
    for path,digest in provenance['input_sha256'].items():
        if sha(path)!=digest:raise ValueError('Changed frozen input')
    root=Path('artifacts/public/cof/SIGIR');corpus=root/'unique-profile-corpus'
    papers=unique_records([r for _,r in records(root/'SIGIR_papers_test.json')],'paper')
    reviewers=unique_records([r for _,r in records(root/'SIGIR_reviewers_test.json')],'reviewer')
    qids=provenance['query_ids'][:30];pids=sorted({p for r in reviewers.values() for p in r['papers']});owners={p:[] for p in pids}
    for eid,r in reviewers.items():
        for pid in set(r['papers']):owners[pid].append(eid)
    evidence={r['id']:r for _,r in records(corpus/'evidence.jsonl')}
    def paper_text(pid):return (papers[pid].get('title','')+'\n'+papers[pid].get('abstract','')).strip()
    if set(evidence)!=set(pids) or any(evidence[p]['text']!=paper_text(p) for p in pids):raise ValueError('Source mismatch')
    model_path=Path('artifacts/models/qwen3-embedding-0.6b');model_spec=json.loads((model_path/'checkpoint-provenance.json').read_text())
    vector_root=root/'qwen-embedding-torch-v1';protocol=json.loads((vector_root/'protocol.json').read_text())
    if model_spec!=protocol['checkpoint']:raise ValueError('Model metadata mismatch')
    for name,digest in model_spec['files'].items():
        if sha(model_path/name)!=digest:raise ValueError('Changed model')
    vectors=np.load(vector_root/'vectors.npy',mmap_mode='r',allow_pickle=False)
    if vectors.shape[0]!=len(pids)+len(provenance['query_ids']) or vectors.shape[1]!=1024:raise ValueError('Changed vectors')
    lexical=FastSearch(corpus,corpus/'fast.sqlite');resident=ResidentCandidateBM25(corpus/'fast.sqlite',adaptive_intersection=True)
    gate=SharedOwnerGate(resident,{e:r['papers'] for e,r in reviewers.items()})
    index=GpuOwnedVectorIndex(ExactOwnedVectorIndex(vectors[:len(pids)],pids,[owners[p] for p in pids]))
    start=time.perf_counter();model=QwenTorchEmbedding(max_length=512,batch_size=1);model_init=time.perf_counter()-start
    model.encode([paper_text(qids[0])],query=True);index.search_vector(vectors[len(pids)],k=1)
    def run(query,arm):
        started=time.perf_counter();encoded=model.encode([query],query=True)[0]
        hits=index.search_vector(encoded,k=1)['hits']
        if len(hits)!=1:raise ValueError('Missing candidate')
        hit=hits[0];eid=hit['id']
        if arm=='bounded':accepted=gate.contains(query,eid,50)['accepted']
        else:
            values=lexical.document_scores(query)
            per={e:max((values.get(p,0.) for p in set(r['papers'])),default=0.) for e,r in reviewers.items()}
            ranked=sorted(per,key=lambda e:(-per[e],e));accepted=eid in ranked[:50] and per[eid]>0
        selected=[]
        if accepted:
            for witness in hit['evidence']:
                sid=witness['id'];record=evidence.get(sid)
                if record is None or sid not in set(reviewers[eid]['papers']):raise ValueError('Invalid owner/source')
                selected.append({'id':sid,'sha256':hashlib.sha256(record['text'][:1200].encode()).hexdigest()})
        return {'candidate':eid,'accepted':accepted,'sources':selected,'seconds':time.perf_counter()-started}
    # Warm all stages without retaining measurements.
    run(paper_text(qids[0]),'full');run(paper_text(qids[0]),'bounded')
    rows=[]
    for i,qid in enumerate(qids):
        query=paper_text(qid);pairs=[]
        for repeat in range(5):
            results={}
            for arm in (('full','bounded') if (i+repeat)%2==0 else ('bounded','full')):
                results[arm]=run(query,arm)
            x,y=results['full'],results['bounded']
            pairs.append({'seconds':{arm:results[arm]['seconds'] for arm in results},
                          'same_candidate':x['candidate']==y['candidate'],'same_decision':x['accepted']==y['accepted'],
                          'same_sources':x['sources']==y['sources'],'accepted':x['accepted']})
        rows.append({'query_id':qid,'pairs':pairs})
    flattened=[pair for row in rows for pair in row['pairs']]
    per_query={a:[statistics.mean(p['seconds'][a] for p in row['pairs']) for row in rows] for a in ('full','bounded')}
    def percentile(values,p):
        values=sorted(values);return values[min(len(values)-1,int((len(values)-1)*p))]
    report={'queries':len(rows),'pairs':len(flattened),'repeats_per_query':5,
            'same_candidate':sum(r['same_candidate'] for r in flattened),
            'same_decision':sum(r['same_decision'] for r in flattened),'same_sources':sum(r['same_sources'] for r in flattened),
            'accepted_pairs':sum(r['accepted'] for r in flattened),
            'mean_seconds':{a:statistics.mean(per_query[a]) for a in per_query},
            'p50_p95_seconds':{a:[percentile([p['seconds'][a] for p in flattened],q) for q in (.5,.95)] for a in per_query},
            'bounded_minus_full_query_mean_seconds':paired_interval([b-f for b,f in zip(per_query['bounded'],per_query['full'])]),
            'model_init_seconds':model_init,'resident_init_seconds':resident.initialization_seconds,'bound_init_seconds':gate.initialization_seconds,
            'gpu':torch.cuda.get_device_name(0),'slurm_job_id':os.environ['SLURM_JOB_ID'],
            'inputs_sha256':{'provenance':sha(provenance_path),'script':sha(__file__),'gate':sha('src/expert_search/shared_owner_gate.py')},
            'scope':'First 30 reused CoF SIGIR queries, five alternating pairs after warmup. Repeats on the same query are dependent; query bootstrap uses their means. P50/P95 describe 150 sequential observations, not independent service requests. Each arm independently encodes, runs dense scoring, gates one candidate and reads cached source text if accepted. Structural source hashes do not prove semantic support. Initialization, index creation, explanation, feedback, concurrency, queueing and serving excluded. Not full-agent accuracy, non-inferiority or enterprise fit.'}
    output.mkdir(parents=True,exist_ok=False);(output/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
    report['rows_sha256']=sha(output/'rows.json');(output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.output)
