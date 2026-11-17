"""Measure live single-query encoding and source retrieval on CoF SIGIR."""
import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from expert_search.candidate_evidence_pipeline import verify_evidence_documents
from expert_search.cof_benchmark import precision_metrics
from expert_search.cof_data import unique_records
from expert_search.corpus import records
from expert_search.fast_retrieval import FastSearch
from expert_search.gpu_owned_vector_index import GpuOwnedVectorIndex
from expert_search.owned_vector_index import ExactOwnedVectorIndex
from expert_search.qwen_embedding import TASK
from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import paired_interval


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(output):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Run on an allocated CUDA compute node')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    provenance_path = Path('artifacts/cof-gpu-public-input-v1/input-provenance.json')
    provenance = json.loads(provenance_path.read_text())
    for path, digest in provenance['input_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Changed frozen input: '+path)
    root = Path('artifacts/public/cof/SIGIR')
    papers = unique_records([r for _, r in records(root/'SIGIR_papers_test.json')], 'paper')
    reviewers = unique_records([r for _, r in records(root/'SIGIR_reviewers_test.json')], 'reviewer')
    labels = {task:{r['query_id']:r['score'] for _, r in records(root/f'SIGIR_queries_test_{task}.json')} for task in ('soft','hard')}
    qids = sorted(labels['soft']); pids = sorted({pid for r in reviewers.values() for pid in r['papers']})
    if qids != provenance['query_ids'] or set(qids) != set(labels['hard']) or any(set(labels[t][q]) != set(reviewers) for t in labels for q in qids):
        raise ValueError('Changed or incomplete query labels')
    owners = {pid:[] for pid in pids}
    for eid,r in reviewers.items():
        if not r['papers']: raise ValueError('Reviewer lacks a source')
        for pid in set(r['papers']): owners[pid].append(eid)
    def text(pid): return (papers[pid].get('title','')+'\n'+papers[pid].get('abstract','')).strip()
    model_path = Path('artifacts/models/qwen3-embedding-0.6b')
    model_provenance = json.loads((model_path/'checkpoint-provenance.json').read_text())
    previous = root/'qwen-embedding-torch-v1'
    previous_protocol = json.loads((previous/'protocol.json').read_text())
    if model_provenance != previous_protocol['checkpoint']: raise ValueError('Checkpoint provenance mismatch')
    for name,digest in model_provenance['files'].items():
        if sha(model_path/name) != digest: raise ValueError('Changed checkpoint file: '+name)
    items = [dict(id='doc:'+pid,text=text(pid),query=False) for pid in pids]+[dict(id='query:'+qid,text=text(qid),query=True) for qid in qids]
    if hashlib.sha256(json.dumps(items,sort_keys=True).encode()).hexdigest() != previous_protocol['input_sha256']:
        raise ValueError('Changed vector order')
    corpus = root/'unique-profile-corpus'
    evidence = {r['id']:r for _,r in records(corpus/'evidence.jsonl')}
    if set(evidence) != set(pids) or any(evidence[p]['expert_id'] != p or evidence[p]['text'] != text(p) for p in pids):
        raise ValueError('Lexical records differ from official sources')
    lexical = FastSearch(corpus,corpus/'fast.sqlite'); verify_evidence_documents(corpus,lexical)
    matrix = np.load(previous/'vectors.npy',mmap_mode='r')
    if matrix.shape != (len(items),1024) or matrix.dtype != np.float32: raise ValueError('Invalid frozen vectors')
    index = GpuOwnedVectorIndex(ExactOwnedVectorIndex(matrix[:len(pids)],pids,[owners[p] for p in pids]))
    start=time.perf_counter(); model=QwenTorchEmbedding(max_length=512,batch_size=1); init=time.perf_counter()-start
    def lex(query):
        values=lexical.document_scores(query)
        scores={eid:max((values.get(p,0.) for p in set(r['papers'])),default=0.) for eid,r in reviewers.items()}
        ranking=sorted(reviewers,key=lambda eid:(-scores[eid],eid))[:10]
        return [dict(id=eid,evidence=[dict(id=p) for p in sorted(set(reviewers[eid]['papers']),key=lambda p:(-values.get(p,0.),p))[:3]]) for eid in ranking]
    # Warm model, scorer and lexical code; no batching or concurrent requests.
    model.encode([text(qids[0])],query=True);index.search_vector(matrix[len(pids)],k=10);lex(text(qids[0]))
    rows=[]
    for i,qid in enumerate(qids):
        query=text(qid)
        token_length=len(model.tokenizer.encode('Instruct: '+TASK+'\nQuery:'+query,add_special_tokens=True))
        def dense():
            start=time.perf_counter();q=model.encode([query],query=True)[0]; encoding=time.perf_counter()-start
            start=time.perf_counter();hits=index.search_vector(q,k=10)['hits'];scoring=time.perf_counter()-start
            return hits,encoding,scoring,q
        times={};hits={}
        for method in (('dense','lexical') if i%2==0 else ('lexical','dense')):
            torch.cuda.synchronize();start=time.perf_counter()
            if method=='dense':hits[method],encoding,scoring,q=dense()
            else:hits[method]=lex(query)
            times[method]=time.perf_counter()-start
        rankings={m:[h['id'] for h in hits[m]] for m in hits}
        old=json.loads((previous/(qid+'.json')).read_text())['results']['soft']['max']['ranking'][:10]
        rows.append(dict(query_id=qid,token_length=token_length,truncated=token_length>512,timings=times,encoding_seconds=encoding,scoring_seconds=scoring,
            rankings=rankings,old_order_equal=rankings['dense']==old,old_top10_overlap=len(set(rankings['dense'])&set(old))/10,
            old_query_cosine=float(q@matrix[len(pids)+i]),metrics={task:{m:precision_metrics(rankings[m],labels[task][qid]) for m in rankings} for task in labels}))
    summary=dict(queries=len(rows),truncated=sum(r['truncated'] for r in rows),max_token_length=max(r['token_length'] for r in rows),old_order_equal=sum(r['old_order_equal'] for r in rows),
        mean_old_top10_overlap=statistics.mean(r['old_top10_overlap'] for r in rows),mean_old_query_cosine=statistics.mean(r['old_query_cosine'] for r in rows),
        median_seconds={m:statistics.median(r['timings'][m] for r in rows) for m in ('dense','lexical')},
        median_encoding_seconds=statistics.median(r['encoding_seconds'] for r in rows),median_scoring_seconds=statistics.median(r['scoring_seconds'] for r in rows),
        metrics={task:{m:{k:statistics.mean(r['metrics'][task][m][k] for r in rows) for k in ('p@5','ndcg@10')} for m in ('dense','lexical')} for task in labels},
        dense_minus_lexical={task:{k:paired_interval([r['metrics'][task]['dense'][k]-r['metrics'][task]['lexical'][k] for r in rows]) for k in ('p@5','ndcg@10')} for task in labels})
    protocol=dict(dataset='CoF SIGIR',queries=len(qids),reviewers=len(reviewers),profile_papers=len(pids),gpu=torch.cuda.get_device_name(0),torch_version=torch.__version__,slurm_job_id=os.environ['SLURM_JOB_ID'],
        script_sha256=sha(__file__),input_manifest_sha256=sha(provenance_path),encoder_sha256=sha('src/expert_search/qwen_embedding_torch.py'),gpu_index_sha256=sha('src/expert_search/gpu_owned_vector_index.py'),initialization_seconds=init,
        timing='One warmed single-query observation per query, alternating order, one CPU thread; dense includes tokenization, host/device transfers, encoding and source/expert aggregation; lexical includes tokenization, ranking and top-three source selection. Excludes initialization, source-vector creation, reasoning, I/O, serving and queueing.',
        limitation='Reused reviewer labels are not enterprise expert-fit, source-entailment or individual-contribution judgments. No model training; contamination unknown. Single-node component test, not production end-to-end agent latency.')
    output.mkdir(parents=True,exist_ok=False)
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');(output/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
    summary.update(protocol_sha256=sha(output/'protocol.json'),rows_sha256=sha(output/'rows.json'))
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);main(parser.parse_args().output)
