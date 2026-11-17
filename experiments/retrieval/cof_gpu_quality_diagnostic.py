"""Compare full-source GPU scoring with a lexical baseline on frozen CoF labels."""
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
    for path, expected in provenance['input_sha256'].items():
        if sha(path) != expected:
            raise ValueError('Changed frozen input: ' + path)
    root = Path('artifacts/public/cof/SIGIR')
    papers = unique_records([r for _, r in records(root/'SIGIR_papers_test.json')], 'paper')
    reviewers = unique_records([r for _, r in records(root/'SIGIR_reviewers_test.json')], 'reviewer')
    labels = {task: {r['query_id']: r['score'] for _, r in records(root/f'SIGIR_queries_test_{task}.json')} for task in ('soft', 'hard')}
    qids = sorted(labels['soft'])
    if qids != provenance['query_ids'] or set(qids) != set(labels['hard']):
        raise ValueError('Changed query universe')
    if any(set(labels[t][q]) != set(reviewers) for t in labels for q in qids):
        raise ValueError('Incomplete judged reviewer universe')
    pids = sorted({pid for r in reviewers.values() for pid in r['papers']})
    owners = {pid: [] for pid in pids}
    for eid, reviewer in reviewers.items():
        if not reviewer['papers']:
            raise ValueError('Reviewer has no source')
        for pid in set(reviewer['papers']):
            owners[pid].append(eid)
    def text(pid):
        return (papers[pid].get('title', '')+'\n'+papers[pid].get('abstract', '')).strip()
    items = [dict(id='doc:'+pid, text=text(pid), query=False) for pid in pids]
    items += [dict(id='query:'+qid, text=text(qid), query=True) for qid in qids]
    model_root = root/'qwen-embedding-torch-v1'
    model_protocol = json.loads((model_root/'protocol.json').read_text())
    if hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest() != model_protocol['input_sha256']:
        raise ValueError('Cached vector order does not match source data')
    corpus = root/'unique-profile-corpus'
    evidence = {r['id']: r for _, r in records(corpus/'evidence.jsonl')}
    if set(evidence) != set(pids) or any(evidence[pid]['expert_id'] != pid or evidence[pid]['text'] != text(pid) for pid in pids):
        raise ValueError('Lexical source records differ from official papers')
    lexical = FastSearch(corpus, corpus/'fast.sqlite')
    verify_evidence_documents(corpus, lexical)
    vectors = np.load(model_root/'vectors.npy', mmap_mode='r')
    if vectors.shape != (len(items), 1024) or vectors.dtype != np.float32:
        raise ValueError('Unexpected frozen vector matrix')
    cpu = ExactOwnedVectorIndex(vectors[:len(pids)], pids, [owners[p] for p in pids])
    gpu = GpuOwnedVectorIndex(cpu)
    def lex(query):
        scores = lexical.document_scores(query)
        reviewer_scores = {eid: max((scores.get(pid, 0.) for pid in set(r['papers'])), default=0.) for eid, r in reviewers.items()}
        ranking = sorted(reviewers, key=lambda eid: (-reviewer_scores[eid], eid))[:10]
        return [dict(id=eid, evidence=[dict(id=pid) for pid in sorted(set(reviewers[eid]['papers']), key=lambda pid: (-scores.get(pid, 0.), pid))[:3]]) for eid in ranking]
    # Warm each path; query vectors were encoded before timing.
    lex(text(qids[0])); cpu.search_vector(vectors[len(pids)], k=10); gpu.search_vector(vectors[len(pids)], k=10)
    rows = []
    for i, qid in enumerate(qids):
        operations = {'lexical': lambda: lex(text(qid)), 'cpu': lambda: cpu.search_vector(vectors[len(pids)+i], k=10)['hits'], 'gpu': lambda: gpu.search_vector(vectors[len(pids)+i], k=10)['hits']}
        methods = list(operations); methods = methods[i % 3:]+methods[:i % 3]
        hits = {}; timings = {}
        for method in methods:
            torch.cuda.synchronize(); start = time.perf_counter(); hits[method] = operations[method](); timings[method] = time.perf_counter()-start
        rankings = {m: [h['id'] for h in hits[m]] for m in hits}
        historical = json.loads((model_root/(qid+'.json')).read_text())['results']['soft']['max']['ranking'][:10]
        metrics = {task: {m: precision_metrics(rankings[m], labels[task][qid]) for m in rankings} for task in labels}
        rows.append(dict(query_id=qid, timings=timings, rankings=rankings, metrics=metrics,
                         historical_cpu_equal=historical == rankings['cpu'],
                         cpu_gpu_equal=rankings['cpu'] == rankings['gpu'],
                         cpu_gpu_first_source_equal=all(a['evidence'][0]['id'] == b['evidence'][0]['id'] for a, b in zip(hits['cpu'], hits['gpu'])),
                         cpu_gpu_three_sources_equal=all([x['id'] for x in a['evidence']] == [x['id'] for x in b['evidence']] for a, b in zip(hits['cpu'], hits['gpu']))))
    protocol = dict(dataset='CoF SIGIR', queries=len(qids), reviewers=len(reviewers), unique_profile_papers=len(pids),
                    input_manifest_sha256=sha(provenance_path), script_sha256=sha(__file__), gpu_index_sha256=sha('src/expert_search/gpu_owned_vector_index.py'), cpu_index_sha256=sha('src/expert_search/owned_vector_index.py'),
                    slurm_job_id=os.environ['SLURM_JOB_ID'], gpu=torch.cuda.get_device_name(0), torch_version=torch.__version__,
                    timing='Single thread, nonconcurrent, warm paths, one rotating-order pass. GPU timing includes host/device transfers and CPU aggregation; excludes query encoding, model loading, index construction and reasoning. Lexical timing includes tokenization and ranking.',
                    limitation='Original public reviewer labels are not industrial expert-fit labels, human source-entailment judgments, or evidence of individual contribution. Cached frozen query embeddings exclude encoder latency. TF32 disabled; CPU/GPU floating-point equivalence is not guaranteed.')
    summary = dict(queries=len(rows), cpu_gpu_ordered_equal=sum(r['cpu_gpu_equal'] for r in rows), cpu_gpu_first_source_equal=sum(r['cpu_gpu_first_source_equal'] for r in rows), cpu_gpu_three_sources_equal=sum(r['cpu_gpu_three_sources_equal'] for r in rows), historical_cpu_equal=sum(r['historical_cpu_equal'] for r in rows),
                   median_seconds={m: statistics.median(r['timings'][m] for r in rows) for m in ('lexical','cpu','gpu')},
                   metrics={task:{m:{k:statistics.mean(r['metrics'][task][m][k] for r in rows) for k in ('p@5','ndcg@10')} for m in ('lexical','cpu','gpu')} for task in labels},
                   gpu_minus_lexical={task:{k:paired_interval([r['metrics'][task]['gpu'][k]-r['metrics'][task]['lexical'][k] for r in rows]) for k in ('p@5','ndcg@10')} for task in labels})
    output.mkdir(parents=True, exist_ok=False)
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    (output/'rows.json').write_text(json.dumps(rows, indent=2)+'\n')
    summary.update(protocol_sha256=sha(output/'protocol.json'), rows_sha256=sha(output/'rows.json'))
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
