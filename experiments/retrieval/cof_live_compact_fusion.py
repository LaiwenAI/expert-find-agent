"""Compare lexical and compact cross-encoder fusion on a fixed expert pool."""
import hashlib
import math
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.decision.merit_open_decision_pilot import sha
from expert_search.cof_benchmark import load
from experiments.evidencebench_pilot import bm25
from expert_search.qwen_reranker import PREFIX, SUFFIX, TASK, pack_tokens
from expert_search.gpu_owned_vector_index import GpuOwnedVectorIndex
from expert_search.owned_vector_index import ExactOwnedVectorIndex
from expert_search.qwen_embedding_torch import QwenTorchEmbedding
from expert_search.statistics import paired_interval

def main():
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Requires Slurm CUDA allocation')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    manifest = Path('artifacts/cof-gpu-public-input-v1/input-provenance.json')
    spec = json.loads(manifest.read_text())
    for file, expected in spec['input_sha256'].items():
        if sha(Path(file)) != expected:
            raise ValueError(f'Changed input: {file}')
    root, papers, reviewers, labels = load('SIGIR')
    query_ids = spec['query_ids']
    ownership = {}
    for expert, profile in reviewers.items():
        for paper in set(profile['papers']):
            ownership.setdefault(paper, []).append(expert)
    paper_ids = sorted(ownership)
    vectors = np.load(root/'qwen-embedding-torch-v1/vectors.npy', allow_pickle=False)
    if vectors.shape != (len(paper_ids)+len(query_ids), 1024):
        raise ValueError('Changed vector layout')
    embed_dir = Path('artifacts/models/qwen3-embedding-0.6b')
    embed_manifest = embed_dir/'checkpoint-provenance.json'
    if json.loads(embed_manifest.read_text()) != json.loads(
        (root/'qwen-embedding-torch-v1/protocol.json').read_text())['checkpoint']:
        raise ValueError('Embedding model differs from index')
    model_dir = Path('artifacts/models/qwen3-reranker-0.6b')
    model_manifest = model_dir/'checkpoint-provenance.json'
    for directory, provenance in ((embed_dir, embed_manifest), (model_dir, model_manifest)):
        for file, expected in json.loads(provenance.read_text())['files'].items():
            if sha(directory/file) != expected:
                raise ValueError(f'Changed model file: {file}')
    label_hashes = {task: sha(root/f'SIGIR_queries_test_{task}.json') for task in labels}
    for task in labels:
        if any(set(labels[task][q]) != set(reviewers) for q in query_ids):
            raise ValueError('Incomplete reviewer grades')

    started = time.perf_counter()
    index = GpuOwnedVectorIndex(ExactOwnedVectorIndex(vectors[:len(paper_ids)], paper_ids,
                                                     [ownership[p] for p in paper_ids]))
    encoder = QwenTorchEmbedding(max_length=512, batch_size=1)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, dtype=torch.bfloat16).to('cuda').eval()
    init_seconds = time.perf_counter() - started
    digits = [tokenizer.encode(label, add_special_tokens=False) for label in ('yes', 'no')]
    if any(len(token) != 1 for token in digits):
        raise ValueError('Expected single-token labels')
    token_ids = [token[0] for token in digits]

    def paper_text(paper):
        return (papers[paper].get('title', '') + '\n' + papers[paper].get('abstract', '')).strip()

    prefix = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix = tokenizer.encode(SUFFIX, add_special_tokens=False)

    def run(query_id, arm):
        began = time.perf_counter()
        query = paper_text(query_id)
        encoded = encoder.encode([query], query=True)[0]
        candidates = index.search_vector(encoded, k=10)['hits']
        ids = [candidate['id'] for candidate in candidates]
        source_ids = [candidate['evidence'][0]['id'] for candidate in candidates]
        if any(expert not in ownership[source] for expert, source in zip(ids, source_ids)):
            raise ValueError('Unexpected source ownership')
        views = [paper_text(source)[:1200] for source in source_ids]
        order = list(range(len(ids)))
        truncations = 0
        if arm == 'lexical_fusion':
            second_order = bm25(query, views)
        elif arm == 'cross_fusion':
            sequences = []
            for view in views:
                body = tokenizer.encode(f'<Instruct>: {TASK}\n<Query>: {query}\n<Document>: {view}',
                                        add_special_tokens=False)
                packed, raw = pack_tokens(prefix, body, suffix, 1024)
                truncations += raw > len(packed)
                sequences.append({'input_ids': packed})
            inputs = tokenizer.pad(sequences, padding=True, return_tensors='pt').to('cuda')
            with torch.inference_mode():
                logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, token_ids].float().cpu()
            scores = (logits[:, 0] - logits[:, 1]).tolist()
            second_order = sorted(order, key=lambda i: (-scores[i], i))
        if arm != 'dense':
            second_rank = {i: rank for rank, i in enumerate(second_order, start=1)}
            order.sort(key=lambda i: -(1/(61+i) + 1/(60+second_rank[i])))
        hits = [{'expert':ids[i], 'source':source_ids[i], 'view_sha256':hashlib.sha256(views[i].encode()).hexdigest()}
                for i in order[:5]]
        return {'hits':hits, 'pool':ids, 'sources':source_ids, 'seconds':time.perf_counter()-began,
                'truncations':truncations}

    methods = ('dense','lexical_fusion','cross_fusion')
    for arm in methods:
        run(query_ids[0], arm)
    rows = []
    for i, query_id in enumerate(query_ids):
        offset = i % len(methods)
        arms = methods[offset:]+methods[:offset]
        results = {arm:run(query_id, arm) for arm in arms}
        for result in results.values():
            if any(result[key] != results['dense'][key] for key in ('pool','sources')):
                raise ValueError('Matched arms produced different candidates or evidence')
            result['precision5'] = {task:sum(labels[task][query_id][h['expert']] > 0 for h in result['hits'])/5
                                    for task in labels}
        rows.append({'query_id':query_id, 'arms':results})

    out = Path('artifacts/cof-live-compact-fusion-v1')
    out.mkdir(exist_ok=False)
    (out/'rows.json').write_text(json.dumps(rows, indent=2)+'\n')
    report = {'queries':len(rows), 'job_id':os.environ['SLURM_JOB_ID'], 'init_seconds':init_seconds,
              'rows_sha256':sha(out/'rows.json'), 'input_sha256':sha(manifest), 'label_sha256':label_hashes,
              'script_sha256':sha(Path(__file__)), 'models_sha256':[sha(embed_manifest),sha(model_manifest)],
              'results':{}, 'paired_delta':{},
              'scope':'Reused CoF SIGIR reviewer self-ratings. Matched sources and candidates. Lexical fusion runs candidate-local BM25 and cross fusion uses Qwen3-Reranker-0.6B. Both score the same bounded source views. Both fuse ranks with dense using equal-weight reciprocal rank fusion k=60. Warm sequential timing excludes initialization, index build, storage and serving.'}
    for arm in methods:
        seconds = sorted(r['arms'][arm]['seconds'] for r in rows)
        report['results'][arm] = {
            'mean_ms':1000*statistics.mean(seconds),
            'p95_ms':1000*seconds[math.ceil(.95*len(seconds))-1],
            'truncated_prompts':sum(r['arms'][arm]['truncations'] for r in rows),
            'precision5':{task:statistics.mean(r['arms'][arm]['precision5'][task] for r in rows) for task in labels},
        }
        if arm != 'dense':
            report['paired_delta'][arm] = {task: paired_interval([
                r['arms'][arm]['precision5'][task] - r['arms']['dense']['precision5'][task] for r in rows])
                for task in labels}
    (out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
