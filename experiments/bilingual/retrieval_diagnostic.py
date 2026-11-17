"""Bilingual rank stability plus exact/HNSW fidelity on Slurm compute only."""
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
from expert_search.owned_vector_index import ExactOwnedVectorIndex, OwnedVectorIndex

ROOT = Path('artifacts/bilingual-retrieval-v1')
TASK = 'Find researchers whose documented research background could help address the technical requirement.'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class Effort:
    def __init__(self, index, floor):
        self.index, self.floor, self.ntotal = index, floor, index.ntotal

    def search(self, query, k):
        self.index.hnsw.efSearch = max(self.floor, k)
        return self.index.search(query, k)


def run():
    import faiss
    import torch
    from expert_search.qwen_embedding_torch import QwenTorchEmbedding
    parent = Path('artifacts/global-expert-branch-benchmark-v1')
    spec = json.loads((parent / 'protocol.json').read_text())
    old_summary = json.loads((parent / 'summary.json').read_text())
    selection = json.loads((ROOT / 'selection.json').read_text())
    queries = [json.loads(s) for s in (ROOT / 'queries.jsonl').read_text().splitlines()]
    source = Path('artifacts/global-expert-branches-v1/branches.jsonl')
    vp = parent / 'vectors.npy'
    if sha(ROOT / 'queries.jsonl') != selection['queries_sha256'] or sha(source) != spec['roots_sha256'] or sha(vp) != old_summary['vectors_sha256']:
        raise ValueError('Changed query/source/vector inventory')
    for name, expected in spec['checkpoint']['files'].items():
        if sha(Path('artifacts/models/qwen3-embedding-0.6b') / name) != expected:
            raise ValueError('Changed official checkpoint')
    protocol = dict(selection_sha256=sha(ROOT / 'selection.json'), source_sha256=sha(source), source_vectors_sha256=sha(vp), checkpoint=spec['checkpoint'], encoder=dict(task=TASK, max_length=4096, batch_size=16, backend='torch'), methods=['exact', 'ef64', 'ef512'], k=10, cpu_threads=1, timing='One alternating-order timed call per method/language/query after warmup; cached vectors, excludes encoding/index/source loading. GPU encoding is batch-throughput wall time, not individual-request latency.', code_sha256={str(p): sha(p) for p in [Path(__file__), Path('src/expert_search/owned_vector_index.py'), Path('src/expert_search/source_routing.py'), Path('src/expert_search/qwen_embedding_torch.py'), Path('src/expert_search/qwen_embedding.py')]}, scope='Construction diagnostic on synthetic intent pairs, not independently labeled full-query benchmark. Frozen-score fidelity and language stability are not expert relevance, semantic equivalence or end-to-end agent accuracy.')
    marker = ROOT / 'protocol.json'
    if marker.exists() and json.loads(marker.read_text()) != protocol:
        raise ValueError('Changed frozen bilingual diagnostic')
    marker.write_text(json.dumps(protocol, indent=2) + '\n')
    npz = ROOT / 'query-vectors.npz'
    if not npz.exists():
        load_start = time.perf_counter()
        model = QwenTorchEmbedding(max_length=4096, batch_size=16)
        load_seconds = time.perf_counter() - load_start
        texts = ['Instruct: ' + TASK + '\nQuery:' + row[language] for row in queries for language in ('zh', 'en')]
        lengths = [len(model.tokenizer(text, truncation=False)['input_ids']) for text in texts]
        if max(lengths) > 4096:
            raise ValueError('Query would be truncated')
        model.encode(texts[:2])
        start = time.perf_counter()
        vectors = model.encode(texts)
        seconds = time.perf_counter() - start
        np.savez(npz, vectors=vectors, ids=np.asarray([q['id'] for q in queries]), languages=np.asarray(['zh', 'en']), token_lengths=np.asarray(lengths))
        (ROOT / 'encoding.json').write_text(json.dumps(dict(model_load_seconds=load_seconds, batch_encoding_wall_seconds=seconds, pairs=len(queries), vectors=len(texts), max_query_tokens=max(lengths), truncated=0, query_vectors_sha256=sha(npz), protocol_sha256=sha(marker), node=platform.node(), gpu=torch.cuda.get_device_name(), torch=torch.__version__), indent=2) + '\n')
        del model
        torch.cuda.empty_cache()
    encoding = json.loads((ROOT / 'encoding.json').read_text())
    if sha(npz) != encoding['query_vectors_sha256'] or sha(marker) != encoding['protocol_sha256']:
        raise ValueError('Changed cached query vectors')
    with np.load(npz, allow_pickle=False) as cached:
        vectors = cached['vectors']
        if list(cached['ids']) != [r['id'] for r in queries] or vectors.shape != (2000, 1024):
            raise ValueError('Bad bilingual vector IDs/shape')
    faiss.omp_set_num_threads(1)
    records = [json.loads(s) for s in source.read_text().splitlines()]
    matrix = np.load(vp, mmap_mode='r', allow_pickle=False)
    ids, owners = [r['id'] for r in records], [[r['expert_id']] for r in records]
    start = time.perf_counter()
    exact = ExactOwnedVectorIndex(matrix, ids, owners)
    graph = OwnedVectorIndex(matrix, ids, owners)
    indexes = dict(exact=exact, ef64=OwnedVectorIndex(matrix, ids, owners, index=Effort(graph.index, 64)), ef512=OwnedVectorIndex(matrix, ids, owners, index=Effort(graph.index, 512)))
    build_seconds = time.perf_counter() - start
    for index in indexes.values():
        index.search_vector(vectors[0], k=10)
    rows = []
    output = ROOT / 'evaluation.jsonl'
    with output.open('w') as stream:
        for qi, query in enumerate(queries):
            result = dict(id=query['id'], parent_id=query['parent_id'], group_id=query['group_id'], intent_type=query['intent_type'], status=query['status'], query_cosine=float(vectors[2*qi] @ vectors[2*qi+1]), languages={})
            for li, language in enumerate(('zh', 'en')):
                methods = {}
                order = list(indexes) if (qi + li) % 2 == 0 else list(indexes)[::-1]
                for method in order:
                    start = time.perf_counter()
                    found = indexes[method].search_vector(vectors[2*qi+li], k=10)
                    elapsed = time.perf_counter() - start
                    methods[method] = dict(seconds=elapsed, ranking=[h['id'] for h in found['hits']], top_source_ids=[h['evidence'][0]['id'] for h in found['hits']], source_rows=found['source_rows'], branch=found['branch'])
                truth = methods['exact']['ranking']
                for method, value in methods.items():
                    value.update(ordered_top10_equal=value['ranking'] == truth, exact_top10_retained=len(set(value['ranking']) & set(truth)) / len(truth))
                result['languages'][language] = methods
            result['language_overlap'] = {m: len(set(result['languages']['zh'][m]['ranking']) & set(result['languages']['en'][m]['ranking'])) / 10 for m in indexes}
            result['language_ordered_equal'] = {m: result['languages']['zh'][m]['ranking'] == result['languages']['en'][m]['ranking'] for m in indexes}
            stream.write(json.dumps(result) + '\n')
            stream.flush()
            rows.append(result)
            if (qi+1) % 50 == 0:
                print('Completed paired queries', qi+1, flush=True)
    def aggregate(subset):
        return dict(pairs=len(subset), mean_query_cosine=float(np.mean([r['query_cosine'] for r in subset])), methods={m: dict(mean_language_top10_overlap=float(np.mean([r['language_overlap'][m] for r in subset])), language_ordered_equal=sum(r['language_ordered_equal'][m] for r in subset), mean_query_seconds=float(np.mean([r['languages'][l][m]['seconds'] for r in subset for l in ('zh', 'en')])), ordered_fidelity=sum(r['languages'][l][m]['ordered_top10_equal'] for r in subset for l in ('zh', 'en')), mean_exact_top10_retained=float(np.mean([r['languages'][l][m]['exact_top10_retained'] for r in subset for l in ('zh', 'en')]))) for m in indexes})
    summary = dict(all=aggregate(rows), by_status={s: aggregate([r for r in rows if r['status']==s]) for s in sorted({r['status'] for r in rows})}, by_intent={k: aggregate([r for r in rows if r['intent_type']==k]) for k in sorted({r['intent_type'] for r in rows})}, sources=len(records), experts=len(exact.owners), index_build_and_copies_seconds=build_seconds, encoding=encoding, protocol_sha256=sha(marker), evaluation_sha256=sha(output), node=platform.node(), faiss=faiss.__version__, scope=protocol['scope'])
    (ROOT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary['all'], indent=2))


if __name__ == '__main__':
    run()
