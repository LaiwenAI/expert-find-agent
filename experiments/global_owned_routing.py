"""Full enterprise corpus retrieval fidelity and CPU routing cost, not fit quality."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from expert_search.owned_vector_index import OwnedVectorIndex,ExactOwnedVectorIndex

ROOT=Path('artifacts/global-owned-routing-v1')
QUERIES=Path('artifacts/demand-first-paper-retrieval-v1/queries.jsonl')
TASK='Find researchers whose documented research background could help address the technical requirement.'


def run(stage):
    ROOT.mkdir(exist_ok=True)
    parent=Path('artifacts/global-expert-branch-benchmark-v1')
    spec=json.loads((parent/'protocol.json').read_text());summary=json.loads((parent/'summary.json').read_text())
    queries=[json.loads(line) for line in QUERIES.read_text().splitlines()]
    protocol=dict(queries_sha256=hashlib.sha256(QUERIES.read_bytes()).hexdigest(),source_sha256=spec['roots_sha256'],vectors_sha256=summary['vectors_sha256'],
                  code_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['experiments/global_owned_routing.py','src/expert_search/owned_vector_index.py','src/expert_search/source_routing.py']},
                  methods=['exact_full','owned_ann'],k=10,M=32,efConstruction=128,initial=32,limit=1024,target_experts=10,threads=1,repeats=5,
                  encoder=dict(task=TASK,backend='torch',max_length=4096,batch_size=8),
                  scope='All20 reused fixed original enterprise demands, all owned paper/profile sources in frozen full matrix, no labels or known-positive injections. Fidelity to exact max-source FP32 ordering only, not suitability/precision. CPU paths alternate per repeat, warmup excluded. Query cache excludes fresh encoding in CPU timings; GPU encoding separately recorded.')
    marker=ROOT/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=protocol:raise ValueError('Changed global routing protocol')
    marker.write_text(json.dumps(protocol,indent=2))
    if stage=='encode':
        from expert_search.qwen_embedding_torch import QwenTorchEmbedding
        model_root=Path('artifacts/models/qwen3-embedding-0.6b')
        for name,expected in spec['checkpoint']['files'].items():
            with (model_root/name).open('rb') as stream:
                if hashlib.file_digest(stream,'sha256').hexdigest()!=expected:raise ValueError('Changed query encoder checkpoint')
        model=QwenTorchEmbedding(max_length=4096,batch_size=8)
        vectors=[];seconds=[]
        for query in queries:
            start=time.perf_counter();vectors.append(model.encode(['Instruct: '+TASK+'\nQuery:'+query['demand']])[0]);seconds.append(time.perf_counter()-start)
        np.savez(ROOT/'query-vectors.npz',vectors=np.asarray(vectors),seconds=np.asarray(seconds),ids=np.asarray([q['id'] for q in queries]))
        print('Encoded',len(vectors),'queries',flush=True);return
    import faiss
    import platform,resource
    faiss.omp_set_num_threads(1)
    source=Path('artifacts/global-expert-branches-v1/branches.jsonl')
    if hashlib.sha256(source.read_bytes()).hexdigest()!=spec['roots_sha256']:raise ValueError('Changed source inventory')
    with (parent/'vectors.npy').open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=summary['vectors_sha256']:raise ValueError('Changed source vectors')
    records=[json.loads(line) for line in source.read_text().splitlines()]
    matrix=np.load(parent/'vectors.npy',mmap_mode='r',allow_pickle=False)
    if matrix.shape!=(len(records),1024):raise ValueError('Wrong source shape')
    ids=[r['id'] for r in records];owners=[[r['expert_id']] for r in records]
    start=time.perf_counter();exact=ExactOwnedVectorIndex(matrix,ids,owners);exact_build=time.perf_counter()-start
    start=time.perf_counter();ann=OwnedVectorIndex(matrix,ids,owners);ann_build=time.perf_counter()-start
    indexes=dict(exact_full=exact,owned_ann=ann)
    with np.load(ROOT/'query-vectors.npz',allow_pickle=False) as saved:
        qvectors=saved['vectors'];encoding=saved['seconds'];qids=saved['ids']
    if list(qids)!=[q['id'] for q in queries] or qvectors.shape!=(len(queries),1024):raise ValueError('Query inventory mismatch')
    results=[]
    for qi,query in enumerate(queries):
        q=qvectors[qi];timings={m:[] for m in indexes};outputs={}
        for index in indexes.values():index.search_vector(q,k=10)
        for repeat in range(5):
            order=list(indexes) if repeat%2==0 else list(indexes)[::-1]
            for mode in order:
                start=time.perf_counter();outputs[mode]=indexes[mode].search_vector(q,k=10);timings[mode].append(time.perf_counter()-start)
        truth=[h['id'] for h in outputs['exact_full']['hits']];retrieved=[h['id'] for h in outputs['owned_ann']['hits']]
        results.append(dict(id=query['id'],encoding_seconds=float(encoding[qi]),ordered_top10_equal=truth==retrieved,
                            exact_top10_retained=len(set(truth)&set(retrieved))/len(truth),
                            outputs=outputs,timings={m:dict(samples=v,median=float(np.median(v))) for m,v in timings.items()}))
        print(query['id'],results[-1]['ordered_top10_equal'],outputs['owned_ann']['source_rows'],flush=True)
    report=dict(queries=len(results),sources=len(records),experts=len(exact.owners),faiss=faiss.__version__,node=platform.node(),peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                build_seconds=dict(exact_full=exact_build,owned_ann=ann_build),
                ordered_top10_equal=sum(r['ordered_top10_equal'] for r in results),mean_exact_top10_retained=float(np.mean([r['exact_top10_retained'] for r in results])),
                mean_source_rows=float(np.mean([r['outputs']['owned_ann']['source_rows'] for r in results])),
                mean_median_seconds={m:float(np.mean([r['timings'][m]['median'] for r in results])) for m in indexes},
                mean_query_encoding_seconds=float(np.mean(encoding)),
                warning='Fidelity to one frozen semantic ranking, not expert relevance, ability or end-to-end quality. ANN is heuristic, count stopping is not an exactness certificate. Single node/20 reused queries, not SLA. CPU times exclude index initialization, query encoding, source reads, planning and assessment.')
    (ROOT/'evaluation.json').write_text(json.dumps(results,indent=2));(ROOT/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['encode','benchmark'],required=True);run(parser.parse_args().stage)
