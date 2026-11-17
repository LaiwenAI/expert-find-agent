"""Standard HNSW search-effort contrast after default route fidelity loss."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from expert_search.owned_vector_index import OwnedVectorIndex,ExactOwnedVectorIndex


class EffortFloor:
    """Minimal search interface; keep one shared graph with a fixed effort floor."""
    def __init__(self,index):self.index=index;self.ntotal=index.ntotal
    def search(self,q,k):
        self.index.hnsw.efSearch=max(512,k)
        return self.index.search(q,k)


def run():
    import faiss
    faiss.omp_set_num_threads(1)
    parent=Path('artifacts/global-owned-routing-v1');out=Path('artifacts/global-owned-effort-v1');out.mkdir(exist_ok=True)
    parent_protocol=json.loads((parent/'protocol.json').read_text())
    protocol=dict(parent_protocol_sha256=hashlib.sha256((parent/'protocol.json').read_bytes()).hexdigest(),query_vectors_sha256=hashlib.sha256((parent/'query-vectors.npz').read_bytes()).hexdigest(),
                  code_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['experiments/global_owned_effort.py','src/expert_search/owned_vector_index.py','src/expert_search/source_routing.py']},
                  methods=['exact_full','ef64_floor','ef512_floor'],k=10,repeats=5,threads=1,M=32,efConstruction=128,
                  scope='Exploratory fixed higher-effort standard baseline after default query-level fidelity loss. Same20 reused demands, cached queries, one shared HNSW graph; mode order alternates, one warmup each. No relevance labels, fitted thresholds, new stopping rule or semantic accuracy claim. ANN remains heuristic even if these20 match.')
    marker=out/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=protocol:raise ValueError('Changed higher-effort protocol')
    marker.write_text(json.dumps(protocol,indent=2))
    source=Path('artifacts/global-expert-branches-v1/branches.jsonl');vp=Path('artifacts/global-expert-branch-benchmark-v1/vectors.npy')
    if hashlib.sha256(source.read_bytes()).hexdigest()!=parent_protocol['source_sha256']:raise ValueError('Changed source snapshot')
    with vp.open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=parent_protocol['vectors_sha256']:raise ValueError('Changed vectors')
    records=[json.loads(s) for s in source.read_text().splitlines()];vectors=np.load(vp,mmap_mode='r',allow_pickle=False)
    ids=[r['id'] for r in records];owners=[[r['expert_id']] for r in records]
    start=time.perf_counter();exact=ExactOwnedVectorIndex(vectors,ids,owners);default=OwnedVectorIndex(vectors,ids,owners)
    high=OwnedVectorIndex(vectors,ids,owners,index=EffortFloor(default.index));build=time.perf_counter()-start
    indexes=dict(exact_full=exact,ef64_floor=default,ef512_floor=high)
    with np.load(parent/'query-vectors.npz',allow_pickle=False) as f:qvectors=f['vectors'];qids=f['ids']
    rows=[]
    for qid,q in zip(qids,qvectors):
        timing={m:[] for m in indexes};results={}
        for index in indexes.values():index.search_vector(q,k=10)
        for repeat in range(5):
            order=list(indexes) if repeat%2==0 else list(indexes)[::-1]
            for method in order:
                start=time.perf_counter();results[method]=indexes[method].search_vector(q,k=10);timing[method].append(time.perf_counter()-start)
        truth=[h['id'] for h in results['exact_full']['hits']]
        rows.append(dict(id=str(qid),results=results,timing={m:dict(samples=t,median=float(np.median(t))) for m,t in timing.items()},
                         fidelity={m:dict(ordered_top10_equal=[h['id'] for h in r['hits']]==truth,exact_top10_retained=len(set(truth)&{h['id'] for h in r['hits']})/len(truth)) for m,r in results.items()}))
        print(qid,rows[-1]['fidelity']['ef512_floor'],flush=True)
    summary=dict(queries=len(rows),shared_graph_build_and_index_copies_seconds=build,
                 methods={m:dict(mean_median_seconds=float(np.mean([r['timing'][m]['median'] for r in rows])),ordered_top10_equal=sum(r['fidelity'][m]['ordered_top10_equal'] for r in rows),mean_exact_top10_retained=float(np.mean([r['fidelity'][m]['exact_top10_retained'] for r in rows]))) for m in indexes},
                 warning='Higher effort is a mature approximate-search baseline. This reused20-demand fidelity result is not generalization, expert relevance/precision, whole-agent speedup or exactness guarantee. HNSW internal graph distance work is additional to reported source rescoring rows.')
    (out/'evaluation.json').write_text(json.dumps(rows,indent=2));(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':run()
