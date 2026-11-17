"""Compare independent and shared exact-reference construction on frozen vectors."""
import argparse
import json
from pathlib import Path
import platform
import statistics
import time
import numpy as np
from expert_search.owned_vector_index import ExactOwnedVectorIndex,OwnedVectorIndex
from expert_search.statistics import clustered_paired_interval
from experiments.bilingual.adaptive_effort_diagnostic import sha


def main(output):
    import faiss
    root=Path('artifacts/bilingual-retrieval-v1');selection=json.loads((root/'selection.json').read_text());encoding=json.loads((root/'encoding.json').read_text());prior=json.loads((root/'summary.json').read_text());spec=json.loads((root/'protocol.json').read_text())
    source=Path('artifacts/global-expert-branches-v1/branches.jsonl');vp=Path('artifacts/global-expert-branch-benchmark-v1/vectors.npy')
    if sha(root/'queries.jsonl')!=selection['queries_sha256'] or sha(root/'query-vectors.npz')!=encoding['query_vectors_sha256'] or sha(root/'protocol.json')!=prior['protocol_sha256'] or sha(source)!=spec['source_sha256'] or sha(vp)!=spec['source_vectors_sha256']:raise ValueError('Changed frozen inputs')
    queries=[json.loads(line) for line in (root/'queries.jsonl').read_text().splitlines()]
    with np.load(root/'query-vectors.npz',allow_pickle=False) as arrays:
        vectors=arrays['vectors'].copy();qids=arrays['ids'].tolist()
    if len(queries)!=1000 or qids!=[r['id'] for r in queries] or vectors.shape!=(2000,1024):raise ValueError('Changed query inventory')
    records=[json.loads(s) for s in source.read_text().splitlines()];ids=[r['id'] for r in records];owners=[[r['expert_id']] for r in records];matrix=np.load(vp,mmap_mode='r',allow_pickle=False)
    if matrix.shape!=(len(records),1024):raise ValueError('Changed source inventory')
    output.mkdir(parents=True,exist_ok=False)
    protocol={'pairs':400,'views':800,'query_selection':'first 400 frozen construction pairs','source_sha256':sha(source),'source_vectors_sha256':sha(vp),'queries_sha256':sha(root/'queries.jsonl'),'query_vectors_sha256':sha(root/'query-vectors.npz'),
              'construction':'One independent reference followed by one shared reference after common ANN graph construction.',
              'query_timing':'One alternating-order query per exact reference/view; excludes index build, model loading, query encoding and reasoning.',
              'code_sha256':{str(p):sha(p) for p in (Path(__file__),Path('src/expert_search/owned_vector_index.py'),Path('src/expert_search/source_routing.py'),Path('src/expert_search/statistics.py'))},
              'scope':'Post-hoc reused development views. Compare exact reference behavior and memory allocation only; no expert-fit relevance or model quality claim. The shared arm reuses a read-only vector array whose owner must not mutate it.'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    faiss.omp_set_num_threads(1);start=time.perf_counter();graph=OwnedVectorIndex(matrix,ids,owners);graph_time=time.perf_counter()-start;references={};construction={}
    for mode in ('independent','shared'):
        started=time.perf_counter();references[mode]=ExactOwnedVectorIndex(matrix,ids,owners) if mode=='independent' else ExactOwnedVectorIndex.from_owned(graph);construction[mode]=time.perf_counter()-started
    if not np.shares_memory(references['shared'].vectors,graph.vectors) or np.shares_memory(references['independent'].vectors,graph.vectors):raise ValueError('Unexpected vector allocation')
    if references['shared'].vectors.flags.writeable:raise ValueError('Shared snapshot is writable')
    for reference in references.values():reference.search_vector(vectors[0],k=10)
    rows=[]
    for i,row in enumerate(queries[:400]):
        for j,language in enumerate(('zh','en')):
            order=('independent','shared') if (2*i+j)%2==0 else ('shared','independent');result={};seconds={}
            for name in order:
                start=time.perf_counter();result[name]=references[name].search_vector(vectors[2*i+j],k=10);seconds[name]=time.perf_counter()-start
            if {k:v for k,v in result['independent'].items() if k!='seconds'}!={k:v for k,v in result['shared'].items() if k!='seconds'}:raise ValueError('Changed complete result')
            rows.append({'id':row['id'],'group_id':row['group_id'],'language':language,'seconds':seconds})
        if (i+1)%100==0:print('Completed pairs',i+1,flush=True)
    (output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    interval=clustered_paired_interval([r['seconds']['shared']-r['seconds']['independent'] for r in rows],[r['group_id'] for r in rows]);interval['sampling_unit']='whole company group; language-view-weighted mean'
    report={'pairs':400,'views':len(rows),'matching_complete_results':len(rows),'node':platform.node(),'faiss':faiss.__version__,
            'graph_build_seconds':graph_time,'reference_construction_seconds':construction,'vector_bytes_per_matrix':int(graph.vectors.nbytes),
            'distinct_vector_matrix_bytes_graph_plus_reference':{'independent':int(graph.vectors.nbytes+references['independent'].vectors.nbytes),'shared':int(graph.vectors.nbytes)},
            'mean_query_seconds':{a:statistics.mean(r['seconds'][a] for r in rows) for a in references},'paired_query_seconds':interval,
            'protocol_sha256':sha(output/'protocol.json'),'results_sha256':sha(output/'results.json'),
            'scope':protocol['scope']+' Vector array bytes exclude ANN graph, metadata, allocator, original mmap, temporary memory and process RSS. One construction timing per arm is not a cold-start distribution. Query timings are alternating but share hardware, with one observation per view. No concurrency safety guarantee if underlying snapshot or graph is mutated.'}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
