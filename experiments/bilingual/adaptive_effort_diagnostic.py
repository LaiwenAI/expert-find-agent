"""Test a fixed score-gap heuristic for adaptive HNSW effort on frozen vectors."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time
import numpy as np
from expert_search.owned_vector_index import ExactOwnedVectorIndex,OwnedVectorIndex
from expert_search.statistics import clustered_paired_interval


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


class Effort:
    def __init__(self,index,floor):self.index=index;self.floor=floor;self.ntotal=index.ntotal
    def search(self,query,k):
        self.index.hnsw.efSearch=max(self.floor,k)
        return self.index.search(query,k)


def main(output):
    import faiss
    root=Path('artifacts/bilingual-retrieval-v1');baseline=json.loads((root/'summary.json').read_text());encoding=json.loads((root/'encoding.json').read_text());selection=json.loads((root/'selection.json').read_text());proto=json.loads((root/'protocol.json').read_text())
    source=Path('artifacts/global-expert-branches-v1/branches.jsonl');vector_path=Path('artifacts/global-expert-branch-benchmark-v1/vectors.npy')
    if sha(root/'queries.jsonl')!=selection['queries_sha256'] or sha(root/'query-vectors.npz')!=encoding['query_vectors_sha256'] or sha(root/'protocol.json')!=encoding['protocol_sha256'] or sha(source)!=proto['source_sha256'] or sha(vector_path)!=proto['source_vectors_sha256'] or sha(root/'protocol.json')!=baseline['protocol_sha256']:
        raise ValueError('Changed frozen inputs')
    rows=[json.loads(line) for line in (root/'queries.jsonl').read_text().splitlines()]
    with np.load(root/'query-vectors.npz',allow_pickle=False) as q:
        query_vectors=q['vectors'].copy();ids=q['ids'].tolist()
    if len(rows)!=1000 or ids!=[r['id'] for r in rows] or query_vectors.shape!=(2000,1024) or not np.isfinite(query_vectors).all():raise ValueError('Changed query inventory')
    records=[json.loads(line) for line in source.read_text().splitlines()];ids=[r['id'] for r in records];owners=[[r['expert_id']] for r in records];matrix=np.load(vector_path,mmap_mode='r',allow_pickle=False)
    if len(matrix)!=len(records) or matrix.shape[1]!=1024:raise ValueError('Changed source vectors')
    output.mkdir(parents=True,exist_ok=False)
    protocol={'pairs':400,'views':800,'query_selection':'first 400 rows of frozen 1000-pair construction diagnostic',
              'source_sha256':sha(source),'source_vectors_sha256':sha(vector_path),'queries_sha256':sha(root/'queries.jsonl'),'query_vectors_sha256':sha(root/'query-vectors.npz'),
              'baseline_protocol_sha256':sha(root/'protocol.json'),'baseline_summary_sha256':sha(root/'summary.json'),
              'methods':['exact','ef64','ef512','gap_gate'],'gap_threshold':0.002,'k':10,'gate':'Run ef64 for 11 results; if score gap between ranks 10 and 11 is at most threshold, rerun ef512; otherwise keep ef64. Fewer than 11 results triggers ef512.',
              'timing':'One rotating-order measurement per query; gate includes repeated low/high calls when escalated. Excludes query encoding, loading, build and model inference.',
              'code_sha256':{str(p):sha(p) for p in (Path(__file__),Path('src/expert_search/owned_vector_index.py'),Path('src/expert_search/source_routing.py'),Path('src/expert_search/statistics.py'))},
              'scope':'Post-hoc development cohort. Gap rule is a heuristic and does not certify nearest neighbors, evidence relevance or expert fit. Same construction pairs previously evaluated; no unseen test claim.'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    faiss.omp_set_num_threads(1);start=time.perf_counter();exact=ExactOwnedVectorIndex(matrix,ids,owners);graph=OwnedVectorIndex(matrix,ids,owners)
    low=OwnedVectorIndex(matrix,ids,owners,index=Effort(graph.index,64));high=OwnedVectorIndex(matrix,ids,owners,index=Effort(graph.index,512));build=time.perf_counter()-start
    def run(method,q):
        started=time.perf_counter();escalated=False
        if method=='exact':result=exact.search_vector(q,k=10)
        elif method=='ef64':result=low.search_vector(q,k=11)
        elif method=='ef512':result=high.search_vector(q,k=11)
        else:
            result=low.search_vector(q,k=11);hits=result['hits'];escalated=len(hits)<11 or hits[9]['score']-hits[10]['score']<=.002
            if escalated:result=high.search_vector(q,k=11)
        elapsed=time.perf_counter()-started;hits=result['hits'][:10]
        return {'seconds':elapsed,'ranking':[h['id'] for h in hits],'sources':[h['evidence'][0]['id'] for h in hits],'escalated':escalated}
    methods=('exact','ef64','ef512','gap_gate')
    for name in methods:run(name,query_vectors[0])
    results=[]
    for i,row in enumerate(rows[:400]):
        for j,lang in enumerate(('zh','en')):
            order=methods[(2*i+j)%4:]+methods[:(2*i+j)%4];arms={name:run(name,query_vectors[2*i+j]) for name in order}
            reference=arms['exact']['ranking'];refsource=arms['exact']['sources']
            for arm in arms.values():arm['ordered_equal']=arm['ranking']==reference;arm['top10_retention']=len(set(arm['ranking'])&set(reference))/10;arm['same_source_ordered']=arm['ranking']==reference and arm['sources']==refsource
            results.append({'id':row['id'],'group_id':row['group_id'],'language':lang,'arms':arms})
        if (i+1)%50==0:print('Completed pairs',i+1,flush=True)
    (output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    paired_time=clustered_paired_interval([r['arms']['gap_gate']['seconds']-r['arms']['ef512']['seconds'] for r in results],[r['group_id'] for r in results])
    paired_time['sampling_unit']='whole company group; language-view-weighted mean'
    report={'pairs':400,'views':len(results),'index_build_and_copies_seconds':build,'faiss':faiss.__version__,'node':platform.node(),
            'methods':{a:{'mean_seconds':statistics.mean(r['arms'][a]['seconds'] for r in results),'ordered_equal':sum(r['arms'][a]['ordered_equal'] for r in results),
                          'same_experts_and_sources':sum(r['arms'][a]['same_source_ordered'] for r in results),'mean_exact_top10_retained':statistics.mean(r['arms'][a]['top10_retention'] for r in results),
                          'escalated':sum(r['arms'][a]['escalated'] for r in results)} for a in methods},
            'paired_gate_minus_high_seconds':paired_time,
            'protocol_sha256':sha(output/'protocol.json'),'results_sha256':sha(output/'results.json'),
            'scope':protocol['scope']+' Bootstrap clusters both language views by company group; shared hardware, repeated measurements and one observation per view preclude service-level inference. Fidelity to exact vector scores is not relevance or accuracy.'}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
