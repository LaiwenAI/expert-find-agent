"""Training-free two-channel fusion controls, with exact/ANN references."""
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
from expert_search.channel_fusion import fuse
from expert_search.owned_vector_index import ExactOwnedVectorIndex,OwnedVectorIndex

ROOT=Path('artifacts/bilingual-fusion-diagnostic-v1')
PARENT=Path('artifacts/bilingual-retrieval-v1')


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


class Effort:
    def __init__(self,index):self.index=index;self.ntotal=index.ntotal
    def search(self,query,k):
        self.index.hnsw.efSearch=max(512,k)
        return self.index.search(query,k)


def run():
    import faiss
    faiss.omp_set_num_threads(1)
    previous=json.loads((PARENT/'protocol.json').read_text());summary=json.loads((PARENT/'summary.json').read_text())
    source=Path('artifacts/global-expert-branches-v1/branches.jsonl');vp=Path('artifacts/global-expert-branch-benchmark-v1/vectors.npy')
    if sha(source)!=previous['source_sha256'] or sha(vp)!=previous['source_vectors_sha256'] or sha(PARENT/'query-vectors.npz')!=summary['encoding']['query_vectors_sha256'] or sha(PARENT/'evaluation.jsonl')!=summary['evaluation_sha256']:raise ValueError('Changed frozen data')
    protocol=dict(parent_protocol_sha256=sha(PARENT/'protocol.json'),source_sha256=sha(source),vectors_sha256=sha(vp),query_vectors_sha256=sha(PARENT/'query-vectors.npz'),k=10,channel_depth=50,rrf_constant=60,channel_order=['paper','profile'],ann=dict(M=32,efConstruction=128,efSearch_floor=512,source_limit=1024),threads=1,code_sha256={str(p):sha(p) for p in [Path(__file__),Path('src/expert_search/channel_fusion.py'),Path('src/expert_search/owned_vector_index.py'),Path('src/expert_search/source_routing.py')]},scope='Standard RRF and interleaving as source-channel controls, no algorithmic novelty or relevance claim. Same1000construction derived pairs, no new embeddings/training/LLM. Exact fusion uses channel top50 truncation, not full-universe RRF. Timings exclude encoding/build/loading; ANN pair retrieval shared between fusion alternatives, each adds its own fusion time. Missing a channel is not inability; RRF may reward multi-channel representation.')
    ROOT.mkdir(exist_ok=True);marker=ROOT/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=protocol:raise ValueError('Changed fusion diagnostic')
    marker.write_text(json.dumps(protocol,indent=2)+'\n')
    records=list(map(json.loads,source.read_text().splitlines()));matrix=np.load(vp,mmap_mode='r',allow_pickle=False)
    with np.load(PARENT/'query-vectors.npz',allow_pickle=False) as cache:queries=cache['vectors'];ids=list(cache['ids'])
    references=list(map(json.loads,(PARENT/'evaluation.jsonl').read_text().splitlines()))
    if len(references)!=1000 or ids!=[r['id'] for r in references]:raise ValueError('Changed query order')
    start=time.perf_counter();all_index=ExactOwnedVectorIndex(matrix,[r['id'] for r in records],[[r['expert_id']] for r in records]);exact={};ann={}
    for kind in ['paper','profile']:
        indices=[i for i,r in enumerate(records) if r['kind']==kind];vectors=matrix[indices];selected=[records[i] for i in indices]
        arguments=(vectors,[r['id'] for r in selected],[[r['expert_id']] for r in selected])
        exact[kind]=ExactOwnedVectorIndex(*arguments);ann[kind]=OwnedVectorIndex(*arguments);ann[kind].index=Effort(ann[kind].index)
    build=time.perf_counter()-start
    all_index.search_vector(queries[0],k=10)
    for index in [*exact.values(),*ann.values()]:index.search_vector(queries[0],k=50)
    def retrieve(indexes,query):
        start=time.perf_counter();rankings={}
        for name,index in indexes.items():rankings[name]=[r['id'] for r in index.search_vector(query,k=50)['hits']]
        return rankings,time.perf_counter()-start
    results=[];path=ROOT/'evaluation.jsonl'
    with path.open('w') as stream:
        for qi,reference in enumerate(references):
            row={key:reference[key] for key in ['id','parent_id','group_id','intent_type','status']};row['languages']={}
            for li,language in enumerate(['zh','en']):
                query=queries[2*qi+li];values={};retrieved={}
                modes=['raw','exact_channels','ann_channels'];offset=(qi+li)%3;modes=modes[offset:]+modes[:offset]
                for mode in modes:
                    if mode=='raw':
                        start=time.perf_counter();hits=all_index.search_vector(query,k=10);seconds=time.perf_counter()-start;ranking=[h['id'] for h in hits['hits']]
                        if ranking!=reference['languages'][language]['exact']['ranking']:raise ValueError('Raw reference mismatch')
                        values['raw']=dict(ranking=ranking,seconds=seconds)
                    else:retrieved[mode]=retrieve(exact if mode=='exact_channels' else ann,query)
                for prefix,(rankings,seconds) in retrieved.items():
                    for fusion in ['rrf','interleave']:
                        start=time.perf_counter();hits=fuse(rankings,k=10,depth=50,method=fusion,constant=60);elapsed=time.perf_counter()-start
                        values[prefix+'_'+fusion]=dict(ranking=[h['id'] for h in hits],seconds=seconds+elapsed,channel_ranks={h['id']:h['channel_ranks'] for h in hits})
                for fusion in ['rrf','interleave']:
                    baseline=values['exact_channels_'+fusion]['ranking'];value=values['ann_channels_'+fusion]
                    value.update(ordered_fidelity=value['ranking']==baseline,exact_top10_retained=len(set(value['ranking'])&set(baseline))/10)
                row['languages'][language]=values
            row['overlap']={m:len(set(row['languages']['zh'][m]['ranking'])&set(row['languages']['en'][m]['ranking']))/10 for m in row['languages']['zh']}
            stream.write(json.dumps(row)+'\n');stream.flush();results.append(row)
            if (qi+1)%100==0:print('Fusion pairs',qi+1,flush=True)
    def aggregate(rows):
        return {m:dict(mean_overlap=float(np.mean([r['overlap'][m] for r in rows])),mean_seconds=float(np.mean([r['languages'][l][m]['seconds'] for r in rows for l in ['zh','en']])),paper_backed_fraction=float(np.mean([sum(e in exact['paper'].owners for e in r['languages'][l][m]['ranking'])/10 for r in rows for l in ['zh','en']])),**(dict(ordered_fidelity=sum(r['languages'][l][m]['ordered_fidelity'] for r in rows for l in ['zh','en']),mean_exact_top10_retained=float(np.mean([r['languages'][l][m]['exact_top10_retained'] for r in rows for l in ['zh','en']]))) if m.startswith('ann_channels') else {})) for m in rows[0]['overlap']}
    report=dict(pairs=len(results),methods=aggregate(results),by_status={s:aggregate([r for r in results if r['status']==s]) for s in sorted({r['status'] for r in results})},build_seconds=build,protocol_sha256=sha(marker),evaluation_sha256=sha(path),node=platform.node(),scope=protocol['scope'])
    (ROOT/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['methods'],indent=2))


if __name__=='__main__':run()
