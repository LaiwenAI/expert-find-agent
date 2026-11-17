"""Auxiliary evidence extraction development test, never expert-fit evaluation."""
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
import numpy as np
from expert_search.evidencebench_data import prepare,aspect_coverage
from expert_search.qwen_embedding import QwenEmbedding
from expert_search.retrieval import tokens

TASK='Retrieve sentences containing concrete scientific evidence relevant to the hypothesis.'


def bm25(query,sentences):
    counts=[Counter(tokens(s)) for s in sentences];n=len(counts);avg=sum(map(lambda c:sum(c.values()),counts))/max(n,1)
    df=Counter(t for c in counts for t in c);scores=np.zeros(n)
    for term in sorted(set(tokens(query))):
        idf=math.log(1+(n-df[term]+.5)/(df[term]+.5))
        for i,c in enumerate(counts):
            tf=c.get(term,0)
            if tf:scores[i]+=idf*tf*2.2/(tf+1.2*(.25+.75*sum(c.values())/max(avg,1e-30)))
    return sorted(range(n),key=lambda i:(-float(scores[i]),i))


def run():
    data=prepare();out=Path('artifacts/evidencebench-pilot-v1');out.mkdir(exist_ok=True)
    protocol=dict(version=1,split='dev',dataset_sha256=hashlib.sha256(Path('artifacts/public/evidencebench/dev.json').read_bytes()).hexdigest(),methods=['bm25','qwen','mmr'],k=10,mmr_weight=.5,mmr_semantic_pool=50,task=TASK,max_length=512,batch_size=8,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='All 37 upstream development papers, unchanged sentence pools. Gold aspects used only after selection. BM25 k1=1.2,b=.75; frozen Qwen; standard MMR at fixed weight. No own training or originality claim. Auxiliary evidence extraction only, no expert ranking/enterprise labels.')
    marker=out/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())!=protocol:raise ValueError('Changed evidence pilot protocol')
    marker.write_text(json.dumps(protocol,indent=2));model=QwenEmbedding(max_length=512,batch_size=8);rows=[]
    for key,row in data.items():
        sentences=row['paper_as_candidate_pool'];target=out/(key+'.json');vp=out/(key+'.npy')
        if target.exists():result=json.loads(target.read_text())
        else:
            start=time.perf_counter()
            if vp.exists():vectors=np.load(vp,allow_pickle=False)
            else:vectors=model.encode([s if s.strip() else '[empty sentence]' for s in sentences]);np.save(vp,vectors)
            offline=time.perf_counter()-start;start=time.perf_counter();q=model.encode(['Instruct: '+TASK+'\nQuery: '+row['hypothesis']])[0];encoding=time.perf_counter()-start
            start=time.perf_counter();scores=vectors@q;rank=sorted(range(len(sentences)),key=lambda i:(-float(scores[i]),i));pool=rank[:50];chosen=[]
            while len(chosen)<min(10,len(pool)):
                candidates=[i for i in pool if i not in chosen]
                selected=min(candidates,key=lambda i:(-(.5*float(scores[i])-.5*max((float(vectors[i]@vectors[j]) for j in chosen),default=0.)),i));chosen.append(selected)
            dense_seconds=time.perf_counter()-start;start=time.perf_counter();lex=bm25(row['hypothesis'],sentences);lexical_seconds=time.perf_counter()-start
            selections=dict(bm25=lex[:10],qwen=rank[:10],mmr=chosen)
            result=dict(id=key,metrics={m:aspect_coverage(row,ids) for m,ids in selections.items()},selections=selections,offline_vector_seconds=offline,query_encoding_seconds=encoding,dense_and_mmr_seconds=dense_seconds,lexical_seconds=lexical_seconds)
            temp=target.with_suffix('.partial');temp.write_text(json.dumps(result,indent=2));temp.replace(target)
        rows.append(result);print(key,'completed',flush=True)
    report=dict(protocol=protocol,papers=len(rows),sentences=sum(len(r['paper_as_candidate_pool']) for r in data.values()),
        mean_aspect_recall={m:float(np.mean([r['metrics'][m]['aspect_recall'] for r in rows])) for m in protocol['methods']},mean_query_encoding_seconds=float(np.mean([r['query_encoding_seconds'] for r in rows])),
        warning='Upstream aspect labels, not all-human ground truth or future expert ability. Development-only auxiliary comparison. Offline/cache timings separated; no complete enterprise-agent latency/precision claim.')
    (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':run()
