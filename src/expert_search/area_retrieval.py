"""Compact research-area BM25 baseline; no model calls or training."""
import math
import time
from collections import Counter,defaultdict
import numpy as np
from .corpus import records
from .retrieval import tokens
from .progressive_dense import top_indices


class AreaSearch:
    def __init__(self,corpus='artifacts/corpus/evidence.jsonl'):
        rows=[r for _,r in records(corpus) if r.get('field')=='\u7814\u7a76\u9886\u57df_CN']
        self.ids=[r['expert_id'] for r in rows];self.evidence=rows
        if len(self.ids)!=len(set(self.ids)):raise ValueError('Duplicate research-area profiles')
        counts=[Counter(tokens(r['text'])) for r in rows];n=len(rows)
        avg=sum(sum(c.values()) for c in counts)/max(n,1);df=Counter(t for c in counts for t in c)
        posting=defaultdict(list)
        for i,c in enumerate(counts):
            length=sum(c.values())
            for t,tf in c.items():
                idf=math.log(1+(n-df[t]+.5)/(df[t]+.5))
                value=idf*tf*2.2/(tf+1.2*(.25+.75*length/max(avg,1e-30)))
                posting[t].append((i,value,idf))
        self.postings={t:(np.array([x[0] for x in p],dtype=np.int32),np.array([x[1] for x in p]),np.array([x[2] for x in p])) for t,p in posting.items()}

    def search(self,query,k=10,method='bm25'):
        if k<1 or method not in ['bm25','coverage']:raise ValueError('Invalid search arguments')
        start=time.perf_counter();scores=np.zeros(len(self.ids))
        for token in sorted(set(tokens(query))):
            if token in self.postings:
                ids,bm25,coverage=self.postings[token];scores[ids]+=bm25 if method=='bm25' else coverage
        selected=top_indices(scores,self.ids,min(k,len(scores)))
        return dict(results=[dict(id=self.ids[i],score=float(scores[i]),evidence_id=self.evidence[i]['id'],source=self.evidence[i]['source']) for i in selected if scores[i]>0],seconds=time.perf_counter()-start)
