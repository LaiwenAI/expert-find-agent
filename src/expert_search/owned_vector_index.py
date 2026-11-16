"""Shared-source vector retrieval with explicit ownership and heuristic routing."""
import time
import numpy as np
from .source_routing import expand_sources


class OwnedVectorIndex:
    def __init__(self, vectors, source_ids, source_owners, index=None, switch_sources=1024):
        x=np.asarray(vectors,dtype=np.float32)
        if x.ndim!=2 or not len(x) or not x.shape[1] or not np.isfinite(x).all():
            raise ValueError('Invalid source vectors')
        if len(source_ids)!=len(x) or len(set(source_ids))!=len(x) or any(not isinstance(s,str) or not s for s in source_ids):
            raise ValueError('Invalid source IDs')
        if len(source_owners)!=len(x) or any(not o or any(not isinstance(e,str) or not e for e in o) for o in source_owners):
            raise ValueError('Invalid source ownership')
        if type(switch_sources) is not int or switch_sources<1:raise ValueError('Invalid switch budget')
        self.vectors=np.ascontiguousarray(x).copy();self.vectors.setflags(write=False)
        self.source_ids=list(source_ids);self.source_owners=[sorted(set(o)) for o in source_owners]
        self.owners={}
        for i,owners in enumerate(self.source_owners):
            for eid in owners:self.owners.setdefault(eid,[]).append(i)
        if index is None:
            import faiss
            index=faiss.IndexHNSWFlat(x.shape[1],32,faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction=128;index.add(self.vectors)
        if getattr(index,'ntotal',len(x))!=len(x):raise ValueError('Index/source count mismatch')
        self.index=index;self.switch_sources=switch_sources

    def search_vector(self, query, k=5, eligible=None):
        start=time.perf_counter();q=np.asarray(query,dtype=np.float32)
        if q.shape!=(self.vectors.shape[1],) or not np.isfinite(q).all() or type(k) is not int or not 1<=k<=500:
            raise ValueError('Invalid query vector or result budget')
        requested=set(self.owners) if eligible is None else set(eligible)
        available=requested & set(self.owners);unavailable=sorted(requested-set(self.owners))
        if not available:return dict(hits=[],unavailable_experts=unavailable,seconds=time.perf_counter()-start,branch='no_sources')
        all_rows=sorted({i for e in available for i in self.owners[e]});route=None
        if len(all_rows)<=self.switch_sources:candidates=sorted(available);branch='eligible_full'
        else:
            route=expand_sources(self.index,q,self.source_owners,available,target=max(k,10),initial=32,limit=1024)
            candidates=route['candidates'];branch='ann'
            if not route['target_met']:candidates=sorted(available);branch='fallback_full'
        selected=np.array(sorted({i for e in candidates for i in self.owners[e]}),dtype=np.int64)
        scores=np.einsum('ij,j->i',self.vectors[selected],q,optimize=False)
        lookup={int(i):float(s) for i,s in zip(selected,scores)};hits=[]
        for eid in candidates:
            ordered=sorted(self.owners[eid],key=lambda i:(-lookup[i],self.source_ids[i]))
            hits.append(dict(id=eid,score=lookup[ordered[0]],evidence=[dict(id=self.source_ids[i],score=lookup[i]) for i in ordered[:3]]))
        hits.sort(key=lambda h:(-h['score'],h['id']))
        return dict(hits=hits[:k],branch=branch,source_rows=len(selected),candidate_experts=len(candidates),
                    depths=route['depths'] if route else [],unavailable_experts=unavailable,seconds=time.perf_counter()-start,
                    notice='Maximum source dot product; ANN stopping is heuristic. Retrieval relevance and ownership metadata need original-source verification.')
