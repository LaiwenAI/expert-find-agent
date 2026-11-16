"""Eligibility-aware expansion of a source index; no relevance labels consumed."""
import numpy as np


def expand_sources(index, query, source_owners, eligible, target=10, initial=32, limit=1024):
    eligible=set(eligible)
    if not eligible or type(target) is not int or target<1 or not 1<=initial<=limit or not source_owners:
        raise ValueError('Invalid eligible pool or expansion budget')
    target=min(target,len(eligible));depth=min(initial,len(source_owners));limit=min(limit,len(source_owners))
    candidates=set();attempts=[]
    while True:
        if hasattr(index,'hnsw'):index.hnsw.efSearch=max(64,depth)
        _,indices=index.search(np.ascontiguousarray(query,dtype=np.float32).reshape(1,-1),depth)
        attempts.append(depth)
        for i in indices[0]:
            if i<0:continue
            if i>=len(source_owners):raise ValueError('Invalid source index response')
            candidates.update(e for e in source_owners[i] if e in eligible)
        if len(candidates)>=target or depth>=limit:break
        depth=min(depth*2,limit)
    return dict(candidates=sorted(candidates),depths=attempts,target_met=len(candidates)>=target,
                notice='Candidate-count stopping is heuristic, not a certificate that the best eligible experts were found.')
