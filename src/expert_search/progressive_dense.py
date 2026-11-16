"""Training-free prefix shortlist + full-vector refinement baseline.
Prefix truncation is approximate and model-dependent, never an exact bound.
"""
import numpy as np


def progressive_search(matrix,query,ids,k=10,prefix=256,depth=100):
    x=np.asarray(matrix);q=np.asarray(query)
    if x.ndim!=2 or q.shape!=(x.shape[1],) or len(ids)!=len(x) or len(set(ids))!=len(ids):raise ValueError('Invalid matrix or IDs')
    if not 1<=k<=depth or prefix<1 or prefix>x.shape[1]:raise ValueError('Invalid budgets')
    if not np.isfinite(x).all() or not np.isfinite(q).all():raise ValueError('Nonfinite vectors')
    a=x[:,:prefix];b=q[:prefix];norms=np.linalg.norm(a,axis=1);normq=np.linalg.norm(b)
    coarse=(a@b)/np.maximum(norms*normq,1e-30)
    shortlisted=sorted(range(len(x)),key=lambda i:(-float(coarse[i]),ids[i]))[:depth]
    scores=x[shortlisted]@q
    order=sorted(range(len(shortlisted)),key=lambda j:(-float(scores[j]),ids[shortlisted[j]]))[:k]
    return [ids[shortlisted[j]] for j in order]


def top_indices(scores,ids,k):
    """Partial selection with deterministic tie handling at the cutoff."""
    k=min(k,len(scores))
    if k<1:return []
    cutoff=np.partition(scores,len(scores)-k)[len(scores)-k]
    better=np.flatnonzero(scores>cutoff).tolist()
    ties=np.flatnonzero(scores==cutoff).tolist();ties.sort(key=lambda i:ids[i])
    chosen=better+ties[:k-len(better)]
    return sorted(chosen,key=lambda i:(-float(scores[i]),ids[i]))


class ProgressiveIndex:
    def __init__(self,matrix,ids,prefix=256):
        self.matrix=np.ascontiguousarray(matrix,dtype=np.float32);self.ids=list(ids);self.prefix=prefix
        if self.matrix.ndim!=2 or len(ids)!=len(matrix) or not 1<=prefix<=self.matrix.shape[1]:raise ValueError('Invalid index')
        a=self.matrix[:,:prefix].copy();norm=np.linalg.norm(a,axis=1,keepdims=True)
        self.coarse=np.ascontiguousarray(a/np.maximum(norm,1e-30))

    def full(self,q,k=10):
        scores=self.matrix@q
        return [self.ids[i] for i in top_indices(scores,self.ids,k)]

    def search(self,q,k=10,depth=100):
        if not 1<=k<=depth:raise ValueError('Invalid depth')
        b=q[:self.prefix];norm=np.linalg.norm(b)
        coarse=self.coarse@(b/max(float(norm),1e-30))
        chosen=top_indices(coarse,self.ids,depth)
        scores=self.matrix[chosen]@q
        local_ids=[self.ids[i] for i in chosen]
        return [local_ids[j] for j in top_indices(scores,local_ids,k)]
