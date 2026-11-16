"""Training-free representative/radius index for exact maximum dot products.
Uses standard Cauchy-Schwarz bounds; no novelty claim for ball bounds.
"""
import numpy as np


class BoundedProfile:
    def __init__(self,vectors,groups=4):
        x=np.asarray(vectors,dtype=np.float64)
        if x.ndim!=2 or not len(x) or not np.isfinite(x).all() or groups<1:raise ValueError('Invalid vectors/groups')
        self.x=x;representatives=[0]
        nearest=np.sum((x-x[0])**2,axis=1)
        while len(representatives)<min(groups,len(x)):
            nxt=int(np.argmax(nearest))
            if nearest[nxt]<=1e-24:break
            representatives.append(nxt)
            nearest=np.minimum(nearest,np.sum((x-x[nxt])**2,axis=1))
        centers=x[representatives]
        assignments=np.argmin(np.sum((x[:,None,:]-centers[None,:,:])**2,axis=2),axis=1)
        self.blocks=[]
        for i,c in enumerate(centers):
            members=np.where(assignments==i)[0]
            if not len(members):continue
            radius=float(np.max(np.linalg.norm(x[members]-c,axis=1)))
            self.blocks.append((c,radius,members))

    def maximum(self,query,stable_dots=False):
        q=np.asarray(query,dtype=np.float64)
        if q.shape!=(self.x.shape[1],) or not np.isfinite(q).all():raise ValueError('Invalid query')
        norm=float(np.linalg.norm(q))
        bounds=[(float(c@q)+radius*norm,members) for c,radius,members in self.blocks]
        bounds.sort(key=lambda row:-row[0]);best=-float('inf');evaluated=0;skipped=0
        for upper,members in bounds:
            # Numerical slack preserves exactness near a floating-point boundary.
            if upper+1e-12<best:skipped+=len(members);continue
            scores=np.einsum('ij,j->i',self.x[members],q,optimize=False) if stable_dots else self.x[members]@q
            best=max(best,float(np.max(scores)));evaluated+=len(members)
        return dict(score=best,member_dot_products=evaluated,bound_dot_products=len(bounds),skipped_members=skipped)
