"""Experimental full-source GPU scorer with CPU ownership aggregation."""
import time
import numpy as np
from .owned_vector_index import ExactOwnedVectorIndex


class GpuOwnedVectorIndex:
    """Keep a FP32 source snapshot on CUDA; results may differ numerically.

    Query vectors and scores cross the host/device boundary on each call.
    All source scores are computed; this is not ANN or an exact equivalence
    certificate relative to the CPU reduction order. Model encoding and GPU
    contention are outside this component. Selected-source scoring is scoped
    to the supplied expert IDs; missing IDs remain explicitly unavailable.
    """
    def __init__(self,reference,device='cuda'):
        import torch
        if not isinstance(reference,ExactOwnedVectorIndex):raise ValueError('Require an exact owned reference')
        self.device=torch.device(device)
        if self.device.type!='cuda' or not torch.cuda.is_available():raise ValueError('Require a CUDA device')
        if torch.backends.cuda.matmul.allow_tf32:raise ValueError('Disable TF32 for the FP32 diagnostic')
        self.torch=torch;self.reference=reference
        # Snapshot contents and owner metadata must remain unchanged for the index lifetime.
        self.vectors=torch.tensor(reference.vectors,device=self.device,dtype=torch.float32)
        torch.cuda.synchronize(self.device)

    def search_vector(self,query,k=5,eligible=None,*,evidence_limit=3):
        if self.torch.backends.cuda.matmul.allow_tf32:raise ValueError('TF32 was enabled after initialization')
        if type(evidence_limit) is not int or evidence_limit < 1:raise ValueError('Invalid source evidence limit')
        q=np.asarray(query,dtype=np.float32)
        if q.shape!=(self.reference.vectors.shape[1],) or not np.isfinite(q).all() or type(k) is not int or not 1<=k<=500:raise ValueError('Invalid query vector or result budget')
        start=time.perf_counter()
        if eligible is not None:
            requested=set(eligible)
            if any(not isinstance(e,str) or not e for e in requested):raise ValueError('Invalid eligible expert IDs')
            candidates=sorted(requested & self.reference.owners.keys())
            unavailable=sorted(requested-self.reference.owners.keys())
            if not candidates:
                return {'hits':[],'branch':'no_sources','source_rows':0,'candidate_experts':0,'depths':[],
                        'unavailable_experts':unavailable,'seconds':time.perf_counter()-start,
                        'notice':'No eligible source; absence does not establish inability.'}
            selected=sorted({i for eid in candidates for i in self.reference.owners[eid]})
            with self.torch.inference_mode():
                device_q=self.torch.as_tensor(q,device=self.device,dtype=self.torch.float32)
                indices=self.torch.as_tensor(selected,device=self.device,dtype=self.torch.long)
                scores=self.torch.mv(self.vectors.index_select(0,indices),device_q).cpu().numpy()
            if not np.isfinite(scores).all():raise ValueError('Non-finite GPU scores')
            lookup={i:float(score) for i,score in zip(selected,scores)};hits=[]
            for eid in candidates:
                ordered=sorted(self.reference.owners[eid],key=lambda i:(-lookup[i],self.reference.source_ids[i]))
                hits.append(dict(id=eid,score=lookup[ordered[0]],evidence=[dict(id=self.reference.source_ids[i],score=lookup[i]) for i in ordered[:evidence_limit]]))
            hits.sort(key=lambda hit:(-hit['score'],hit['id']))
            return {'hits':hits[:k],'branch':'gpu_selected','source_rows':len(selected),'candidate_experts':len(candidates),
                    'depths':[],'unavailable_experts':unavailable,'seconds':time.perf_counter()-start,
                    'notice':'Selected-source FP32 GPU scoring; CPU reductions can differ. Candidate selection and source relevance are unverified.'}
        with self.torch.inference_mode():
            device_q=self.torch.as_tensor(q,device=self.device,dtype=self.torch.float32)
            scores=self.torch.mv(self.vectors,device_q).cpu().numpy()
        if not np.isfinite(scores).all():raise ValueError('Non-finite GPU scores')
        maxima=np.maximum.reduceat(scores[self.reference.owner_rows],self.reference.owner_starts)
        positions=np.argsort(-maxima,kind='stable')[:k];hits=[]
        for position in positions:
            eid=self.reference.owner_ids[position]
            ordered=sorted(self.reference.owners[eid],key=lambda i:(-float(scores[i]),self.reference.source_ids[i]))
            hits.append(dict(id=eid,score=float(maxima[position]),evidence=[dict(id=self.reference.source_ids[i],score=float(scores[i])) for i in ordered[:evidence_limit]]))
        return {'hits':hits,'branch':'gpu_full','source_rows':len(self.reference.vectors),'candidate_experts':len(self.reference.owner_ids),
                'depths':[],'unavailable_experts':[],'seconds':time.perf_counter()-start,
                'notice':'Full FP32 GPU scoring with CPU ownership aggregation; results can differ from CPU reductions. Not semantic expertise accuracy.'}
