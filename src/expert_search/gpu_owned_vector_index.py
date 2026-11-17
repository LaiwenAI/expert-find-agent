"""Experimental full-source GPU scorer with CPU ownership aggregation."""
import time
import numpy as np
from .owned_vector_index import ExactOwnedVectorIndex


class GpuOwnedVectorIndex:
    """Keep a FP32 source snapshot on CUDA; results may differ numerically.

    Query vectors and scores cross the host/device boundary on each call.
    All source scores are computed; this is not ANN or an exact equivalence
    certificate relative to the CPU reduction order. Model encoding and GPU
    contention are outside this component. Eligibility filtering is unsupported.
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

    def search_vector(self,query,k=5,eligible=None):
        if eligible is not None:raise ValueError('GPU scorer does not support eligibility filtering')
        if self.torch.backends.cuda.matmul.allow_tf32:raise ValueError('TF32 was enabled after initialization')
        q=np.asarray(query,dtype=np.float32)
        if q.shape!=(self.reference.vectors.shape[1],) or not np.isfinite(q).all() or type(k) is not int or not 1<=k<=500:raise ValueError('Invalid query vector or result budget')
        start=time.perf_counter()
        with self.torch.inference_mode():
            device_q=self.torch.as_tensor(q,device=self.device,dtype=self.torch.float32)
            scores=self.torch.mv(self.vectors,device_q).cpu().numpy()
        if not np.isfinite(scores).all():raise ValueError('Non-finite GPU scores')
        maxima=np.maximum.reduceat(scores[self.reference.owner_rows],self.reference.owner_starts)
        positions=np.argsort(-maxima,kind='stable')[:k];hits=[]
        for position in positions:
            eid=self.reference.owner_ids[position]
            ordered=sorted(self.reference.owners[eid],key=lambda i:(-float(scores[i]),self.reference.source_ids[i]))
            hits.append(dict(id=eid,score=float(maxima[position]),evidence=[dict(id=self.reference.source_ids[i],score=float(scores[i])) for i in ordered[:3]]))
        return {'hits':hits,'branch':'gpu_full','source_rows':len(self.reference.vectors),'candidate_experts':len(self.reference.owner_ids),
                'depths':[],'unavailable_experts':[],'seconds':time.perf_counter()-start,
                'notice':'Full FP32 GPU scoring with CPU ownership aggregation; results can differ from CPU reductions. Not semantic expertise accuracy.'}
