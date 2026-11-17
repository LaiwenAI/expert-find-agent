"""Conservative owner-level upper bounds for exact maximum source BM25."""
import math
import time
import numpy as np
from .retrieval import tokens


class ExpertTermUpperBounds:
    """Offline per-term maximum source contribution for each owner.

    Summing these maxima bounds every source owned by the expert, although
    maxima for different terms may come from different sources. Bounds are
    rounded upward per addition; this certifies an execution result, not fit.
    """
    def __init__(self,resident,evidence_by_expert):
        start=time.perf_counter();self.resident=resident;self.experts=tuple(sorted(evidence_by_expert));owners={};self.positions={e:i for i,e in enumerate(self.experts)};self.owned={e:tuple(evidence_by_expert[e]) for e in self.experts}
        for expert,ids in evidence_by_expert.items():
            for eid in ids:
                if eid in owners:raise ValueError('Duplicate evidence owner')
                owners[eid]=self.positions[expert]
        if set(owners)!=set(resident.ids):raise ValueError('Owner and resident corpus mismatch')
        owner_by_doc=np.asarray([owners[eid] for eid in resident.ids],dtype=np.int32);self.postings={};size=0
        for term,(lo,hi) in resident.offsets.items():
            owner_ids,inverse=np.unique(owner_by_doc[resident.indices[lo:hi]],return_inverse=True)
            maxima=np.zeros(len(owner_ids),dtype=np.float64)
            np.maximum.at(maxima,inverse,resident.values[lo:hi]);owner_ids.flags.writeable=False;maxima.flags.writeable=False
            self.postings[term]=(owner_ids,maxima);size+=owner_ids.nbytes+maxima.nbytes
        self.array_bytes=size;self.initialization_seconds=time.perf_counter()-start

    def upper(self,query,experts):
        experts=list(experts)
        if len(set(experts))!=len(experts) or set(experts)-set(self.positions):raise ValueError('Invalid bound candidates')
        positions=np.array([self.positions[e] for e in experts],dtype=np.int32);totals=np.zeros(len(experts),dtype=np.float64)
        for term in sorted(set(tokens(query))):
            posting=self.postings.get(term)
            if posting is None:continue
            owners,values=posting;at=np.searchsorted(owners,positions);slots=np.flatnonzero(at<len(owners));slots=slots[owners[at[slots]]==positions[slots]]
            totals[slots]=np.nextafter(totals[slots]+values[at[slots]],np.inf)
        if not np.isfinite(totals).all():raise ValueError('Nonfinite upper bound')
        return dict(zip(experts,totals.tolist()))


def certified_candidate_max(pipeline,bounds,query,k=5,pool=50,seed=20,evidence_per_expert=1):
    """Return exact pool-reranked top-k and owned evidence using safe pruning."""
    if pipeline.resident is None or bounds.resident is not pipeline.resident:raise ValueError('Bounds must match resident snapshot')
    if any(type(n)!=int for n in (k,pool,seed,evidence_per_expert)) or not 1<=k<=seed<=pool<=50 or not 1<=evidence_per_expert<=10:raise ValueError('Invalid budgets')
    pipeline._check_resident_snapshot();start=time.perf_counter()
    candidates=pipeline.ranker.search(query,pool);rows=candidates['results'];ids=[r['expert']['id'] for r in rows];experts={r['expert']['id']:r['expert'] for r in rows}
    if any(tuple(pipeline.evidence_by_expert[e])!=bounds.owned.get(e) for e in ids):raise ValueError('Bounds owner mapping changed')
    initial=ids[:seed];rest=ids[seed:]
    def score(group):
        source_ids=[s for expert in group for s in pipeline.evidence_by_expert[expert]]
        source_scores=pipeline.resident.selected_scores(query,source_ids)
        owner_scores={expert:max((source_scores[s] for s in pipeline.evidence_by_expert[expert]),default=0.) for expert in group}
        return source_scores,owner_scores
    source_scores,scores=score(initial)
    if len(initial)<k:keep=rest
    else:
        threshold=sorted(scores.values(),reverse=True)[k-1]
        upper=bounds.upper(query,rest);keep=[e for e in rest if upper.get(e,0.)>=threshold]
    added,owner_scores=score(keep);source_scores.update(added);scores.update(owner_scores)
    ranking=sorted(scores,key=lambda e:(-scores[e],e))[:k]
    evidence=pipeline.evidence.evidence_from_scores(ranking,pipeline.evidence_by_expert,source_scores,evidence_per_expert)
    result=[dict(expert=experts[e],score=scores[e],evidence=evidence[e]) for e in ranking]
    pipeline._check_resident_snapshot()
    return {'results':result,'pool_size':len(ids),'scored_candidates':len(initial)+len(keep),
            'pruned_candidates':len(rest)-len(keep),'seconds':time.perf_counter()-start,
            'source_snapshot_sha256':candidates['source_snapshot_sha256'],
            'scope':'Exact max-BM25 reranking of the supplied candidate-LM pool, not global ranking or a relevance guarantee.'}
