"""Experimental bounded top-k membership for shared-source BM25 indexes."""
import math
import time
import numpy as np
from .retrieval import tokens


class SharedOwnerGate:
    """Compare one owner's max source score with all possible competitors.

    Per-term maxima are rounded upward during accumulation; shared sources
    contribute to every listed owner. The supplied resident index and owner
    mapping must remain immutable. This checks the numerical ranking only,
    never relevance, provenance, or individual contribution.
    """
    def __init__(self,resident,evidence_by_owner):
        start=time.perf_counter()
        if not isinstance(evidence_by_owner,dict) or not evidence_by_owner:
            raise ValueError('Require owner mapping')
        self.resident=resident;self.owners=tuple(sorted(evidence_by_owner))
        if any(not isinstance(e,str) or not e for e in self.owners):raise ValueError('Invalid owner')
        self.sources={e:tuple(sorted(set(evidence_by_owner[e]))) for e in self.owners}
        if any(not v for v in self.sources.values()):raise ValueError('Owner has no source')
        reverse={sid:[] for sid in resident.ids}
        if len(reverse)!=len(resident.ids):raise ValueError('Ambiguous source IDs')
        for pos,eid in enumerate(self.owners):
            for sid in self.sources[eid]:
                if sid not in reverse:raise ValueError('Unknown source')
                reverse[sid].append(pos)
        if any(not owners for owners in reverse.values()):raise ValueError('Source has no owner')
        self.positions={e:i for i,e in enumerate(self.owners)};self.bounds={}
        for term,(lo,hi) in resident.offsets.items():
            maximum={}
            for doc,value in zip(resident.indices[lo:hi],resident.values[lo:hi]):
                value=float(value)
                if not math.isfinite(value) or value<0:raise ValueError('Require nonnegative finite postings')
                for owner in reverse[resident.ids[int(doc)]]:
                    maximum[owner]=max(maximum.get(owner,0.),value)
            self.bounds[term]=maximum
        self.initialization_seconds=time.perf_counter()-start

    def contains(self,query,candidate,k):
        if not isinstance(query,str) or not query.strip() or candidate not in self.positions or type(k)!=int or not 1<=k<=len(self.owners):
            raise ValueError('Invalid query, candidate or budget')
        scores=self.resident.selected_scores(query,self.sources[candidate])
        threshold=max(scores.values(),default=0.)
        if not math.isfinite(threshold):raise ValueError('Nonfinite score')
        if threshold<=0:return {'accepted':False,'possible_competitors':0,'candidate_score':threshold}
        upper=np.zeros(len(self.owners),dtype=np.float64)
        for term in sorted(set(tokens(query))):
            for owner,value in self.bounds.get(term,{}).items():
                upper[owner]=np.nextafter(upper[owner]+value,np.inf)
        if not np.isfinite(upper).all():raise ValueError('Nonfinite upper bound')
        possible=[e for e in self.owners if e!=candidate and upper[self.positions[e]]>=threshold]
        source_ids=sorted({sid for e in possible for sid in self.sources[e]})
        exact=self.resident.selected_scores(query,source_ids)
        ahead=0
        for eid in possible:
            value=max((exact[sid] for sid in self.sources[eid]),default=0.)
            if value>threshold or (value==threshold and eid<candidate):ahead+=1
        return {'accepted':ahead<k,'possible_competitors':len(possible),'candidate_score':threshold}
