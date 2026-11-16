"""Indexed source-neighborhood enrichment; standard hypergeometric ranking component."""
import math
from functools import lru_cache


def _logcomb(n,k):
    return math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)


@lru_cache(maxsize=None)
def log_tails(population,profile_size,neighborhood_size):
    """log P(X >= k) under random source sampling, not P(expert is qualified)."""
    if (any(type(x)!=int for x in (population,profile_size,neighborhood_size)) or
        population<1 or not 0<=profile_size<=population or not 0<=neighborhood_size<=population):
        raise ValueError('Invalid source-count universe')
    lo=max(0,neighborhood_size-(population-profile_size));hi=min(profile_size,neighborhood_size)
    result=[0.]*(hi+1);accumulator=-math.inf
    for k in range(hi,lo-1,-1):
        value=_logcomb(profile_size,k)+_logcomb(population-profile_size,neighborhood_size-k)-_logcomb(population,neighborhood_size)
        if accumulator==-math.inf:accumulator=value
        else:
            maximum=max(accumulator,value)
            accumulator=maximum+math.log(math.exp(accumulator-maximum)+math.exp(value-maximum))
        result[k]=min(0.,accumulator)
    result[lo]=0.
    return tuple(result)


class EnrichmentIndex:
    def __init__(self,documents,profiles,depths=(10,50,100,200)):
        self.documents=list(documents);self.profiles={eid:set(ids) for eid,ids in profiles.items()}
        if not self.documents or len(set(self.documents))!=len(self.documents):raise ValueError('Invalid unique source universe')
        universe=set(self.documents)
        if any(not ids or ids-universe for ids in self.profiles.values()):raise ValueError('Invalid source memberships')
        self.depths=sorted({min(len(self.documents),d) for d in depths})
        if not self.depths or self.depths[0]<1:raise ValueError('Invalid neighborhood depths')
        self.postings={pid:[] for pid in self.documents}
        for eid,ids in self.profiles.items():
            for pid in ids:self.postings[pid].append(eid)
        for n in {len(ids) for ids in self.profiles.values()}:
            for depth in self.depths:log_tails(len(self.documents),n,depth)

    def score(self,ranking):
        if len(ranking)!=len(set(ranking)) or any(pid not in self.postings for pid in ranking):
            raise ValueError('Invalid source ranking')
        if len(ranking)<self.depths[-1]:raise ValueError('Incomplete neighborhood ranking')
        counts={eid:0 for eid in self.profiles};scores={eid:0. for eid in self.profiles};witness={}
        for depth,pid in enumerate(ranking[:self.depths[-1]],1):
            for eid in self.postings[pid]:counts[eid]+=1
            if depth in self.depths:
                for eid,profile in self.profiles.items():
                    value=-log_tails(len(self.documents),len(profile),depth)[counts[eid]]
                    if value>scores[eid]:
                        scores[eid]=value;witness[eid]=dict(depth=depth,hits=counts[eid],profile_sources=len(profile))
        return dict(scores=scores,witness=witness,
                    warning='Maximum over fixed depths is a ranking statistic, not a calibrated expertise probability or a multiple-testing certificate.')
