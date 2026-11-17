"""Standard rank-fusion controls for independently retrieved source channels.

RRF is inherited from Cormack et al., SIGIR2009. Neither fusion nor interleaving
establishes semantic relevance. Missing a channel is missing evidence, not proof
of inability; fusion can nevertheless favor experts represented in both channels.
"""
import math


def fuse(rankings,k=10,depth=50,method='rrf',constant=60):
    if not isinstance(rankings,dict) or not rankings or type(k) is not int or k<1 or type(depth) is not int or depth<1 or method not in ('rrf','interleave'):
        raise ValueError('Invalid channel-fusion configuration')
    if type(constant) not in (int,float) or not math.isfinite(constant) or constant<0:
        raise ValueError('Invalid RRF constant')
    truncated={}
    for channel,ranking in rankings.items():
        if not isinstance(channel,str) or not channel or not isinstance(ranking,(list,tuple)) or len(set(ranking))!=len(ranking) or any(not isinstance(e,str) or not e for e in ranking):
            raise ValueError('Invalid channel IDs or duplicate candidate')
        truncated[channel]=ranking[:depth]
    positions={}
    for channel,ranking in truncated.items():
        for i,eid in enumerate(ranking,1):positions.setdefault(eid,{})[channel]=i
    if method=='rrf':
        scores={eid:sum(1/(constant+rank) for rank in ranks.values()) for eid,ranks in positions.items()}
        selected=sorted(scores,key=lambda eid:(-scores[eid],eid))[:k]
    else:
        selected=[]
        for i in range(depth):
            for ranking in truncated.values():
                if i<len(ranking) and ranking[i] not in selected:selected.append(ranking[i])
                if len(selected)==k:break
            if len(selected)==k:break
        scores={eid:None for eid in positions}
    return [dict(id=eid,fusion_score=scores[eid],channel_ranks=positions[eid]) for eid in selected]
