"""Training-free bounded aggregation prototype; not a novel threshold theorem.
Certificates concern frozen evaluator scores on a fixed candidate set only.
"""
import math


def select(ids, weights, evaluate, k=5, budget=None):
    if not ids or len(set(ids))!=len(ids) or not 1<=k<=len(ids):
        raise ValueError('Require unique candidate IDs and valid k')
    if not weights or any(not math.isfinite(w) or w<=0 for w in weights):
        raise ValueError('Require finite positive weights')
    total=sum(weights); weights=[w/total for w in weights]
    known={eid:{} for eid in ids}; calls=0; trace=[]
    maximum=len(ids)*len(weights) if budget is None else budget
    if maximum<0: raise ValueError('Negative budget')
    def bounds(eid):
        lower=sum(weights[j]*v for j,v in known[eid].items())
        return lower, lower+sum(w for j,w in enumerate(weights) if j not in known[eid])
    while True:
        intervals={eid:bounds(eid) for eid in ids}
        leaders=sorted(ids,key=lambda eid:(-intervals[eid][0],eid))[:k]
        outsiders=[eid for eid in ids if eid not in leaders]
        # Strict separation avoids false certificates when deterministic tie order matters.
        certified=(not outsiders or min(intervals[eid][0] for eid in leaders)>max(intervals[eid][1] for eid in outsiders))
        done=all(len(known[eid])==len(weights) for eid in ids)
        if certified or done or calls>=maximum:
            return dict(ranking=leaders,certified=certified or done,calls=calls,
                        bounds=intervals,trace=trace,score_semantics='fixed evaluator; candidate-set membership only')
        unresolved=[eid for eid in ids if len(known[eid])<len(weights)]
        eid=max(unresolved,key=lambda x:(intervals[x][1],-ids.index(x)))
        j=max((j for j in range(len(weights)) if j not in known[eid]),key=lambda j:(weights[j],-j))
        value=evaluate(eid,j)
        if not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1:
            raise ValueError('Evaluator must return a finite score in [0,1]')
        known[eid][j]=value;calls+=1
        trace.append(dict(candidate=eid,constraint=j,score=value))
