"""Fixed-view ranking bounds for relevance, personal contribution and time."""
import math


DIMENSIONS = ('method_support', 'personal_contribution', 'temporal_eligibility')
STATES = ('supported', 'unsupported', 'unknown')


def attributed_gain_interval(ranking, judgments, k=5):
    """Discount only jointly supported views; bound incompletely judged ones.

    Ranking keys must identify a fixed query, expert, source view and cutoff.
    The supplied judgments are scoped to exactly those views. An unsupported
    dimension means that this view does not satisfy that dimension, not that
    the expert lacks ability. Missing fields are unknown, never negative.
    A supported personal-contribution label requires separate role evidence;
    paper authorship, position and citations alone do not supply this label.
    Short rankings receive zero gain for missing output positions.
    """
    if type(k) is not int or k < 1 or not isinstance(ranking,(list,tuple)):
        raise ValueError('Invalid cutoff or ranking')
    if any(not isinstance(key,str) or not key for key in ranking) or len(set(ranking)) != len(ranking):
        raise ValueError('Ranking must contain unique fixed-view keys')
    if not isinstance(judgments,dict):
        raise ValueError('Invalid judgment mapping')
    for key, row in judgments.items():
        if not isinstance(key,str) or not key or not isinstance(row,dict) or set(row)-set(DIMENSIONS):
            raise ValueError('Invalid fixed-view judgment')
        if any(value not in STATES for value in row.values()):
            raise ValueError('Invalid judgment state')
    weights=[1/math.log2(i+2) for i in range(k)]
    denom=sum(weights);lower=upper=0.;states={'supported':0,'unsupported':0,'unknown':0}
    per_dimension={name:0 for name in DIMENSIONS}
    for i,key in enumerate(ranking[:k]):
        row=judgments.get(key,{})
        dimensions=[row.get(name,'unknown') for name in DIMENSIONS]
        for name,state in zip(DIMENSIONS,dimensions):
            per_dimension[name]+=state!='unknown'
        status='unsupported' if 'unsupported' in dimensions else 'supported' if all(s=='supported' for s in dimensions) else 'unknown'
        states[status]+=1
        if status=='supported':lower+=weights[i];upper+=weights[i]
        elif status=='unknown':upper+=weights[i]
    return {'lower':lower/denom,'upper':upper/denom,'returned':sum(states.values()),'budget':k,
            'joint_status_counts':states,'judged_by_dimension':per_dimension,
            'scope':'Joint fixed-view support bounds conditional on supplied labels. Missing role or date is unknown. Not a confidence interval, ability estimate, or verification of source truth.'}
