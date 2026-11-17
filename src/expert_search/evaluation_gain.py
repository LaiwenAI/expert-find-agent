"""Task protocol for incomplete, snapshot-scoped judgments; not new metric theory."""
import math


def _gain_interval(ranking,judgments,k,gains):
    """Unknown annotations widen the interval; missing outputs earn no gain.

    Grade0 means judged lack of documented fit in this source snapshot, never
    actual expert inability. None means unjudged and must not become grade0.
    Bounds are conditional on the supplied grades, not semantic confidence.
    """
    if type(k) is not int or k<1 or len(set(ranking))!=len(ranking):raise ValueError('Invalid ranking or cutoff')
    if any(v is not None and (type(v) is not int or not 0<=v<=3) for v in judgments.values()):raise ValueError('Invalid partial-fit grade')
    weights=[1/math.log2(i+2) for i in range(k)];maximum=max(gains.values());denominator=maximum*sum(weights);low=high=0.;known=0
    returned=ranking[:k]
    for i,eid in enumerate(returned):
        grade=judgments.get(eid)
        if grade is None:high+=maximum*weights[i]
        else:
            gain=gains[grade]*weights[i];low+=gain;high+=gain;known+=1
    return dict(lower=low/denominator,upper=high/denominator,judged_positions=known,returned_positions=len(returned),budget=k,
                judged_fraction_of_returned=known/len(returned) if returned else None,
                scope='Discounted documented-method support conditional on snapshot grades; unjudged bounds are not model confidence or actual expert ability.')


def snapshot_gain_interval(ranking,judgments,k=5):
    return _gain_interval(ranking,judgments,k,{g:2**g-1 for g in range(4)})


def actionable_gain_interval(ranking,judgments,k=5):
    """Credit a scoped method connection, not merely shared research vocabulary."""
    return _gain_interval(ranking,judgments,k,{0:0,1:0,2:1,3:1})


def latest_gain_interval(events,deadline):
    """Complete result snapshots (time,lower,upper); latest, never best-ever."""
    if type(deadline) not in (int,float) or not math.isfinite(deadline) or deadline<0:raise ValueError('Invalid deadline')
    previous=-1.;answer=(0.,0.)
    for timestamp,lower,upper in events:
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in (timestamp,lower,upper)) or timestamp<previous or timestamp<0 or not 0<=lower<=upper<=1:raise ValueError('Invalid result snapshot')
        previous=timestamp
        if timestamp<=deadline:answer=(lower,upper)
    return dict(lower=answer[0],upper=answer[1])


def condensed_ndcg(ranking,judgments,k=5):
    """Diagnostic on judged items only, explicitly undefined without positives."""
    snapshot_gain_interval(ranking,judgments,k)
    retained=[eid for eid in ranking if judgments.get(eid) is not None][:k]
    ideal=sorted((v for v in judgments.values() if v is not None),reverse=True)[:k]
    dcg=lambda values:sum((2**g-1)/math.log2(i+2) for i,g in enumerate(values))
    denominator=dcg(ideal)
    return dcg([judgments[e] for e in retained])/denominator if denominator else None
