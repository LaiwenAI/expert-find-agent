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


def time_integrated_gain_interval(events,horizon):
    """Integrate the latest complete-result gain over a fixed time budget.

    Before the first result, gain is zero. Later snapshots replace earlier
    ones, even when quality falls. Bounds reflect unjudged results rather
    than uncertainty from sampling, rating disagreement, or timing noise.
    """
    if type(horizon) not in (int,float) or not math.isfinite(horizon) or horizon<=0:raise ValueError('Invalid horizon')
    events=list(events);latest_gain_interval(events,horizon)
    cursor=0.;lower=upper=0.;current=(0.,0.)
    for timestamp,lo,hi in events:
        if timestamp>horizon:break
        duration=timestamp-cursor
        lower+=duration*current[0];upper+=duration*current[1]
        cursor=timestamp;current=(lo,hi)
    lower+=(horizon-cursor)*current[0];upper+=(horizon-cursor)*current[1]
    return {'lower':lower/horizon,'upper':upper/horizon,'horizon':horizon,
            'scope':'Time average of latest complete snapshot gain; no result before its arrival. Incomplete-label bounds are not confidence intervals or actual expert ability.'}


def condensed_ndcg(ranking,judgments,k=5):
    """Diagnostic on judged items only, explicitly undefined without positives."""
    snapshot_gain_interval(ranking,judgments,k)
    retained=[eid for eid in ranking if judgments.get(eid) is not None][:k]
    ideal=sorted((v for v in judgments.values() if v is not None),reverse=True)[:k]
    dcg=lambda values:sum((2**g-1)/math.log2(i+2) for i,g in enumerate(values))
    denominator=dcg(ideal)
    return dcg([judgments[e] for e in retained])/denominator if denominator else None


def paired_actionable_gain_interval(first,second,judgments,k=3):
    """Tighter shared-label bounds for two rankings of the SAME fixed views.

    Keys must identify identical query/expert/source snapshots in both rankings.
    Different views or unverified cross-language equivalence cannot share labels.
    This is standard linear box-bound arithmetic, not a confidence interval.
    """
    a=actionable_gain_interval(first,judgments,k);b=actionable_gain_interval(second,judgments,k)
    denominator=sum(1/math.log2(i+2) for i in range(k));coefficients={}
    for sign,ranking in [(1,first),(-1,second)]:
        for i,key in enumerate(ranking[:k]):coefficients[key]=coefficients.get(key,0.)+sign/(math.log2(i+2)*denominator)
    low=high=0.;uncertain=cancelled=0
    for key,c in coefficients.items():
        grade=judgments.get(key)
        if grade is None:
            low+=min(0.,c);high+=max(0.,c)
            uncertain+=c!=0;cancelled+=c==0
        elif grade>=2:low+=c;high+=c
    return dict(lower=low,upper=high,independent_bounds=[a['lower']-b['upper'],a['upper']-b['lower']],uncertain_nonzero_coefficients=uncertain,cancelled_shared_unjudged=cancelled,
                scope='First-minus-second gain over shared fixed-view labels; tight only for the stated independent unknown-label box. Not semantic confidence, actual ability or a new bound theory. Do not reuse labels across changed snapshots or unverified translations.')


def paired_time_integrated_actionable_gain_interval(first_events, second_events, judgments, horizon, k=3):
    """Bound the difference of time-averaged gains using shared fixed-view labels.

    Each event is ``(completion_time, ranking)``. A snapshot replaces the
    previous one, including regressions, and no result has zero gain. Unknown
    labels are shared across all appearances of the same fixed view in both
    streams. Returned bounds assume independent binary unknown gains; they are
    not confidence intervals or evidence of individual ability.
    """
    if type(horizon) not in (int, float) or not math.isfinite(horizon) or horizon <= 0:
        raise ValueError('Invalid horizon')
    streams = [list(first_events), list(second_events)]
    gain_streams = []
    for stream in streams:
        previous = -1.; gain_events = []
        for timestamp, ranking in stream:
            if type(timestamp) not in (int, float) or not math.isfinite(timestamp) or timestamp < 0 or timestamp < previous:
                raise ValueError('Invalid result snapshot time')
            previous = timestamp
            interval = actionable_gain_interval(ranking, judgments, k)
            gain_events.append((timestamp, interval['lower'], interval['upper']))
        gain_streams.append(gain_events)
    # Validate labels and cutoff even when both streams are empty.
    actionable_gain_interval([], judgments, k)
    independent = [time_integrated_gain_interval(events, horizon) for events in gain_streams]
    denominator = sum(1 / math.log2(i + 2) for i in range(k))
    coefficients = {}
    for sign, stream in zip((1, -1), streams):
        for index, (timestamp, ranking) in enumerate(stream):
            if timestamp >= horizon:
                break
            end = min(horizon, stream[index + 1][0] if index + 1 < len(stream) else horizon)
            duration = max(0., end - timestamp) / horizon
            for rank, key in enumerate(ranking[:k]):
                coefficients[key] = coefficients.get(key, 0.) + sign * duration / (math.log2(rank + 2) * denominator)
    low = high = 0.
    for key, coefficient in coefficients.items():
        grade = judgments.get(key)
        if grade is None:
            low += min(0., coefficient); high += max(0., coefficient)
        elif grade >= 2:
            low += coefficient; high += coefficient
    return dict(lower=low, upper=high,
                independent_bounds=[independent[0]['lower'] - independent[1]['upper'], independent[0]['upper'] - independent[1]['lower']],
                horizon=horizon, fixed_view_keys=len(coefficients),
                scope='First-minus-second time-averaged gain with shared labels for identical fixed views. Bounds assume independent unknown gains, not sampling uncertainty or actual ability.')
