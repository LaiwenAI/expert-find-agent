"""Constraint-evidence protocol prototype, not a claim of new metric theory."""
import math


def evidence_gain(constraints, judgments):
    """Only externally verified support counts; unsupported mandatory items gate full fit.

    constraints: {id: {weight:positive, mandatory:bool}}
    judgments: {id: supported|contradicted|unknown}, absent means unknown.
    """
    if not constraints:raise ValueError('No constraints')
    if any(not math.isfinite(c['weight']) or c['weight']<=0 for c in constraints.values()):raise ValueError('Invalid weights')
    if any(v not in ['supported','contradicted','unknown'] for v in judgments.values()):raise ValueError('Invalid judgment')
    total=sum(c['weight'] for c in constraints.values())
    supported=sum(c['weight'] for i,c in constraints.items() if judgments.get(i)=='supported')/total
    required=[i for i,c in constraints.items() if c.get('mandatory',False)]
    full=all(judgments.get(i)=='supported' for i in required)
    return dict(supported_coverage=supported,mandatory_satisfied=full,
        gated_gain=supported if full else 0.,
        unknown_weight=sum(c['weight'] for i,c in constraints.items() if judgments.get(i,'unknown')=='unknown')/total)


def deadline_gain(events, deadline):
    """Events are complete, externally scored result snapshots (time_seconds,gain).
    Latest snapshot is used, not the best-ever gain, so regressions are visible.
    """
    if deadline<0:raise ValueError('Invalid deadline')
    last=-1; gain=0
    for time,value in events:
        if not math.isfinite(time) or time<0 or time<last or not math.isfinite(value) or not 0<=value<=1:raise ValueError('Invalid event')
        last=time
        if time<=deadline:gain=value
    return gain
