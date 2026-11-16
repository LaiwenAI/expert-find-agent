"""Query-association lift with identification bounds for missing judgments.

This complements absolute relevance; it is not a replacement for user utility
or a claim that chance correction is new statistical theory.
"""
import math
from collections import defaultdict


def _exposure(rankings,k,discount):
    if type(k)!=int or k<1 or len(rankings)<2:raise ValueError('Require k>=1 and at least two queries')
    weights=[1/math.log2(i+2) if discount else 1. for i in range(k)];normalizer=sum(weights)
    rows={}
    for qid,ranking in rankings.items():
        if not isinstance(qid,str) or not qid or not isinstance(ranking,(list,tuple)):
            raise ValueError('Invalid query or ranking')
        if any(not isinstance(eid,str) or not eid for eid in ranking) or len(set(ranking))!=len(ranking):
            raise ValueError('Require unique nonempty expert IDs')
        rows[qid]={eid:weights[i]/normalizer for i,eid in enumerate(ranking[:k])}
    return rows


def _coefficients(rankings,k,discount):
    rows=_exposure(rankings,k,discount);qids=sorted(rows);experts=set().union(*(set(r) for r in rows.values()))
    means={eid:math.fsum(rows[q].get(eid,0.) for q in qids)/len(qids) for eid in experts}
    coefficients={}
    for qid in qids:
        for eid in experts:
            difference=rows[qid].get(eid,0.)-means[eid]
            if abs(difference)>1e-15:coefficients[(qid,eid)]=difference/len(qids)
    return coefficients,rows,means


def _bounds(coefficients,judgments):
    known=0.;lower_unknown=0.;upper_unknown=0.;unjudged=0
    for (qid,eid),coefficient in coefficients.items():
        value=judgments.get(qid,{}).get(eid)
        if value is None:
            unjudged+=1;lower_unknown+=min(0.,coefficient);upper_unknown+=max(0.,coefficient)
        else:
            if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1:
                raise ValueError('Judgments must be bounded numeric utility or None; no implicit unknown-to-zero mapping')
            known+=coefficient*value
    return dict(lower=known+lower_unknown,upper=known+upper_unknown,identified=unjudged==0,
                unjudged_nonzero_cells=unjudged)


def query_lift(rankings,judgments,k=5,discount=False):
    """Actual weighted gain minus its expectation under uniform query permutation.

    Complete judgments: the expectation uses expert relevance marginals and
    the method's own exposure marginals. Missing judgments: return exact box
    bounds, never impute nonrelevance. Constant rankings have lift exactly zero
    for every label completion, even when their absolute precision is high.
    """
    coefficients,rows,means=_coefficients(rankings,k,discount)
    result=_bounds(coefficients,judgments)
    observed={(q,eid):weight/len(rows) for q,r in rows.items() for eid,weight in r.items()}
    shuffled={(q,eid):weight/len(rows) for q in rows for eid,weight in means.items()}
    return dict(**result,observed_gain=_bounds(observed,judgments),query_shuffle_gain=_bounds(shuffled,judgments),
                queries=len(rows),k=k,discount=discount,
                note='Identification bounds, not statistical confidence intervals. Conditional on this query distribution and candidate universe. Lift measures query association, not total relevance; report absolute gain alongside it.')


def comparison_coefficients(first,second,k=5,discount=False,metric='lift'):
    """Linear coefficients for first minus second, without inspecting labels."""
    if set(first)!=set(second):raise ValueError('Require the same query set')
    if metric not in ('lift','gain'):raise ValueError('Unknown comparison metric')
    if metric=='lift':
        a,_,_=_coefficients(first,k,discount);b,_,_=_coefficients(second,k,discount)
    else:
        def gain_coefficients(run):
            rows=_exposure(run,k,discount)
            return {(q,e):weight/len(rows) for q,row in rows.items() for e,weight in row.items()}
        a,b=gain_coefficients(first),gain_coefficients(second)
    difference={cell:a.get(cell,0.)-b.get(cell,0.) for cell in set(a)|set(b)}
    return {cell:value for cell,value in difference.items() if abs(value)>1e-15}


def compare_lift(first,second,judgments,k=5,discount=False):
    """Bound a paired lift difference using the SAME unknown labels."""
    return _bounds(comparison_coefficients(first,second,k,discount),judgments)


def plan_comparison_judgments(first,second,judgments,budget,k=5,discount=False,metric='lift'):
    """Prioritize labels to minimize worst-case interval width at equal unit cost.

    This is a standard linear-box observation, not new sampling theory. A
    resolved numeric label removes exactly abs(coefficient) from the width.
    Unknown/ambiguous annotations do not resolve a cell and confer no reduction.
    """
    if type(budget)!=int or budget<0:raise ValueError('Require a nonnegative integer budget')
    coefficients=comparison_coefficients(first,second,k,discount,metric)
    bounds=_bounds(coefficients,judgments)  # Also reject invalid known labels.
    unknown=sorted(((cell,value) for cell,value in coefficients.items()
                    if judgments.get(cell[0],{}).get(cell[1]) is None),
                   key=lambda item:(-abs(item[1]),item[0]))
    selected=unknown[:budget]
    return dict(metric=metric,difference='first-minus-second',bounds=bounds,
                requests=[dict(query_id=cell[0],expert_id=cell[1],coefficient=value,
                               width_reduction_if_resolved=abs(value)) for cell,value in selected],
                minimum_remaining_width_if_all_resolved=math.fsum(abs(value) for _,value in unknown[budget:]),
                note='Equal-cost exact numeric judgments assumed. Width is an identification bound, not statistical uncertainty. No hidden labels used for selection. Unknown annotations do not shrink it. Standard coefficient-mass scheduling; no novelty or unbiased-estimator claim.')
