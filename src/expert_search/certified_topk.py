"""Vectorized score intervals followed by exact candidate-LM top-k replay.

An execution optimization of an existing additive ranker. It certifies ordered
top-k for this numerical score, not relevance, entailment, or agent decisions.
"""
import math
from collections import Counter
import numpy as np
from .retrieval import tokens


def certified_topk(model, query, k=5, eligible=None, *, _tokens=None):
    if type(k) is not int or k < 1:
        raise ValueError('Require positive integer k')
    candidates = sorted(model.available if eligible is None else eligible)
    if len(set(candidates)) != len(candidates) or set(candidates)-model.available:
        raise ValueError('Duplicate or unavailable candidate')
    if not candidates:
        return dict(results=[],exact_candidates=0,total_candidates=0,postings_visited=0)
    k = min(k,len(candidates))
    positions = np.asarray([model.positions[e] for e in candidates],dtype=np.int64)
    lower = np.zeros(len(model.experts),dtype=np.float64)
    upper = np.zeros_like(lower)
    active = []
    for term,count in Counter(tokens(query) if _tokens is None else _tokens).items():
        posting = model.postings.get(term)
        if posting is None:
            continue
        if count > 2**53:
            raise ValueError('Query count exceeds exact float64 integer range')
        indices,values = posting
        # The rounded products match count*float(value) in the reference.
        with np.errstate(over='raise',invalid='raise'):
            product = values*count
            lower[indices] = np.nextafter(lower[indices]+product,-np.inf)
            upper[indices] = np.nextafter(upper[indices]+product,np.inf)
        active.append((indices,values,count))
    lo,hi = lower[positions],upper[positions]
    threshold = np.partition(lo,len(lo)-k)[len(lo)-k]
    above = int(np.count_nonzero(lo>threshold))
    # candidates are already lexicographically ordered: ties use that order.
    cutoff = np.flatnonzero(lo==threshold)[k-above-1]
    possible = (hi>threshold) | ((hi==threshold) & (np.arange(len(candidates))<=cutoff))
    selected_slots = np.flatnonzero(possible)
    selected_positions = positions[selected_slots]
    contributions = [[] for _ in selected_slots]
    for indices,values,count in active:
        offsets = np.searchsorted(indices,selected_positions)
        present = offsets < len(indices)
        slots = np.flatnonzero(present)
        slots = slots[indices[offsets[slots]]==selected_positions[slots]]
        for slot in slots:
            contributions[slot].append(count*float(values[offsets[slot]]))
    scores = {candidates[slot]:math.fsum(parts) for slot,parts in zip(selected_slots,contributions)}
    ranked = sorted(scores,key=lambda eid:(-scores[eid],eid))[:k]
    return dict(results=[dict(id=eid,score=scores[eid]) for eid in ranked],
                exact_candidates=len(scores),total_candidates=len(candidates),
                postings_visited=sum(len(i) for i,_,_ in active))
