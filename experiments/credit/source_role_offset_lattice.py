"""Exact constrained token decoding for two topic–activity pairs."""

import numpy as np


def lattice():
    """Each nonbackground label starts once; topic 1 precedes topic 2."""
    states = [(mask, active) for mask in range(16) if not mask & 4 or mask & 1
              for active in range(5) if not active or mask & (1 << (active - 1))]
    destination = {state: i for i, state in enumerate(states)}
    predecessors = [[] for _ in states]
    for source, (mask, active) in enumerate(states):
        predecessors[destination[mask, 0]].append(source)
        if active:
            predecessors[source].append(source)
        for label in range(1, 5):
            bit = 1 << (label - 1)
            if mask & bit or (label == 3 and not mask & 1):
                continue
            predecessors[destination[mask | bit, label]].append(source)
    # The last predecessor indexes a sentinel row with score -infinity.
    width = max(map(len, predecessors))
    incoming = np.array([row + [len(states)] * (width - len(row)) for row in predecessors])
    emissions = np.array([active for _, active in states])
    complete = np.array([mask == 15 for mask, _ in states])
    return incoming, emissions, complete


INCOMING, EMISSIONS, COMPLETE = lattice()


def best_paths(logits):
    """Return the best full path and margins to incomplete and alternate full paths."""
    scores = np.full((len(EMISSIONS) + 1, 2), -np.inf)
    scores[0, 0] = 0.
    back = []
    for row in logits:
        # A token contributes its logit relative to background. All paths
        # assign every token, so dropping the shared background term is exact.
        options = scores[INCOMING].reshape(len(EMISSIONS), -1)
        order = np.argsort(-options, axis=1, kind="stable")[:, :2]
        scores[:-1] = np.take_along_axis(options, order, axis=1) + (row[EMISSIONS] - row[0])[:, None]
        back.append(order)
    endpoints = np.flatnonzero(COMPLETE)
    full = scores[endpoints].ravel()
    winners = np.argsort(-full, kind="stable")[:2]
    value, alternate = full[winners]
    if not np.isfinite(value):
        return (), {}
    incomplete = np.max(scores[:-1][~COMPLETE, 0])
    path = []
    target, rank = endpoints[winners[0] // 2], winners[0] % 2
    for chosen in reversed(back):
        path.append(int(EMISSIONS[target]))
        selected = chosen[target, rank]
        target, rank = INCOMING[target, selected // 2], selected % 2
    return tuple(reversed(path)), dict(presence=float(value - incomplete), alternative=float(value - alternate))
