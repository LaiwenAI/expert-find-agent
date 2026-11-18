"""Bound verifier corrections while retaining an exact ordered Top-K."""

import numpy as np


def adjusted_ranking(base, evaluate_block, *, weight, top=5, block_size=8, exhaust=False):
    """Return dense positions and checks for a fixed block-level verifier.

    The verifier must produce the same margins for a block regardless of
    evaluation order. Blocks use dense order. Dense position breaks ties.
    """
    base = np.asarray(base, dtype=np.float64)
    if (base.ndim != 1 or not np.isfinite(base).all() or np.any(base[1:] > base[:-1])
            or type(top) is not int or not 0 < top <= len(base)
            or type(block_size) is not int or block_size < 1 or not np.isfinite(weight) or weight < 0):
        raise ValueError("Invalid dense scores or ranking parameters")
    if weight == 0:
        return np.arange(top).tolist(), 0

    lower = base - weight
    upper = base + weight
    known = np.zeros(len(base), dtype=bool)
    for start in range(0, len(base), block_size):
        if not exhaust:
            threshold = np.partition(lower, len(base) - top)[len(base) - top]
            if not np.any((upper >= threshold) & ~known):
                break
        end = min(start + block_size, len(base))
        margins = np.asarray(evaluate_block(start, end), dtype=np.float64)
        if margins.shape != (end - start,) or not np.isfinite(margins).all():
            raise ValueError("Invalid verifier margins")
        scores = base[start:end] + weight * np.tanh(margins / 4.)
        lower[start:end] = upper[start:end] = scores
        known[start:end] = True

    ranked = np.argsort(-lower, kind="stable")[:top]
    if not known[ranked].all():
        raise AssertionError("Top-K contains unresolved scores")
    return ranked.tolist(), int(known.sum())


def normalized_dense(scores):
    values = np.asarray(scores, dtype=np.float64)
    return (values - values[-1]) / max(float(values[0] - values[-1]), 1e-7)
