"""Lightweight paired query bootstrap, independent of model clients."""
import numpy as np


def paired_interval(differences, seed=20261116, repeats=10000):
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Require finite nonempty paired differences')
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(len(values), size=(repeats, len(values)))].mean(axis=1)
    return dict(mean=float(values.mean()), percentile_95=list(map(float, np.quantile(samples, [.025, .975]))),
                queries=len(values), seed=seed, bootstrap_repeats=repeats)

