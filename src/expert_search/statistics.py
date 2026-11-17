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


def clustered_paired_interval(differences, groups, seed=20261116, repeats=10000):
    """Query-weighted mean, resampling whole source-paper clusters."""
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or len(groups) != len(values):
        raise ValueError('Require aligned finite differences and groups')
    if type(repeats) is not int or repeats < 1:
        raise ValueError('Require positive bootstrap repeats')
    grouped = {}
    for value, group in zip(values, groups):
        grouped.setdefault(group, []).append(value)
    sums = np.array([sum(v) for v in grouped.values()])
    counts = np.array([len(v) for v in grouped.values()])
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(sums), size=(repeats, len(sums)))
    samples = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return dict(mean=float(values.mean()), percentile_95=list(map(float, np.quantile(samples, [.025, .975]))),
                queries=len(values), clusters=len(sums), seed=seed, bootstrap_repeats=repeats,
                sampling_unit='whole source paper; query-weighted mean')

