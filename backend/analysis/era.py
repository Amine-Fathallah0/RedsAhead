"""Comparing managers' eras: not just raw averages, but how confident we
can be they're actually different and not sampling noise."""

import numpy as np
import pandas as pd

DEFAULT_RESAMPLES = 2000
DEFAULT_CONFIDENCE = 0.95


def _era_values(matches: pd.DataFrame, metric: str, manager: str) -> np.ndarray:
    """Every non-missing value of `metric` for that manager's era. Cup and
    European matches have no Understat stats (NaN), the same issue Task 7
    found - dropped here before anything is computed, not after."""
    values = matches.loc[matches["manager"] == manager, metric].dropna().to_numpy(dtype=float)
    if len(values) == 0:
        raise ValueError(f"no {metric!r} data for {manager!r}")
    return values


def era_means(matches: pd.DataFrame, metric: str, manager: str) -> float:
    return float(_era_values(matches, metric, manager).mean())


def bootstrap_ci(
    matches: pd.DataFrame,
    metric: str,
    manager: str,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """A confidence interval for the era's mean, built by resampling the
    era's own matches with replacement rather than assuming a shape for
    the underlying distribution."""
    rng = rng if rng is not None else np.random.default_rng()
    values = _era_values(matches, metric, manager)

    resampled_means = np.array([
        rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_resamples)
    ])
    alpha = (1 - confidence) / 2
    low, high = np.quantile(resampled_means, [alpha, 1 - alpha])
    return float(low), float(high)


def permutation_test(
    matches: pd.DataFrame,
    metric: str,
    manager_a: str,
    manager_b: str,
    n_permutations: int = DEFAULT_RESAMPLES,
    rng: np.random.Generator | None = None,
) -> float:
    """A two-sided p-value for whether the two eras' means genuinely
    differ. Pools both eras' values, repeatedly reshuffles who "belongs"
    to which era, and asks how often a reshuffle produces a difference at
    least as large as the one actually observed - if real labels aren't
    special, most reshuffles should look similar to the real split."""
    values_a = _era_values(matches, metric, manager_a)
    values_b = _era_values(matches, metric, manager_b)
    rng = rng if rng is not None else np.random.default_rng()

    observed = abs(values_a.mean() - values_b.mean())
    pooled = np.concatenate([values_a, values_b])
    split_point = len(values_a)

    at_least_as_extreme = 0
    for _ in range(n_permutations):
        shuffled = rng.permutation(pooled)
        permuted_diff = abs(shuffled[:split_point].mean() - shuffled[split_point:].mean())
        if permuted_diff >= observed:
            at_least_as_extreme += 1

    # +1 on both sides: the observed split is itself one possible permutation,
    # and this avoids ever reporting an impossible p-value of exactly 0.
    return (at_least_as_extreme + 1) / (n_permutations + 1)
