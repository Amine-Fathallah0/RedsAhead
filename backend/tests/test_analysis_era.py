import numpy as np
import pandas as pd
import pytest

from backend.analysis.era import bootstrap_ci, era_means, permutation_test


def _matches(rows):
    return pd.DataFrame(rows, columns=["manager", "ppda"])


def test_era_means_is_the_mean_of_the_metric_for_that_era():
    matches = _matches([("Klopp", 8.0), ("Klopp", 10.0), ("Klopp", 12.0), ("Slot", 99.0)])

    assert era_means(matches, "ppda", "Klopp") == pytest.approx(10.0)


def test_era_means_drops_cup_matches_with_no_understat_stat():
    # The same lesson Task 7 learned the hard way: a cup/European row has
    # NaN for Understat-only stats, and must not silently drag the mean.
    matches = _matches([("Klopp", 8.0), ("Klopp", 12.0), ("Klopp", float("nan"))])

    assert era_means(matches, "ppda", "Klopp") == pytest.approx(10.0)


def test_era_means_rejects_an_era_with_no_data():
    matches = _matches([("Klopp", 8.0)])

    with pytest.raises(ValueError):
        era_means(matches, "ppda", "Slot")


def _clearly_different_eras():
    # Non-overlapping clusters: Klopp presses hard (low PPDA), Slot does not.
    rng = np.random.default_rng(1)
    klopp = rng.normal(loc=6.0, scale=1.0, size=15)
    slot = rng.normal(loc=16.0, scale=1.0, size=15)
    rows = [("Klopp", float(v)) for v in klopp] + [("Slot", float(v)) for v in slot]
    return _matches(rows)


def _indistinguishable_eras():
    # Same underlying distribution for both labels - any apparent
    # difference is just sampling noise.
    rng = np.random.default_rng(2)
    values = rng.normal(loc=10.0, scale=1.0, size=30)
    rows = [("Klopp", float(v)) for v in values[:15]] + [("Slot", float(v)) for v in values[15:]]
    return _matches(rows)


def test_bootstrap_ci_is_narrow_and_does_not_cross_between_two_clearly_different_eras():
    matches = _clearly_different_eras()
    rng = np.random.default_rng(42)

    klopp_low, klopp_high = bootstrap_ci(matches, "ppda", "Klopp", rng=rng)
    slot_low, slot_high = bootstrap_ci(matches, "ppda", "Slot", rng=rng)

    assert klopp_low < klopp_high
    assert slot_low < slot_high
    # The two eras' intervals don't overlap at all - exactly what "clearly different" should look like.
    assert klopp_high < slot_low


def test_permutation_test_finds_a_real_difference_significant():
    matches = _clearly_different_eras()
    rng = np.random.default_rng(42)

    p_value = permutation_test(matches, "ppda", "Klopp", "Slot", rng=rng)

    assert p_value < 0.05


def test_permutation_test_does_not_flag_two_identical_distributions():
    matches = _indistinguishable_eras()
    rng = np.random.default_rng(42)

    p_value = permutation_test(matches, "ppda", "Klopp", "Slot", rng=rng)

    assert p_value > 0.05


def test_permutation_test_rejects_an_era_with_no_data():
    matches = _matches([("Klopp", 8.0)])

    with pytest.raises(ValueError):
        permutation_test(matches, "ppda", "Klopp", "Slot")
