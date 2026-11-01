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


@pytest.mark.integration
def test_klopp_and_slot_eras_are_not_distinguishable_on_real_xg():
    # Klopp's last ~2 seasons (390 league/cup matches) and Slot's full
    # tenure so far (113 matches) produced almost the same average xG per
    # match - a real example of eras that genuinely aren't different, not
    # the easy "obviously different" case the unit tests construct.
    from backend.analysis.data import load_matches
    from backend.config import load_settings

    matches = load_matches(load_settings().db_path)

    klopp_mean = era_means(matches, "xg", "Klopp")
    slot_mean = era_means(matches, "xg", "Slot")
    klopp_low, klopp_high = bootstrap_ci(matches, "xg", "Klopp", rng=np.random.default_rng(0))

    assert klopp_low < klopp_mean < klopp_high
    assert 1.5 < klopp_mean < 3.0
    assert 1.5 < slot_mean < 3.0

    p_value = permutation_test(matches, "xg", "Klopp", "Slot", rng=np.random.default_rng(0))
    assert p_value > 0.05  # the two means are close enough that this isn't a surprise
