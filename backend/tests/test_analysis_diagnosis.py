import math
from datetime import date

import pandas as pd
import pytest

from backend.analysis.diagnosis import era_baseline, z_scores


def _matches(rows):
    return pd.DataFrame(rows, columns=["date", "manager", "xg", "ppda", "deep"])


def test_era_baseline_is_none_with_fewer_than_five_prior_same_era_matches():
    matches = _matches([
        (date(2025, 8, 1), "Klopp", 1.5, 10.0, 8),
        (date(2025, 8, 8), "Klopp", 1.5, 10.0, 8),
        (date(2025, 8, 15), "Klopp", 1.5, 10.0, 8),
        (date(2025, 8, 22), "Klopp", 1.5, 10.0, 8),
        (date(2025, 8, 29), "Klopp", 1.5, 10.0, 8),  # the match being diagnosed
    ])

    assert era_baseline(matches, date(2025, 8, 29)) is None


def test_era_baseline_takes_the_last_ten_same_era_matches_and_no_others():
    # 12 same-era matches before the one being diagnosed, plus a more
    # recent-looking row from a different era that must never appear -
    # era, not recency alone, decides what belongs in the window.
    rows = []
    for i in range(12):
        rows.append((date(2025, 1, 1 + i), "Klopp", float(i), 10.0, 8))
    rows.append((date(2025, 2, 1), "Slot", 99.0, 99.0, 99))  # different era, must be excluded
    diagnosed_date = date(2025, 2, 2)
    rows.append((diagnosed_date, "Klopp", 5.0, 10.0, 8))
    matches = _matches(rows)

    baseline = era_baseline(matches, diagnosed_date)

    assert len(baseline) == 10
    assert (baseline["manager"] == "Klopp").all()
    # The last 10 of the 12 Klopp matches: xg values 2.0 through 11.0.
    assert sorted(baseline["xg"].tolist()) == [float(v) for v in range(2, 12)]


def test_z_scores_matches_a_hand_worked_example():
    # Baseline PPDA: 8, 9, 10, 11, 12 -> mean 10, sample std 1.5811.
    rows = [
        (date(2025, 1, 1), "Klopp", 1.0, 8.0, 8),
        (date(2025, 1, 2), "Klopp", 1.0, 9.0, 8),
        (date(2025, 1, 3), "Klopp", 1.0, 10.0, 8),
        (date(2025, 1, 4), "Klopp", 1.0, 11.0, 8),
        (date(2025, 1, 5), "Klopp", 1.0, 12.0, 8),
        (date(2025, 1, 6), "Klopp", 1.0, 15.0, 8),  # the match being diagnosed: PPDA jumped to 15
    ]
    matches = _matches(rows)

    scores = z_scores(matches, date(2025, 1, 6))

    assert scores["ppda"] == pytest.approx((15.0 - 10.0) / 1.5811, rel=1e-3)


def test_z_scores_ranks_the_largest_absolute_deviation_first():
    rows = [
        (date(2025, 1, 1), "Klopp", 1.5, 10.0, 8),
        (date(2025, 1, 2), "Klopp", 1.5, 10.0, 8),
        (date(2025, 1, 3), "Klopp", 1.5, 10.0, 8),
        (date(2025, 1, 4), "Klopp", 1.5, 10.0, 8),
        (date(2025, 1, 5), "Klopp", 1.5, 10.0, 8),
        # PPDA barely moves (small deviation); xG collapses (large deviation).
        (date(2025, 1, 6), "Klopp", 0.1, 10.1, 8),
    ]
    matches = _matches(rows)

    scores = z_scores(matches, date(2025, 1, 6))

    assert list(scores.index)[0] == "xg"


def test_z_scores_is_none_when_the_baseline_is_none():
    matches = _matches([(date(2025, 1, 1), "Klopp", 1.5, 10.0, 8)])

    assert z_scores(matches, date(2025, 1, 1)) is None


def test_era_baseline_excludes_cup_matches_with_no_understat_stats():
    # Understat covers the Premier League only, so a cup match has NaN for
    # xg/ppda/deep. Four real matches plus one NaN cup match raw-counts as
    # five rows, but only four have actual data - below the minimum, so
    # this must still return None, not silently "pass" on padded NaNs.
    matches = _matches([
        (date(2025, 8, 1), "Klopp", 1.5, 10.0, 8),
        (date(2025, 8, 8), "Klopp", 1.5, 10.0, 8),
        (date(2025, 8, 11), "Klopp", float("nan"), float("nan"), float("nan")),  # cup match
        (date(2025, 8, 15), "Klopp", 1.5, 10.0, 8),
        (date(2025, 8, 22), "Klopp", 1.5, 10.0, 8),
        (date(2025, 8, 29), "Klopp", 1.5, 10.0, 8),  # the match being diagnosed
    ])

    assert era_baseline(matches, date(2025, 8, 29)) is None


def test_era_baseline_drops_cup_matches_out_of_an_otherwise_full_window():
    rows = [
        (date(2025, 8, 1), "Klopp", 1.0, 10.0, 8),
        (date(2025, 8, 8), "Klopp", 2.0, 10.0, 8),
        (date(2025, 8, 11), "Klopp", float("nan"), float("nan"), float("nan")),  # cup match
        (date(2025, 8, 15), "Klopp", 3.0, 10.0, 8),
        (date(2025, 8, 22), "Klopp", 4.0, 10.0, 8),
        (date(2025, 8, 29), "Klopp", 5.0, 10.0, 8),
    ]
    diagnosed_date = date(2025, 9, 5)
    rows.append((diagnosed_date, "Klopp", 6.0, 10.0, 8))
    matches = _matches(rows)

    baseline = era_baseline(matches, diagnosed_date)

    assert len(baseline) == 5
    assert not baseline["xg"].isna().any()
    assert sorted(baseline["xg"].tolist()) == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_era_baseline_raises_when_there_is_no_match_on_that_date():
    matches = _matches([(date(2025, 1, 1), "Klopp", 1.5, 10.0, 8)])

    with pytest.raises(ValueError):
        era_baseline(matches, date(2025, 1, 2))
