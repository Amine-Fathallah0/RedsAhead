import math
from datetime import date

import pandas as pd
import pytest

from backend.analysis.baselines import elo_expected, elo_ratings, last_five_results, league_average


def _league(rows):
    """Build a league history in the same shape parse_league_matches returns."""
    return pd.DataFrame(
        rows,
        columns=["date", "home_team", "away_team", "home_goals", "away_goals", "home_xg", "away_xg"],
    )


FOUR_MATCHES = _league([
    (date(2025, 8, 1), "A", "B", 2, 0, 1.5, 0.5),
    (date(2025, 8, 8), "C", "D", 1, 1, 1.0, 1.0),
    (date(2025, 8, 15), "B", "C", 0, 3, 0.8, 2.0),
    (date(2025, 8, 22), "D", "A", 0, 2, 0.4, 1.6),
])


def test_league_average_is_the_mean_xg_per_team_match():
    # Eight team-matches: xG totals 8.8, so the mean is 1.1.
    assert league_average(FOUR_MATCHES) == pytest.approx(1.1)


def test_league_average_rejects_an_empty_history():
    with pytest.raises(ValueError):
        league_average(FOUR_MATCHES.iloc[0:0])


def test_last_five_results_is_the_mean_points_of_the_last_five_matches():
    # A's matches before the query date, oldest first: L, W, D, L, W, W.
    # The oldest (L) is outside the last five. The 09-06 loss and the
    # 09-07 win are not before the query date and must be ignored.
    history = _league([
        (date(2025, 8, 25), "B", "A", 2, 0, 1.0, 1.0),  # A loses: 0 pts, too old
        (date(2025, 9, 1), "A", "B", 1, 0, 1.0, 1.0),   # A wins: 3
        (date(2025, 9, 2), "C", "A", 1, 1, 1.0, 1.0),   # draw: 1
        (date(2025, 9, 3), "A", "D", 0, 2, 1.0, 1.0),   # A loses: 0
        (date(2025, 9, 4), "E", "A", 0, 1, 1.0, 1.0),   # A wins away: 3
        (date(2025, 9, 5), "A", "F", 2, 0, 1.0, 1.0),   # A wins: 3
        (date(2025, 9, 6), "A", "G", 0, 1, 1.0, 1.0),   # same day as query: ignored
        (date(2025, 9, 7), "A", "H", 3, 0, 1.0, 1.0),   # after query: ignored
    ])

    # (3 + 1 + 0 + 3 + 3) / 5 = 2.0. Counting the old loss would give 1.5.
    assert last_five_results("A", history, date(2025, 9, 6)) == pytest.approx(2.0)


def test_last_five_results_uses_what_is_there_when_fewer_than_five_matches():
    # Team A has one prior match, a win: 3 points per game.
    assert last_five_results("A", FOUR_MATCHES, date(2025, 8, 5)) == pytest.approx(3.0)


def test_last_five_results_is_nan_with_no_prior_matches():
    value = last_five_results("A", FOUR_MATCHES, date(2025, 8, 1))

    assert math.isnan(value)


def test_elo_goes_up_after_a_win_and_down_after_a_loss():
    ratings = elo_ratings(_league([(date(2025, 8, 1), "A", "B", 1, 0, 1.0, 1.0)]))

    assert ratings["A"] > 1500
    assert ratings["B"] < 1500


def test_elo_is_zero_sum_across_a_match():
    ratings = elo_ratings(FOUR_MATCHES)

    # Each match moves points from one side to the other, so the total stays at 1500 per team.
    assert sum(ratings.values()) == pytest.approx(1500 * 4)


def test_elo_rates_every_team_in_the_history():
    ratings = elo_ratings(FOUR_MATCHES)

    assert set(ratings) == {"A", "B", "C", "D"}


def test_elo_expected_is_one_half_between_equal_teams():
    assert elo_expected(1500, 1500) == pytest.approx(0.5)


def test_elo_expected_is_about_ninety_per_cent_for_400_points_more():
    # 400 points is ten to one odds: 1 / (1 + 10 ** -1) = 0.909...
    assert elo_expected(1900, 1500) == pytest.approx(10 / 11)
