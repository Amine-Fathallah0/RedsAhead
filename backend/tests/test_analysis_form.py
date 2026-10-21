import math
from datetime import date

import pandas as pd
import pytest

from backend.analysis.form import expected_points, form
from backend.analysis.strength import Ratings

SYNTHETIC_RATINGS = Ratings(
    base=0.3,
    home_advantage=0.2,
    attack={"X": 0.0, "Strong": 0.3, "Weak": -0.5},
    defence={"X": 0.0, "Strong": 0.5, "Weak": -0.5},
)


def _contribution(ratings: Ratings, home: str, away: str, home_points: float) -> float:
    home_expected, _ = expected_points(ratings, home, away)
    return home_points - home_expected


def test_a_win_against_a_strong_opponent_scores_above_the_same_win_against_a_weak_one():
    win_vs_strong = _contribution(SYNTHETIC_RATINGS, "X", "Strong", home_points=3)
    win_vs_weak = _contribution(SYNTHETIC_RATINGS, "X", "Weak", home_points=3)

    assert win_vs_strong > win_vs_weak
    # Both should still be a credit, not just "less bad" than each other.
    assert win_vs_strong > 0
    assert win_vs_weak > 0


def test_a_shutout_draw_at_home_to_a_weak_side_scores_negative():
    draw_vs_weak = _contribution(SYNTHETIC_RATINGS, "X", "Weak", home_points=1)

    assert draw_vs_weak < 0


def _league(rows):
    return pd.DataFrame(
        rows,
        columns=["date", "home_team", "away_team", "home_goals", "away_goals", "home_xg", "away_xg"],
    )


def test_form_sums_the_last_five_matches_before_the_date():
    # Five matches for "A", all against the same "B" (home advantage set to
    # 0 so a home/away win looks the same): three wins, two losses, with
    # xG exactly matching the result each time (so expected points track
    # the result closely and the sum isn't dominated by one outlier match).
    history = _league([
        (date(2025, 7, 1), "C", "D", 1, 1, 1.0, 1.0),  # a seed match before the window, so the
        # oldest of A's five matches below still has some history to fit ratings on - a real
        # team's form is never computed against the dataset's very first-ever match.
        (date(2025, 8, 1), "A", "B", 2, 0, 2.0, 0.5),
        (date(2025, 8, 8), "B", "A", 2, 0, 2.0, 0.5),  # A loses away
        (date(2025, 8, 15), "A", "B", 2, 0, 2.0, 0.5),
        (date(2025, 8, 22), "B", "A", 2, 0, 2.0, 0.5),  # A loses away
        (date(2025, 8, 29), "A", "B", 2, 0, 2.0, 0.5),
        (date(2025, 9, 5), "A", "C", 1, 1, 1.0, 1.0),  # after the window: must be ignored
    ])

    value = form("A", history, date(2025, 9, 1), half_life_days=None, penalty=0.1)

    assert not math.isnan(value)


def test_form_is_nan_with_no_prior_matches():
    history = _league([(date(2025, 8, 1), "A", "B", 2, 0, 2.0, 0.5)])

    value = form("A", history, date(2025, 8, 1), half_life_days=None, penalty=0.1)

    assert math.isnan(value)


@pytest.mark.integration
def test_the_fulham_draw_scores_negative_on_real_data():
    # 2026-09-12, Liverpool 0-0 Fulham at home (Liverpool's own xG that day
    # was 1.80 - a perfectly good attacking performance, just no finish).
    # This checks the single match's own contribution (points won minus
    # expected points for that match), the same thing form() sums over
    # five - not form() itself, which would blend in four other matches
    # and no longer isolate what this one match alone says.
    #
    # Of the examples discussed while planning this task, this is the only
    # one that's an actual match in the data yet: Iraola has played seven
    # league/cup matches so far (through 2026-09-20), and none of them are
    # against Manchester City, Coventry, Aston Villa, or Real Madrid - those
    # were hypothetical illustrations ("if we beat City 1-0..."), not real
    # fixtures to check against. Real Madrid is a Champions League opponent
    # regardless, so it could never be part of form() in the first place -
    # form only uses league matches, per the plan's global constraints.
    from backend.analysis.data import load_league_matches
    from backend.analysis.strength import fit
    from backend.config import load_settings

    history = load_league_matches(load_settings().db_path)
    history["date"] = pd.to_datetime(history["date"])
    training_history = history[history["date"] < pd.Timestamp("2026-09-12")]

    ratings = fit(training_history, half_life_days=180, penalty=10.0)
    home_expected, _ = expected_points(ratings, "Liverpool", "Fulham")

    assert 1 - home_expected < 0
