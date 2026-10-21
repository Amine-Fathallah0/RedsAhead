"""Tests for the attack/defence xG strength model.

Every recovery test follows the same shape: pick a set of "true" ratings,
generate matches with EXACTLY the xG those ratings predict (no noise), fit
on that, and check the fit finds its way back to (something close to) the
truth. If the fit can't recover a model it was literally handed the answer
key for, it has no business running on real, noisy data."""

import itertools
from datetime import date, timedelta

import pandas as pd
import pytest

from backend.analysis.strength import Ratings, fit, predict_xg


def _round_robin_fixtures(teams, rounds, start, step_days=7):
    """Every pair of teams plays every `rounds`, alternating who's home,
    one match every `step_days`."""
    fixtures = []
    day = start
    for round_number in range(rounds):
        for i, (team_a, team_b) in enumerate(itertools.combinations(teams, 2)):
            home, away = (team_a, team_b) if (round_number + i) % 2 == 0 else (team_b, team_a)
            fixtures.append((day, home, away))
            day += timedelta(days=step_days)
    return fixtures


def _simulate_history(true_ratings: Ratings, fixtures) -> pd.DataFrame:
    """Build a match history where every row's xG is exactly what
    `predict_xg` says the true ratings produce - no noise, so recovering
    the truth is purely a question of whether the fit works."""
    rows = []
    for match_date, home, away in fixtures:
        home_xg, away_xg = predict_xg(true_ratings, home, away)
        rows.append({
            "date": match_date,
            "home_team": home,
            "away_team": away,
            "home_xg": home_xg,
            "away_xg": away_xg,
        })
    return pd.DataFrame(rows)


def test_fit_rejects_an_empty_history():
    with pytest.raises(ValueError):
        fit(pd.DataFrame(columns=["date", "home_team", "away_team", "home_xg", "away_xg"]), half_life_days=None, penalty=0.1)


def test_fit_recovers_the_attack_ordering_on_synthetic_data():
    true_ratings = Ratings(
        base=0.3,
        home_advantage=0.2,
        attack={"A": 0.5, "B": 0.0, "C": -0.3},
        defence={"A": 0.0, "B": 0.0, "C": 0.0},
    )
    fixtures = _round_robin_fixtures(["A", "B", "C"], rounds=12, start=date(2025, 1, 1))
    history = _simulate_history(true_ratings, fixtures)

    fitted = fit(history, half_life_days=None, penalty=0.01)

    assert fitted.attack["A"] > fitted.attack["B"] > fitted.attack["C"]


def test_fit_recovers_an_exaggerated_home_advantage_split():
    true_ratings = Ratings(
        base=0.2,
        home_advantage=0.15,
        attack={"A": 0.0, "B": 0.0, "C": 0.0},
        defence={"A": 0.0, "B": 0.0, "C": 0.0},
        home_advantage_adjustment={"A": 0.6, "B": 0.0, "C": 0.0},
    )
    fixtures = _round_robin_fixtures(["A", "B", "C"], rounds=16, start=date(2025, 1, 1))
    history = _simulate_history(true_ratings, fixtures)

    fitted = fit(history, half_life_days=None, penalty=0.01)

    # home_advantage and the per-team adjustments trade off against each
    # other freely (shifting a constant between them changes no prediction),
    # so only their *relative* values are identifiable - the penalty settles
    # on whichever split happens to minimize the adjustments' sum of squares.
    # What the fit should get right either way: A's split is clearly more
    # exaggerated than B's or C's, and B and C - built from identical true
    # adjustments - land close to one another.
    assert fitted.home_advantage_adjustment["A"] - fitted.home_advantage_adjustment["B"] > 0.4
    assert fitted.home_advantage_adjustment["A"] - fitted.home_advantage_adjustment["C"] > 0.4
    assert abs(fitted.home_advantage_adjustment["B"] - fitted.home_advantage_adjustment["C"]) < 0.05


def test_recency_weighting_rates_an_improved_team_higher_than_a_flat_fit():
    # Team A was weak a year ago and has since improved. B is a constant
    # reference opponent throughout, so the change is attributable to A.
    old_ratings = Ratings(base=0.2, home_advantage=0.1, attack={"A": -0.5, "B": 0.0})
    new_ratings = Ratings(base=0.2, home_advantage=0.1, attack={"A": 0.5, "B": 0.0})

    old_fixtures = _round_robin_fixtures(["A", "B"], rounds=10, start=date(2025, 1, 1))
    new_fixtures = _round_robin_fixtures(["A", "B"], rounds=10, start=date(2025, 12, 1))
    history = pd.concat([
        _simulate_history(old_ratings, old_fixtures),
        _simulate_history(new_ratings, new_fixtures),
    ], ignore_index=True)

    # A short half-life should mostly see the recent, improved A. No decay
    # at all averages the two eras together and lands closer to the middle.
    recency_fit = fit(history, half_life_days=20, penalty=0.01)
    flat_fit = fit(history, half_life_days=None, penalty=0.01)

    assert recency_fit.attack["A"] > flat_fit.attack["A"]


def test_larger_penalty_shrinks_every_rating_toward_zero():
    true_ratings = Ratings(
        base=0.3,
        home_advantage=0.2,
        attack={"A": 0.5, "B": 0.0, "C": -0.3},
        defence={"A": 0.2, "B": 0.0, "C": -0.2},
        home_advantage_adjustment={"A": 0.3, "B": 0.0, "C": -0.3},
    )
    fixtures = _round_robin_fixtures(["A", "B", "C"], rounds=12, start=date(2025, 1, 1))
    history = _simulate_history(true_ratings, fixtures)

    light_penalty = fit(history, half_life_days=None, penalty=0.01)
    heavy_penalty = fit(history, half_life_days=None, penalty=5.0)

    def sum_of_squares(ratings: Ratings) -> float:
        return (
            sum(v**2 for v in ratings.attack.values())
            + sum(v**2 for v in ratings.defence.values())
            + sum(v**2 for v in ratings.home_advantage_adjustment.values())
        )

    assert sum_of_squares(heavy_penalty) < sum_of_squares(light_penalty)
