"""Opponent-adjusted form: not just results, but results against the
strength model's own expectation for that specific opponent, at that
specific point in time."""

import math

import pandas as pd

from backend.analysis.strength import fit, predict_xg
from backend.analysis.validation import match_outcome_probabilities

FORM_MATCHES = 5
DEFAULT_HALF_LIFE_DAYS = 180.0  # Task 5's chosen production value
DEFAULT_PENALTY = 10.0


def expected_points(ratings, home: str, away: str) -> tuple[float, float]:
    """Expected league points for both sides of one match, from the
    strength model's predicted xG run through the Poisson scoreline
    conversion: 3 points for a win, 1 for a draw, weighted by how likely
    each outcome is - not just "this side was favoured"."""
    home_xg, away_xg = predict_xg(ratings, home, away)
    p_home, p_draw, p_away = match_outcome_probabilities(home_xg, away_xg)
    return 3 * p_home + p_draw, 3 * p_away + p_draw


def form(
    team: str,
    history: pd.DataFrame,
    date,
    half_life_days: float | None = DEFAULT_HALF_LIFE_DAYS,
    penalty: float = DEFAULT_PENALTY,
) -> float:
    """Points won minus expected points, summed over the team's last five
    league matches strictly before `date`. Fewer than five prior matches:
    the sum of those. None at all: NaN.

    Each match's expected points come from ratings fitted on history
    strictly before THAT match - not from today's ratings - so an old
    result is judged by what was knowable at the time, never by
    hindsight from how an opponent's form later turned out."""
    cutoff = pd.Timestamp(date)
    dates = pd.to_datetime(history["date"])
    is_team = (history["home_team"] == team) | (history["away_team"] == team)
    prior = history[is_team & (dates < cutoff)].copy()
    prior["_sort_date"] = pd.to_datetime(prior["date"])
    prior = prior.sort_values("_sort_date").tail(FORM_MATCHES)

    if prior.empty:
        return math.nan

    total = 0.0
    for _, row in prior.iterrows():
        training_history = history[pd.to_datetime(history["date"]) < row["_sort_date"]]
        ratings = fit(training_history, half_life_days=half_life_days, penalty=penalty)

        home_expected, away_expected = expected_points(ratings, row["home_team"], row["away_team"])
        if row["home_team"] == team:
            expected, actual_goals_for, actual_goals_against = home_expected, row["home_goals"], row["away_goals"]
        else:
            expected, actual_goals_for, actual_goals_against = away_expected, row["away_goals"], row["home_goals"]

        actual = _points(actual_goals_for, actual_goals_against)
        total += actual - expected

    return total


def _points(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0
