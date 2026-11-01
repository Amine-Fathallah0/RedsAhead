"""The simple predictors the attack/defence model has to beat.

Each one reads only the history it is given. The validation harness decides
what history to pass in, so a baseline can never see the match it predicts."""

import math
from datetime import date as Date

import pandas as pd

ELO_START = 1500
ELO_K = 20
POINTS_WIN, POINTS_DRAW = 3, 1
FORM_MATCHES = 5


def league_average(history: pd.DataFrame) -> float:
    """Mean xG per team-match across every match in the history: each match
    contributes one home xG and one away xG."""
    if history.empty:
        raise ValueError("league_average needs at least one match")
    return float(pd.concat([history["home_xg"], history["away_xg"]]).mean())


def last_five_results(team: str, history: pd.DataFrame, date: Date) -> float:
    """Mean league points per game over the team's last five matches strictly
    before `date`. Fewer than five prior matches: the mean of those. None at
    all: NaN, so the caller can skip the match rather than guess."""
    cutoff = pd.Timestamp(date)
    prior = []
    for row in history.itertuples():
        if pd.Timestamp(row.date) >= cutoff:
            continue
        if row.home_team == team:
            prior.append((row.date, _points(row.home_goals, row.away_goals)))
        elif row.away_team == team:
            prior.append((row.date, _points(row.away_goals, row.home_goals)))

    if not prior:
        return math.nan
    prior.sort(key=lambda item: pd.Timestamp(item[0]))
    last_five = [points for _, points in prior[-FORM_MATCHES:]]
    return sum(last_five) / len(last_five)


def elo_ratings(history: pd.DataFrame) -> dict[str, float]:
    """Elo rating for every team after playing through the history in date
    order. Each team starts at 1500, and a win, draw or loss moves points by
    up to K = 20 toward the result the rating difference predicted."""
    ratings: dict[str, float] = {}
    ordered = history.sort_values("date", kind="stable")
    for row in ordered.itertuples():
        home = ratings.setdefault(row.home_team, ELO_START)
        away = ratings.setdefault(row.away_team, ELO_START)
        expected_home = elo_expected(home, away)
        score_home = _score(row.home_goals, row.away_goals)
        ratings[row.home_team] = home + ELO_K * (score_home - expected_home)
        ratings[row.away_team] = away + ELO_K * ((1 - score_home) - (1 - expected_home))
    return ratings


def elo_expected(rating_a: float, rating_b: float) -> float:
    """Probability that side A scores more than side B, from the two ratings."""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def _points(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return POINTS_WIN
    if goals_for == goals_against:
        return POINTS_DRAW
    return 0


def _score(home_goals: int, away_goals: int) -> float:
    """The home side's result as an Elo score: 1 for a win, 0.5 for a draw, 0 for a loss."""
    if home_goals > away_goals:
        return 1.0
    if home_goals == away_goals:
        return 0.5
    return 0.0
