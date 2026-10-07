"""The held-out validation harness that keeps the gate honest.

Every prediction is made with only the matches dated before it - never the
match itself, and never anything later. This file also carries the one
conversion every baseline and the strength model are judged through when the
question is "who wins", not "how many goals": the Poisson scoreline."""

import math

import pandas as pd

MAX_GOALS = 15  # scorelines past this are vanishingly unlikely and add noise, not signal


def held_out_predictions(history: pd.DataFrame, predictor) -> pd.DataFrame:
    """Run `predictor(training_history, home, away)` once per match in
    `history`, in date order, handing it only the matches strictly before
    the one it is predicting. Returns one row per match: the actual and
    predicted xG for both sides."""
    ordered = history.sort_values("date", kind="stable").reset_index(drop=True)
    rows = []
    for i, row in ordered.iterrows():
        training_history = ordered.iloc[:i]
        predicted_home, predicted_away = predictor(training_history, row["home_team"], row["away_team"])
        for side, actual_xg, predicted_xg in [
            ("home", row["home_xg"], predicted_home),
            ("away", row["away_xg"], predicted_away),
        ]:
            rows.append({
                "date": row["date"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "side": side,
                "actual_xg": actual_xg,
                "predicted_xg": predicted_xg,
            })
    return pd.DataFrame(rows)


def rmse(predictions: pd.DataFrame) -> float:
    """Root-mean-squared error between predicted and actual xG."""
    errors = predictions["predicted_xg"] - predictions["actual_xg"]
    return float((errors**2).mean() ** 0.5)


def _poisson_pmf(k: int, rate: float) -> float:
    return math.exp(-rate) * rate**k / math.factorial(k)


def match_outcome_probabilities(home_xg: float, away_xg: float) -> tuple[float, float, float]:
    """Turn two predicted xG numbers into (P(home win), P(draw), P(away win)).

    Treats each side's xG as the rate of a Poisson distribution over goals
    scored, assumes the two sides score independently, and sums every
    scoreline's probability by who it favours. No fitted parameters: the
    xG numbers are the only inputs."""
    p_home = p_draw = p_away = 0.0
    for home_goals in range(MAX_GOALS):
        p_home_goals = _poisson_pmf(home_goals, home_xg)
        for away_goals in range(MAX_GOALS):
            p = p_home_goals * _poisson_pmf(away_goals, away_xg)
            if home_goals > away_goals:
                p_home += p
            elif home_goals == away_goals:
                p_draw += p
            else:
                p_away += p
    return p_home, p_draw, p_away


def brier_score(predictions: pd.DataFrame) -> float:
    """Mean squared error between a stated (p_home, p_draw, p_away)
    probability triple and the one-hot actual result. 0 is a perfectly
    confident correct call; higher is worse, and a confident wrong call
    costs more than an unsure one."""
    total = 0.0
    for row in predictions.itertuples():
        actual = {"home": (1, 0, 0), "draw": (0, 1, 0), "away": (0, 0, 1)}[row.result]
        total += (row.p_home - actual[0]) ** 2 + (row.p_draw - actual[1]) ** 2 + (row.p_away - actual[2]) ** 2
    return total / len(predictions)
