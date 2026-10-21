"""THE GATE (Task 5). The attack/defence strength model must beat every
baseline on real, held-out Premier League matches, judged on the question
each baseline actually answers - not forced onto a shared metric that
suits one of them and not the others.

half_life_days=180, penalty=10.0 were chosen by a two-round grid search
(docs/superpowers/plans/2026-10-05-redsahead-analysis.md, Task 5): round 1
covered half-life in {14, 30, 60, 120, unlimited} and penalty in
{0.01, 0.1, 1.0, 5.0}, and penalty kept improving all the way to 5.0 - its
upper boundary, so round 2 extended it. Round 2 ({120, 180, 250} days x
{5, 10, 20, 50} penalty) found a genuine interior optimum on both axes:
180 days beat both its neighbors, and penalty peaked at 10 at every
half-life tried, worse to either side. The improvement over round 1's
pick (120, 5.0) was marginal (xG RMSE 0.8224 vs 0.8253), so a third round
wasn't worth running.

This test refits once per distinct match date, not once per match -
several Premier League matches share a date, and (the harness's own
leakage fix) they all correctly get the same training history anyway, so
refitting per match would just redo identical work."""

import pandas as pd
import pytest

from backend.analysis.baselines import elo_expected, elo_ratings, last_five_results, league_average
from backend.analysis.data import load_league_matches
from backend.analysis.strength import fit, predict_xg
from backend.analysis.validation import match_outcome_probabilities, rmse
from backend.config import load_settings

pytestmark = pytest.mark.integration

HOLD_OUT_START = pd.Timestamp("2024-08-01")  # 2024/25 onward: Premier League's close season (June-July) makes this unambiguous from the date alone
HALF_LIFE_DAYS = 180
PENALTY = 10.0


def _actual_score(goals_for: int, goals_against: int) -> float:
    if goals_for > goals_against:
        return 1.0
    if goals_for == goals_against:
        return 0.5
    return 0.0


def _actual_points(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def _rmse(df: pd.DataFrame, actual_col: str, predicted_col: str) -> float:
    subset = df[[actual_col, predicted_col]].dropna()
    renamed = subset.rename(columns={actual_col: "actual_xg", predicted_col: "predicted_xg"})
    return rmse(renamed)


def test_the_strength_model_beats_every_baseline_on_real_premier_league_matches():
    history = load_league_matches(load_settings().db_path)
    history = history.copy()
    history["date"] = pd.to_datetime(history["date"])
    history = history.sort_values("date", kind="stable").reset_index(drop=True)
    held_out = history[history["date"] >= HOLD_OUT_START]
    dates = sorted(held_out["date"].unique())

    model_rows, baseline_rows = [], []
    for d in dates:
        training_history = history[history["date"] < d]
        todays = held_out[held_out["date"] == d]
        if training_history.empty:
            continue

        ratings = fit(training_history, half_life_days=HALF_LIFE_DAYS, penalty=PENALTY)
        league_avg = league_average(training_history)
        elo = elo_ratings(training_history)

        for _, m in todays.iterrows():
            pred_home, pred_away = predict_xg(ratings, m["home_team"], m["away_team"])
            p_home, p_draw, p_away = match_outcome_probabilities(pred_home, pred_away)

            home_score = _actual_score(m["home_goals"], m["away_goals"])
            home_pts = _actual_points(m["home_goals"], m["away_goals"])
            away_pts = _actual_points(m["away_goals"], m["home_goals"])
            elo_home_score = elo_expected(elo.get(m["home_team"], 1500.0), elo.get(m["away_team"], 1500.0))
            last_five_home = last_five_results(m["home_team"], training_history, d)
            last_five_away = last_five_results(m["away_team"], training_history, d)

            model_rows.append({
                "actual_xg": m["home_xg"], "predicted_xg": pred_home,
                "actual_score": home_score, "expected_score": p_home + 0.5 * p_draw,
                "actual_pts": home_pts, "expected_pts": 3 * p_home + p_draw,
            })
            model_rows.append({
                "actual_xg": m["away_xg"], "predicted_xg": pred_away,
                "actual_score": 1.0 - home_score, "expected_score": p_away + 0.5 * p_draw,
                "actual_pts": away_pts, "expected_pts": 3 * p_away + p_draw,
            })
            baseline_rows.append({
                "actual_xg": m["home_xg"], "league_avg_xg": league_avg,
                "actual_score": home_score, "elo_score": elo_home_score,
                "actual_pts": home_pts, "last_five_pts": last_five_home,
            })
            baseline_rows.append({
                "actual_xg": m["away_xg"], "league_avg_xg": league_avg,
                "actual_score": 1.0 - home_score, "elo_score": 1.0 - elo_home_score,
                "actual_pts": away_pts, "last_five_pts": last_five_away,
            })

    model_df = pd.DataFrame(model_rows)
    baseline_df = pd.DataFrame(baseline_rows)

    model_xg_rmse = _rmse(model_df, "actual_xg", "predicted_xg")
    league_avg_rmse = _rmse(baseline_df, "actual_xg", "league_avg_xg")
    model_outcome_rmse = _rmse(model_df, "actual_score", "expected_score")
    elo_rmse = _rmse(baseline_df, "actual_score", "elo_score")
    model_points_rmse = _rmse(model_df, "actual_pts", "expected_pts")
    last_five_rmse = _rmse(baseline_df, "actual_pts", "last_five_pts")

    # xG accuracy: the model's own currency, against the simplest baseline.
    assert model_xg_rmse < league_avg_rmse
    # Outcome accuracy vs Elo, on Elo's native scale (expected score, not a 3-way split).
    assert model_outcome_rmse < elo_rmse
    # Outcome accuracy vs last-five, on last-five's native scale (points per game).
    assert model_points_rmse < last_five_rmse
