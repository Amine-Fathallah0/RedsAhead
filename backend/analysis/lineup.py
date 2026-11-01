"""The score-state control: what the scoreline actually was at the moment
each player entered the match, reconstructed from shot-by-shot goal data.
Task 11's ridge model needs this so a substitute who came on to see out a
3-0 win isn't credited (or blamed) for a game state he didn't create."""

import math

import numpy as np
import pandas as pd

from backend.analysis.form import DEFAULT_HALF_LIFE_DAYS, DEFAULT_PENALTY
from backend.analysis.strength import Ratings, fit
from backend.model.names import NameMap


def reconstruct_score(shots: pd.DataFrame, minute: int) -> tuple[int, int]:
    """(home_goals, away_goals) scored at or before `minute`, from one
    match's shots. `shots` must already be filtered to a single match."""
    goals = shots[(shots["result"] == "Goal") & (shots["minute"] <= minute)]
    home_goals = int((goals["side"] == "h").sum())
    away_goals = int((goals["side"] == "a").sum())
    return home_goals, away_goals


def entry_minute(rosters: pd.DataFrame, roster_id) -> int:
    """The match-clock minute a player entered. 0 for a starter.

    A substitute's own `minutes` is how long *he* was on the pitch, not
    his exit clock-minute - only a starter's `minutes` is a clock position,
    because he started at minute 0. So this walks `roster_out` back to a
    starter, summing each predecessor's `minutes` along the way, rather
    than trusting a single hop - real match 7432 (see the tests) is a
    case where a one-hop lookup gives the wrong answer by a wide margin.
    `rosters` must already be filtered to a single match."""
    by_id = rosters.set_index("roster_id")
    return _entry_minute(by_id, roster_id)


def _entry_minute(by_id: pd.DataFrame, roster_id) -> int:
    row = by_id.loc[roster_id]
    predecessor_id = row["roster_out"]
    if pd.isna(predecessor_id):
        return 0
    predecessor = by_id.loc[predecessor_id]
    return _entry_minute(by_id, predecessor_id) + predecessor["minutes"]


def score_state_at_entry(shots: pd.DataFrame, rosters: pd.DataFrame, roster_id) -> int:
    """The player's own team's goal difference at the moment he entered:
    positive if his team was leading, negative if trailing, 0 if level
    (always true for a starter, who enters before kickoff). `shots` and
    `rosters` must already be filtered to a single match."""
    minute = entry_minute(rosters, roster_id)
    home_goals, away_goals = reconstruct_score(shots, minute)
    side = rosters.set_index("roster_id").loc[roster_id, "side"]
    return home_goals - away_goals if side == "h" else away_goals - home_goals


def score_state_control(shots: pd.DataFrame, side: str, match_length: int = 90) -> float:
    """The player's team's minutes-weighted average goal difference across
    the whole match - how much of the game was spent ahead, level, or
    behind, as a single number per match. Positive means mostly ahead.

    A goal at minute 95 (stoppage time) is clipped to `match_length`
    first, so it contributes a zero-length interval at the very end
    rather than counting as if the team led for "negative" extra time.
    `shots` must already be filtered to a single match."""
    goals = shots[shots["result"] == "Goal"].copy()
    goals["minute"] = goals["minute"].clip(upper=match_length)
    goals = goals.sort_values("minute", kind="stable")

    home_goals = away_goals = 0
    previous_minute = 0
    weighted_sum = 0.0
    for _, goal in goals.iterrows():
        diff = home_goals - away_goals if side == "h" else away_goals - home_goals
        weighted_sum += diff * (goal["minute"] - previous_minute)
        if goal["side"] == "h":
            home_goals += 1
        else:
            away_goals += 1
        previous_minute = goal["minute"]

    diff = home_goals - away_goals if side == "h" else away_goals - home_goals
    weighted_sum += diff * (match_length - previous_minute)
    return weighted_sum / match_length


DEFAULT_RIDGE_ALPHAS = (
    0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0,
    10000.0, 30000.0, 100000.0, 300000.0,
)


def _expanding_folds(n: int, n_folds: int, min_train: int):
    """(train_slice, valid_slice) pairs for an expanding-window, time-
    ordered cross-validation: every fold trains only on rows strictly
    before its validation block, never on a row that comes after it - the
    same leakage validation.held_out_predictions guards against, applied
    to picking alpha instead of to the strength model's own ratings.
    `X`/`y` must already be in date order for these slices to mean that."""
    fold_size = (n - min_train) // n_folds
    for k in range(n_folds):
        train_end = min_train + k * fold_size
        valid_end = min_train + (k + 1) * fold_size if k < n_folds - 1 else n
        yield slice(0, train_end), slice(train_end, valid_end)


def held_out_rmse(X: pd.DataFrame, y: pd.Series, alpha: float, n_folds: int = 5, min_train: int = 100) -> float:
    """Average held-out RMSE across expanding, time-ordered folds, at a
    fixed alpha. `X`/`y` must already be in date order."""
    errors = []
    for train_slice, valid_slice in _expanding_folds(len(X), n_folds, min_train):
        coef = fit_ridge(X.iloc[train_slice], y.iloc[train_slice], alpha)
        predicted = X.iloc[valid_slice].apply(lambda row: predict_lineup_outcome(coef, row), axis=1)
        errors.extend((predicted - y.iloc[valid_slice]).to_numpy())
    return float(np.sqrt(np.mean(np.square(errors))))


def select_ridge_alpha(
    X: pd.DataFrame,
    y: pd.Series,
    alphas=DEFAULT_RIDGE_ALPHAS,
    n_folds: int = 5,
    min_train: int = 100,
) -> float:
    """The alpha with the lowest average held-out RMSE across expanding,
    time-ordered folds. `X`/`y` must already be in date order - picking
    alpha from a randomly shuffled split would let a fold "validate" on a
    match using coefficients fitted partly on later matches."""
    best_alpha, best_rmse = alphas[0], math.inf
    for alpha in alphas:
        rmse = held_out_rmse(X, y, alpha, n_folds, min_train)
        if rmse < best_rmse:
            best_alpha, best_rmse = alpha, rmse
    return best_alpha


def opponent_net_strength(ratings: Ratings, opponent: str, name_map: NameMap) -> float:
    """How strong `opponent` is overall - attack minus defence - from
    already-fitted ratings, one number per team instead of two, since this
    feeds a single predictor column rather than predict_xg's pair.

    strength.fit() keys its ratings by whatever spelling the Understat-
    sourced league history used ("Tottenham"); `opponent` here is a
    Liverpool match's canonical FBref name ("Tottenham Hotspur") - the
    same mismatch build_matches() already resolves, so both sides are
    routed through the same name_map before comparing."""
    canonical = name_map.team(opponent)
    attack = {name_map.team(team): value for team, value in ratings.attack.items()}
    defence = {name_map.team(team): value for team, value in ratings.defence.items()}
    return attack.get(canonical, 0.0) - defence.get(canonical, 0.0)


def build_lineup_design_matrix(
    matches: pd.DataFrame,
    rosters: pd.DataFrame,
    shots: pd.DataFrame,
    league_matches: pd.DataFrame,
    name_map: NameMap,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    penalty: float = DEFAULT_PENALTY,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """One row per Liverpool league match with Understat coverage, in
    date order - cup and European matches carry no Understat data and are
    excluded, the same filter onoff.py and diagnosis.py use. A match
    missing `days_since_previous_match` (the very first match in the
    whole dataset has no earlier one to count from) is dropped rather
    than given an invented rest value.

    Returns (X, y, player_columns): X holds one column per Liverpool
    player who ever appeared - his minutes played, as a fraction of a
    full 90 (0.0 never played, 1.0 the full match, ~0.33 a 30-minute
    cameo) rather than a flat 0/1, so a late substitute isn't credited or
    blamed for the match as if he'd started it. counterfactual_effect's
    "flag at 1 vs 0" comparison then reads as "played the whole match"
    vs. "didn't play at all" - plus opponent_strength (fitted on league
    history strictly before that match), is_home, rest_days, one era
    dummy per manager but the most-common one (the implicit baseline -
    with only one manager in the data, no dummy is needed at all), and
    score_state_control; y is xg minus xga."""
    covered = matches[matches["understat_match_id"].notna() & matches["days_since_previous_match"].notna()]
    covered = covered.sort_values("date", kind="stable").reset_index(drop=True)

    liverpool_side = covered["venue"].map({"Home": "h", "Away": "a"})
    match_side = covered[["understat_match_id"]].assign(side=liverpool_side.values)
    lfc_rosters = rosters.merge(match_side, on=["understat_match_id", "side"])
    played = lfc_rosters[lfc_rosters["minutes"] > 0].assign(minutes=lambda df: df["minutes"].astype(float))

    appearance = played.pivot_table(
        index="understat_match_id", columns="player_id", values="minutes", aggfunc="sum", fill_value=0.0
    )
    appearance = appearance.reindex(covered["understat_match_id"].to_numpy(), fill_value=0.0).reset_index(drop=True)
    appearance = appearance / 90.0
    appearance.columns = [f"player_{player_id}" for player_id in appearance.columns]
    player_columns = list(appearance.columns)

    # The most-common era is the baseline (dropped), not an arbitrary one -
    # the baseline's own coefficients are never estimated directly, only
    # implied by the others, so it should be the era with the most matches
    # behind it, not (say) a 5-match caretaker spell with almost nothing
    # to estimate from.
    baseline_manager = covered["manager"].value_counts().idxmax()
    era_columns = {
        f"era_{manager}": (covered["manager"] == manager).astype(float)
        for manager in sorted(covered["manager"].unique())
        if manager != baseline_manager
    }

    empty_shots = shots.iloc[0:0]
    shots_by_match = {match_id: group for match_id, group in shots.groupby("understat_match_id")}
    opponent_strength, score_state = [], []
    for _, row in covered.iterrows():
        side = "h" if row["venue"] == "Home" else "a"
        match_shots = shots_by_match.get(row["understat_match_id"], empty_shots)
        score_state.append(score_state_control(match_shots, side))

        training_history = league_matches[league_matches["date"] < row["date"]]
        ratings = fit(training_history, half_life_days=half_life_days, penalty=penalty)
        opponent_strength.append(opponent_net_strength(ratings, row["opponent"], name_map))

    controls = pd.DataFrame({
        "opponent_strength": opponent_strength,
        "is_home": (covered["venue"] == "Home").astype(float).to_numpy(),
        "rest_days": covered["days_since_previous_match"].astype(float).to_numpy(),
        "score_state_control": score_state,
        **{name: values.to_numpy() for name, values in era_columns.items()},
    })
    X = pd.concat([appearance, controls], axis=1)
    y = (covered["xg"] - covered["xga"]).reset_index(drop=True)
    return X, y, player_columns


def fit_ridge(X: pd.DataFrame, y: pd.Series, alpha: float) -> pd.Series:
    """Ridge regression coefficients, including an intercept, which is
    never penalized - there's no "average" intercept to shrink toward.

    Every column of X is standardized to mean 0 and std 1 before the
    penalty is applied, so one `alpha` weighs every predictor the same
    way regardless of its native scale (a 0/1 player flag and a
    multi-day rest column would otherwise be penalized unfairly
    differently). The returned coefficients are rescaled back to X's own
    units, so a caller can apply them directly to a raw feature row."""
    means = X.mean()
    stds = X.std(ddof=0).replace(0.0, 1.0)
    standardized = (X - means) / stds
    centered_y = y - y.mean()

    gram = standardized.to_numpy().T @ standardized.to_numpy()
    cross = standardized.to_numpy().T @ centered_y.to_numpy()
    ridge_matrix = gram + alpha * np.eye(len(X.columns))
    standardized_coef = np.linalg.solve(ridge_matrix, cross)

    coef = pd.Series(standardized_coef, index=X.columns) / stds
    intercept = float(y.mean() - (means * coef).sum())
    return pd.concat([pd.Series({"intercept": intercept}), coef])


def predict_lineup_outcome(coefficients: pd.Series, row: pd.Series) -> float:
    """The fitted model's prediction for one feature row, in fit_ridge's
    own units (raw feature values, not standardized ones)."""
    feature_columns = coefficients.index.drop("intercept")
    return float(coefficients["intercept"] + (coefficients[feature_columns] * row[feature_columns]).sum())


def counterfactual_effect(coefficients: pd.Series, row: pd.Series, player_column: str) -> float:
    """The predicted change in outcome from adding `player_column` to the
    lineup: the prediction with his flag at 1 minus the prediction with it
    at 0, everything else in `row` held fixed."""
    with_player = row.copy()
    with_player[player_column] = 1.0
    without_player = row.copy()
    without_player[player_column] = 0.0
    return predict_lineup_outcome(coefficients, with_player) - predict_lineup_outcome(coefficients, without_player)


def player_lineup_effects(
    matches: pd.DataFrame,
    rosters: pd.DataFrame,
    shots: pd.DataFrame,
    league_matches: pd.DataFrame,
    name_map: NameMap,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    penalty: float = DEFAULT_PENALTY,
    alphas=DEFAULT_RIDGE_ALPHAS,
    n_folds: int = 5,
    min_train: int = 100,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    rng: np.random.Generator | None = None,
) -> dict:
    """Per-player adjusted lineup effects with bootstrap intervals, gated
    on actually earning their place: the player-flag model only gets
    reported if it beats the same model with the player columns dropped,
    on held-out matches. If it doesn't, `effects` is None - the plan's
    explicit requirement that unproven player effects are not reported."""
    X, y, player_columns = build_lineup_design_matrix(
        matches, rosters, shots, league_matches, name_map, half_life_days, penalty
    )
    return _gated_player_effects(X, y, player_columns, alphas, n_folds, min_train, n_bootstrap, confidence, rng)


def _gated_player_effects(
    X: pd.DataFrame,
    y: pd.Series,
    player_columns: list[str],
    alphas=DEFAULT_RIDGE_ALPHAS,
    n_folds: int = 5,
    min_train: int = 100,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    rng: np.random.Generator | None = None,
) -> dict:
    """The gate itself, pulled out of player_lineup_effects so it can be
    tested directly against a synthetic design matrix, without needing
    real match data or strength.fit()'s cost to exercise it."""
    control_columns = [column for column in X.columns if column not in player_columns]

    full_alpha = select_ridge_alpha(X, y, alphas, n_folds, min_train)
    full_rmse = held_out_rmse(X, y, full_alpha, n_folds, min_train)

    reduced_alpha = select_ridge_alpha(X[control_columns], y, alphas, n_folds, min_train)
    reduced_rmse = held_out_rmse(X[control_columns], y, reduced_alpha, n_folds, min_train)

    beats_no_player_model = full_rmse < reduced_rmse
    effects = (
        bootstrap_player_effects(X, y, player_columns, full_alpha, n_bootstrap, confidence, rng)
        if beats_no_player_model else None
    )
    return {
        "effects": effects,
        "full_model_rmse": full_rmse,
        "no_player_model_rmse": reduced_rmse,
        "beats_no_player_model": beats_no_player_model,
        "alpha": full_alpha,
    }


def bootstrap_player_effects(
    X: pd.DataFrame,
    y: pd.Series,
    player_columns: list[str],
    alpha: float,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Per-player point effect and a bootstrap confidence interval.

    Resamples match-rows with replacement and refits fit_ridge on each
    resample. A player who appeared in only a handful of matches will, by
    chance, sit in very few rows of a given resample (sometimes none) -
    his coefficient swings around far more across resamples than a
    regular starter's, which is exactly what should make his interval
    wider, with no special-casing for "few minutes" needed here."""
    rng = rng if rng is not None else np.random.default_rng()
    n = len(X)
    samples = {column: [] for column in player_columns}
    for _ in range(n_bootstrap):
        resample_idx = rng.integers(0, n, n)
        resampled_X = X.iloc[resample_idx].reset_index(drop=True)
        resampled_y = y.iloc[resample_idx].reset_index(drop=True)
        coef = fit_ridge(resampled_X, resampled_y, alpha)
        for column in player_columns:
            samples[column].append(coef[column])

    point_coef = fit_ridge(X, y, alpha)
    lower_q, upper_q = (1 - confidence) / 2, 1 - (1 - confidence) / 2
    rows = []
    for column in player_columns:
        values = np.array(samples[column])
        rows.append({
            "player": column,
            "point_effect": point_coef[column],
            "ci_low": float(np.quantile(values, lower_q)),
            "ci_high": float(np.quantile(values, upper_q)),
        })
    return pd.DataFrame(rows).set_index("player")
