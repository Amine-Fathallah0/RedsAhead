import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class Ratings:
    base: float
    home_advantage: float
    attack: dict[str, float] = field(default_factory=dict)
    defence: dict[str, float] = field(default_factory=dict)
    home_advantage_adjustment: dict[str, float] = field(default_factory=dict)


def predict_xg(ratings: Ratings, home: str, away: str) -> tuple[float, float]:
    """Expected xG for both sides of one match, from already-fitted ratings.

    A team with no rating yet (newly promoted, not seen in training) is
    treated as exactly average: 0.0 on attack, defence, and the home
    adjustment. That's a placeholder, not a real answer - a promoted team
    is usually weaker than average, not average - deferred to its own task."""
    home_xg = math.exp(
        ratings.base
        + ratings.home_advantage
        + ratings.home_advantage_adjustment.get(home, 0.0)
        + ratings.attack.get(home, 0.0)
        - ratings.defence.get(away, 0.0)
    )
    away_xg = math.exp(
        ratings.base
        + ratings.attack.get(away, 0.0)
        - ratings.defence.get(home, 0.0)
    )
    return home_xg, away_xg


def _recency_weights(dates: pd.Series, half_life_days: float | None) -> np.ndarray:
    """1.0 for the most recent match, halving every `half_life_days` of
    calendar age before it. `None` means no decay at all - every match
    counts equally, regardless of age."""
    if half_life_days is None:
        return np.ones(len(dates))
    dates = pd.to_datetime(dates)
    age_in_days = (dates.max() - dates).dt.days.to_numpy(dtype=float)
    return 0.5 ** (age_in_days / half_life_days)


def _poisson_deviance(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """How far a predicted xG was from the actual one, in the same units
    Poisson regression uses for a count - this punishes a given-sized miss
    more when the true value is small, less when it's large, instead of
    treating every miss the same way squared error would.

    y * ln(y / ŷ) is defined as 0 when y = 0, by the usual convention
    (the limit of y*ln(y) as y -> 0 is 0). Swapping in 1 for actual=0
    before the log just avoids log(0); the np.where after throws that
    substituted value away, since its term is forced to 0 regardless."""
    safe_actual = np.where(actual > 0, actual, 1.0)
    log_term = np.where(actual > 0, actual * np.log(safe_actual / predicted), 0.0)
    return 2 * (log_term - (actual - predicted))


def fit(history: pd.DataFrame, half_life_days: float | None, penalty: float) -> Ratings:
    """Fit attack, defence, and a per-team home-advantage adjustment to a
    league's match history, by minimizing weighted Poisson deviance (how
    far the model's predicted xG is from the actual xG) plus an L2 penalty
    on the three per-team arrays. `base` and `home_advantage` are shared by
    every team and are never penalized - there's no "average" version of
    the baseline itself to shrink toward.

    `history` needs `date`, `home_team`, `away_team`, `home_xg`, `away_xg`
    - the same shape `load_league_matches` returns."""
    teams = sorted(set(history["home_team"]) | set(history["away_team"]))
    team_index = {team: i for i, team in enumerate(teams)}
    n = len(teams)

    home_idx = history["home_team"].map(team_index).to_numpy()
    away_idx = history["away_team"].map(team_index).to_numpy()
    actual_home = history["home_xg"].to_numpy(dtype=float)
    actual_away = history["away_xg"].to_numpy(dtype=float)
    weights = _recency_weights(history["date"], half_life_days)

    def unpack(x: np.ndarray):
        base, home_advantage = x[0], x[1]
        attack = x[2 : 2 + n]
        defence = x[2 + n : 2 + 2 * n]
        adjustment = x[2 + 2 * n : 2 + 3 * n]
        return base, home_advantage, attack, defence, adjustment

    def objective(x: np.ndarray) -> float:
        base, home_advantage, attack, defence, adjustment = unpack(x)
        predicted_home = np.exp(
            base + home_advantage + adjustment[home_idx] + attack[home_idx] - defence[away_idx]
        )
        predicted_away = np.exp(base + attack[away_idx] - defence[home_idx])

        data_loss = np.sum(
            weights * (_poisson_deviance(actual_home, predicted_home) + _poisson_deviance(actual_away, predicted_away))
        )
        regularization = penalty * (np.sum(attack**2) + np.sum(defence**2) + np.sum(adjustment**2))
        return float(data_loss + regularization)

    x0 = np.zeros(2 + 3 * n)
    mean_xg = float(np.concatenate([actual_home, actual_away]).mean())
    x0[0] = math.log(mean_xg) if mean_xg > 0 else 0.0
    x0[1] = 0.1  # a mild starting guess at home advantage; the optimizer moves it from here

    result = minimize(objective, x0, method="L-BFGS-B")
    base, home_advantage, attack, defence, adjustment = unpack(result.x)

    return Ratings(
        base=float(base),
        home_advantage=float(home_advantage),
        attack=dict(zip(teams, attack.tolist())),
        defence=dict(zip(teams, defence.tolist())),
        home_advantage_adjustment=dict(zip(teams, adjustment.tolist())),
    )
