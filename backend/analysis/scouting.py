"""Scouting: where a player sits relative to his positional peers across
all 20 clubs, and which other players' per-90 profiles look most like
his. Both only ever compare players who clear config/settings.yaml's
min_minutes - Understat's unclassified "S"-only entries (emergency subs
with no settled position, a few games and under 30 minutes on average)
never reach that bar anyway, so no special-casing is needed for them."""

import numpy as np
import pandas as pd

SIMILARITY_METRICS = ["goals", "assists", "shots", "key_passes", "xg", "xa", "npg", "npxg", "xg_chain", "xg_buildup"]
DEFAULT_MIN_MINUTES = 900


def primary_position(position: str) -> str:
    """Understat lists a player's positions most-played first ("F M S"
    for a forward occasionally used in midfield or off the bench) - the
    first token is the position that actually describes him."""
    return position.split()[0]


def _eligible_pool(player_seasons: pd.DataFrame, season: int, position_group: str, min_minutes: int) -> pd.DataFrame:
    """Rows for one season and position group, with at least `min_minutes`."""
    eligible = player_seasons[player_seasons["minutes"] >= min_minutes]
    same_position = eligible["position"].map(primary_position) == position_group
    return eligible[same_position & (eligible["season"] == season)]


def percentile_ranks(
    player_seasons: pd.DataFrame,
    season: int,
    position_group: str,
    metric: str,
    min_minutes: int = DEFAULT_MIN_MINUTES,
) -> pd.DataFrame:
    """One row per eligible player sharing `position_group` in `season`:
    his `metric` per 90 minutes, and his percentile rank (0-100) on it
    among that same pool. Tied values share the average rank rather than
    an arbitrary tie-break order."""
    pool = _eligible_pool(player_seasons, season, position_group, min_minutes).copy()
    pool["per_90"] = pool[metric] / pool["minutes"] * 90
    pool["percentile"] = pool["per_90"].rank(pct=True) * 100
    return pool[["player_id", "player", "team", "per_90", "percentile"]].reset_index(drop=True)


def similar_players(
    player_seasons: pd.DataFrame,
    player_id,
    season: int,
    min_minutes: int = DEFAULT_MIN_MINUTES,
    top_n: int = 10,
) -> pd.DataFrame:
    """The up to `top_n` eligible players (any club) whose standardised
    per-90 profile across SIMILARITY_METRICS is closest to `player_id`'s,
    by cosine similarity - he is never returned as similar to himself.
    Comparison only happens within his own position group and season,
    since a defender's and a striker's profiles aren't comparable to
    begin with. Empty if `player_id` isn't eligible himself in that
    season, or has no eligible peers in his position group."""
    eligible = player_seasons[player_seasons["minutes"] >= min_minutes]
    season_rows = eligible[eligible["season"] == season]
    target_rows = season_rows[season_rows["player_id"] == player_id]
    if target_rows.empty:
        return pd.DataFrame(columns=["player_id", "player", "team", "similarity"])

    position_group = primary_position(target_rows.iloc[0]["position"])
    pool = season_rows[season_rows["position"].map(primary_position) == position_group]

    per_90 = pool[SIMILARITY_METRICS].div(pool["minutes"], axis=0) * 90
    standardized = (per_90 - per_90.mean()) / per_90.std(ddof=0).replace(0.0, 1.0)

    target_vector = standardized.loc[target_rows.index[0]].to_numpy()
    vectors = standardized.to_numpy()
    norms = np.linalg.norm(vectors, axis=1)
    target_norm = np.linalg.norm(target_vector)
    # A pool of just the target himself (no eligible peers in his position
    # group) makes every standardized value - and so every norm - exactly
    # 0: a legitimate 0/0, not an error, and it's filtered out below anyway.
    with np.errstate(invalid="ignore", divide="ignore"):
        similarity = (vectors @ target_vector) / (norms * target_norm)

    result = pool[["player_id", "player", "team"]].copy()
    result["similarity"] = similarity
    result = result[result["player_id"] != player_id]
    return result.sort_values("similarity", ascending=False).reset_index(drop=True).head(top_n)
