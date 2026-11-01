"""Raw on/off: how a team metric looks with a player available, versus
without them - the simplest possible player-impact signal, and the most
confounded. It says nothing about why; Task 11's adjusted lineup effects
are what tries to control for the opponent and the teammates around them."""

import math

import pandas as pd

from backend.analysis.thresholds import display_threshold


def raw_onoff(player_id, matches: pd.DataFrame, rosters: pd.DataFrame, metric: str) -> dict:
    """Split `matches` into those the player appeared in and those they
    didn't, and compare the mean of `metric` between the two groups.

    Only matches with real Understat coverage count - a cup match has no
    `understat_match_id` and so no roster data either, and is excluded
    from both groups rather than guessed into "without the player" just
    because there's no roster row proving otherwise.

    `rosters` holds both teams' players for a match, not just Liverpool's;
    which `side` ('h'/'a') is Liverpool's own is read off `matches`'
    `venue` column (Home -> 'h', Away -> 'a')."""
    covered = matches[matches["understat_match_id"].notna()].copy()

    liverpool_side = covered["venue"].map({"Home": "h", "Away": "a"})
    match_side = covered[["understat_match_id"]].assign(side=liverpool_side.values)
    liverpool_rosters = rosters.merge(match_side, on=["understat_match_id", "side"])

    played_match_ids = set(
        liverpool_rosters.loc[
            (liverpool_rosters["player_id"] == player_id) & (liverpool_rosters["minutes"] > 0),
            "understat_match_id",
        ]
    )
    player_minutes = liverpool_rosters.loc[
        liverpool_rosters["player_id"] == player_id, "minutes"
    ].sum()

    with_player = covered[covered["understat_match_id"].isin(played_match_ids)]
    without_player = covered[~covered["understat_match_id"].isin(played_match_ids)]

    with_mean = with_player[metric].mean() if not with_player.empty else math.nan
    without_mean = without_player[metric].mean() if not without_player.empty else math.nan

    threshold = display_threshold(len(covered))

    return {
        "metric": metric,
        "with_n": len(with_player),
        "without_n": len(without_player),
        "with_player_mean": with_mean,
        "without_player_mean": without_mean,
        "difference": with_mean - without_mean,
        "player_minutes": float(player_minutes),
        "display_threshold": threshold,
        "below_threshold": bool(player_minutes < threshold),
    }
