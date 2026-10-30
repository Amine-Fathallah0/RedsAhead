"""The score-state control: what the scoreline actually was at the moment
each player entered the match, reconstructed from shot-by-shot goal data.
Task 11's ridge model needs this so a substitute who came on to see out a
3-0 win isn't credited (or blamed) for a game state he didn't create."""

import pandas as pd


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
