from datetime import date

import pandas as pd
import pytest

from backend.model.eras import Era
from backend.model.matches import build_matches
from backend.model.names import NameMap

NAME_MAP = NameMap({
    "Liverpool": [],
    "Fulham": [],
    "Atlético Madrid": [],
    "Bournemouth": [],
})
ERAS = [Era("Slot", date(2024, 6, 1), None)]


def fbref_row(d, competition, opponent, goals_for, goals_against):
    return {
        "season": "2025-2026", "date": d, "competition": competition,
        "venue": "Home", "opponent": opponent, "result": "?",
        "goals_for": goals_for, "goals_against": goals_against,
        "penalties_for": None, "penalties_against": None,
        "possession": 55, "formation": "4-2-3-1", "opp_formation": "4-4-2",
    }


def understat_match(match_id, d, home, away):
    return {
        "understat_match_id": match_id, "date": d,
        "home_team": home, "away_team": away,
        "home_goals": 0, "away_goals": 0, "home_xg": 1.0, "away_xg": 0.5,
    }


def team_stats_row(team, d, xg=1.0):
    return {
        "team": team, "date": d, "side": "h",
        "xg": xg, "xga": 0.5, "npxg": xg, "npxga": 0.5,
        "ppda": 10.0, "ppda_allowed": 12.0, "deep": 5, "deep_allowed": 4,
        "goals": 0, "goals_against": 0, "xpts": 1.5,
    }


def test_league_matches_get_understat_stats_and_cup_matches_do_not():
    # The real example this was built from: an ordinary Champions League night
    # against Atlético, then a flat Premier League draw with Fulham 3 days
    # later. Only the Premier League row has Understat stats to attach.
    fbref = pd.DataFrame([
        fbref_row(date(2025, 9, 9), "Champions Lg", "Atlético Madrid", 2, 1),
        fbref_row(date(2025, 9, 12), "Premier League", "Fulham", 0, 0),
    ])
    understat_matches = pd.DataFrame([understat_match("501", date(2025, 9, 12), "Liverpool", "Fulham")])
    team_stats = pd.DataFrame([team_stats_row("Liverpool", date(2025, 9, 12))])

    matches, problems = build_matches(fbref, understat_matches, team_stats, NAME_MAP, ERAS)

    assert problems == []
    cup = matches[matches["competition"] == "Champions Lg"].iloc[0]
    assert pd.isna(cup["xg"])
    league = matches[matches["competition"] == "Premier League"].iloc[0]
    assert league["xg"] == 1.0


def test_days_since_previous_match_spans_every_competition():
    fbref = pd.DataFrame([
        fbref_row(date(2025, 9, 9), "Champions Lg", "Atlético Madrid", 2, 1),
        fbref_row(date(2025, 9, 12), "Premier League", "Fulham", 0, 0),
    ])
    understat_matches = pd.DataFrame([understat_match("501", date(2025, 9, 12), "Liverpool", "Fulham")])
    team_stats = pd.DataFrame([team_stats_row("Liverpool", date(2025, 9, 12))])

    matches, _ = build_matches(fbref, understat_matches, team_stats, NAME_MAP, ERAS)

    matches = matches.set_index("date")
    assert pd.isna(matches.loc[date(2025, 9, 9), "days_since_previous_match"])
    assert matches.loc[date(2025, 9, 12), "days_since_previous_match"] == 3


def test_every_match_is_tagged_with_its_manager():
    fbref = pd.DataFrame([fbref_row(date(2025, 9, 12), "Premier League", "Fulham", 0, 0)])
    understat_matches = pd.DataFrame([understat_match("501", date(2025, 9, 12), "Liverpool", "Fulham")])
    team_stats = pd.DataFrame([team_stats_row("Liverpool", date(2025, 9, 12))])

    matches, _ = build_matches(fbref, understat_matches, team_stats, NAME_MAP, ERAS)

    assert list(matches["manager"]) == ["Slot"]


def test_unmapped_fbref_opponent_is_reported_and_kept_not_dropped():
    fbref = pd.DataFrame([fbref_row(date(2025, 9, 12), "Premier League", "Somewhere FC", 1, 0)])

    matches, problems = build_matches(fbref, pd.DataFrame(), pd.DataFrame(), NAME_MAP, ERAS)

    assert any("Somewhere FC" in p for p in problems)
    assert matches.iloc[0]["opponent"] == "Somewhere FC"   # kept, not dropped


def test_fbref_league_match_missing_from_understat_is_reported():
    fbref = pd.DataFrame([fbref_row(date(2025, 9, 12), "Premier League", "Fulham", 0, 0)])

    _, problems = build_matches(fbref, pd.DataFrame(), pd.DataFrame(), NAME_MAP, ERAS)

    assert any("no Understat match" in p and "Fulham" in p for p in problems)


def test_understat_match_missing_from_fbref_is_reported():
    understat_matches = pd.DataFrame([understat_match("501", date(2025, 9, 12), "Liverpool", "Fulham")])

    _, problems = build_matches(pd.DataFrame(columns=[
        "season", "date", "competition", "venue", "opponent", "result",
        "goals_for", "goals_against", "penalties_for", "penalties_against",
        "possession", "formation", "opp_formation",
    ]), understat_matches, pd.DataFrame(), NAME_MAP, ERAS)

    assert any("501" in p and "has no FBref fixture" in p for p in problems)
