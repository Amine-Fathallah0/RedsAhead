"""Checks our parsers against a real, finished Understat season.

These need internet the first time (the response is then cached on disk, in the
gitignored cache folder, and never fetched again). They are skipped by default;
run them on purpose with:  python -m pytest -m integration
"""
import math

import pytest

from backend.config import load_settings
from backend.ingest.polite_fetcher import PoliteFetcher
from backend.ingest.understat import (
    fetch_league_data,
    parse_league_matches,
    parse_player_seasons,
    parse_team_match_stats,
)

pytestmark = pytest.mark.integration

LEAGUE = "EPL"
SEASON = 2025  # 2025/26 is finished, so a cached copy can never go stale


@pytest.fixture(scope="module")
def season_tables():
    settings = load_settings()
    fetcher = PoliteFetcher(settings.cache_dir, settings.min_request_interval_seconds)
    data = fetch_league_data(fetcher, LEAGUE, SEASON)
    return {
        "matches": parse_league_matches(data),
        "team_stats": parse_team_match_stats(data),
        "players": parse_player_seasons(data, SEASON),
    }


def test_a_full_season_has_380_matches_between_20_teams(season_tables):
    matches = season_tables["matches"]
    teams = set(matches["home_team"]) | set(matches["away_team"])

    assert len(matches) == 380
    assert len(teams) == 20


def test_every_team_played_38_matches(season_tables):
    matches_per_team = season_tables["team_stats"].groupby("team").size()

    assert len(matches_per_team) == 20
    assert (matches_per_team == 38).all()


def test_goals_agree_between_the_match_table_and_the_team_table(season_tables):
    matches = season_tables["matches"]
    team_stats = season_tables["team_stats"]
    goals_in_matches = (matches["home_goals"] + matches["away_goals"]).sum()

    assert team_stats["goals"].sum() == goals_in_matches
    assert team_stats["goals_against"].sum() == goals_in_matches


def test_xg_for_and_against_balance_across_the_league(season_tables):
    team_stats = season_tables["team_stats"]

    assert math.isclose(team_stats["xg"].sum(), team_stats["xga"].sum(), rel_tol=1e-3)


def test_ppda_is_positive_and_rarely_missing(season_tables):
    ppda = season_tables["team_stats"]["ppda"]

    assert (ppda.dropna() > 0).all()
    assert ppda.isna().mean() < 0.01


def test_liverpool_has_a_full_season_in_the_data(season_tables):
    team_stats = season_tables["team_stats"]

    assert (team_stats["team"] == "Liverpool").sum() == 38


def test_player_table_is_populated_with_unique_players(season_tables):
    players = season_tables["players"]

    assert len(players) > 400
    assert (players["minutes"] >= 0).all()
    assert players["player_id"].is_unique
