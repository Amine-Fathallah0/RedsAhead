"""Checks our parsers against a real, finished Understat season.

These need internet the first time (the response is then cached on disk, in the
gitignored cache folder, and never fetched again). They are skipped by default;
run them on purpose with:  python -m pytest -m integration
"""
import math

import pytest
from pandas import isna

from backend.config import load_settings
from backend.ingest.polite_fetcher import PoliteFetcher
from backend.ingest.understat import (
    fetch_league_data,
    fetch_match_data,
    parse_league_matches,
    parse_match_rosters,
    parse_match_shots,
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


# --- single matches: shots and rosters --------------------------------------------------


@pytest.fixture(scope="module")
def sample_matches(season_tables):
    """Three real Liverpool matches: the first, the middle and the last of the season."""
    settings = load_settings()
    fetcher = PoliteFetcher(settings.cache_dir, settings.min_request_interval_seconds)
    matches = season_tables["matches"]
    liverpool = matches[
        (matches["home_team"] == "Liverpool") | (matches["away_team"] == "Liverpool")
    ].sort_values("date")
    picks = liverpool.iloc[[0, len(liverpool) // 2, -1]]

    samples = []
    for _, match in picks.iterrows():
        data = fetch_match_data(fetcher, match["understat_match_id"])
        samples.append({
            "match": match,
            "shots": parse_match_shots(data),
            "rosters": parse_match_rosters(data, match["understat_match_id"]),
        })
    return samples


def test_each_side_starts_eleven_players(sample_matches):
    for sample in sample_matches:
        rosters = sample["rosters"]
        starters_per_side = rosters[rosters["position"] != "Sub"].groupby("side").size()

        assert len(starters_per_side) == 2
        assert (starters_per_side == 11).all()


def test_every_listed_player_played_some_minutes(sample_matches):
    # In the sampled matches Understat lists only players who actually appeared,
    # so no unused substitute shows up. If this ever fails, unused subs do appear.
    for sample in sample_matches:
        assert (sample["rosters"]["minutes"] > 0).all()


def test_each_side_adds_up_to_about_eleven_full_matches(sample_matches):
    for sample in sample_matches:
        minutes_per_side = sample["rosters"].groupby("side")["minutes"].sum()

        assert minutes_per_side.between(980, 1000).all()   # observed: 990 to 992


def test_substitution_links_are_symmetric_and_point_at_real_entries(sample_matches):
    for sample in sample_matches:
        by_id = sample["rosters"].set_index("roster_id")
        assert by_id.index.is_unique
        for row in sample["rosters"].itertuples():
            if isna(row.roster_in):
                continue
            assert row.roster_in in by_id.index                                 # a real entry...
            assert by_id.loc[row.roster_in, "roster_out"] == row.roster_id      # ...that points back


def test_a_substitution_pair_adds_up_to_the_match_length(sample_matches):
    for sample in sample_matches:
        by_id = sample["rosters"].set_index("roster_id")
        for row in sample["rosters"].itertuples():
            if isna(row.roster_in):
                continue
            partner = by_id.loc[row.roster_in]
            if not isna(partner["roster_in"]):
                continue   # the substitute was himself replaced later; a chain, not a simple pair
            assert 90 <= row.minutes + partner["minutes"] <= 92   # observed: 90 or 91


def test_shot_xg_adds_up_to_the_league_level_match_xg(sample_matches):
    # Observed differences are under 0.01 (e.g. 3.000 vs 2.994); cause not investigated.
    for sample in sample_matches:
        match, shots = sample["match"], sample["shots"]
        home_xg = shots[shots["side"] == "h"]["xg"].sum()
        away_xg = shots[shots["side"] == "a"]["xg"].sum()

        assert math.isclose(home_xg, match["home_xg"], abs_tol=0.02)
        assert math.isclose(away_xg, match["away_xg"], abs_tol=0.02)
