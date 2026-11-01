import pytest

from backend.analysis.data import (
    load_league_matches,
    load_matches,
    load_player_seasons,
    load_rosters,
    load_shots,
)
from backend.config import load_settings

pytestmark = pytest.mark.integration


def test_load_matches_returns_every_row_of_the_database():
    matches = load_matches(load_settings().db_path)

    assert len(matches) == 510
    assert "xg" in matches.columns
    assert "manager" in matches.columns


def test_the_other_tables_load_with_their_expected_sizes():
    db = load_settings().db_path

    assert len(load_shots(db)) == 9094
    assert len(load_rosters(db)) == 10021
    assert len(load_player_seasons(db)) == 5240


def test_load_league_matches_covers_every_team_not_just_liverpool():
    league_matches = load_league_matches(load_settings().db_path)

    assert len(league_matches) == 3470
    assert "home_team" in league_matches.columns
    assert "away_team" in league_matches.columns
    # More than two teams ever appear - this is the whole league, not one fixture list.
    assert league_matches["home_team"].nunique() > 2
