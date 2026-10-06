import pytest

from backend.analysis.data import load_matches, load_player_seasons, load_rosters, load_shots
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
