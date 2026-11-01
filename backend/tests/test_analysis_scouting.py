import pandas as pd
import pytest

from backend.analysis.scouting import percentile_ranks, primary_position, similar_players


def _player_seasons(rows):
    columns = ["season", "player_id", "player", "team", "position", "minutes", "goals", "assists",
               "shots", "key_passes", "xg", "xa", "npg", "npxg", "xg_chain", "xg_buildup"]
    df = pd.DataFrame(rows, columns=columns[: len(rows[0])] if rows else columns)
    for column in columns:
        if column not in df.columns:
            df[column] = 0.0
    return df[columns]


def test_primary_position_takes_the_first_listed_token():
    # Understat lists a player's positions most-played first.
    assert primary_position("F M S") == "F"
    assert primary_position("D S") == "D"
    assert primary_position("GK") == "GK"


def test_percentile_ranks_orders_four_distinct_values_correctly():
    player_seasons = _player_seasons([
        (2024, "p1", "Low", "A", "M S", 1000, 2),
        (2024, "p2", "Mid-low", "A", "M S", 1000, 4),
        (2024, "p3", "Mid-high", "A", "M S", 1000, 6),
        (2024, "p4", "High", "A", "M S", 1000, 8),
    ])

    result = percentile_ranks(player_seasons, season=2024, position_group="M", metric="goals")

    by_player = result.set_index("player_id")
    assert by_player.loc["p1", "percentile"] == pytest.approx(25.0)
    assert by_player.loc["p2", "percentile"] == pytest.approx(50.0)
    assert by_player.loc["p3", "percentile"] == pytest.approx(75.0)
    assert by_player.loc["p4", "percentile"] == pytest.approx(100.0)


def test_percentile_ranks_gives_tied_values_the_same_average_rank():
    player_seasons = _player_seasons([
        (2024, "p1", "A", "X", "M", 1000, 5),
        (2024, "p2", "B", "X", "M", 1000, 5),  # tied with p1
        (2024, "p3", "C", "X", "M", 1000, 10),
    ])

    result = percentile_ranks(player_seasons, season=2024, position_group="M", metric="goals")
    by_player = result.set_index("player_id")

    assert by_player.loc["p1", "percentile"] == pytest.approx(50.0)
    assert by_player.loc["p2", "percentile"] == pytest.approx(50.0)
    assert by_player.loc["p3", "percentile"] == pytest.approx(100.0)


def test_percentile_ranks_excludes_players_below_the_minutes_threshold():
    player_seasons = _player_seasons([
        (2024, "p1", "Regular", "X", "M", 1000, 10),
        (2024, "p2", "Bit-part", "X", "M", 200, 10),  # below min_minutes
    ])

    result = percentile_ranks(player_seasons, season=2024, position_group="M", metric="goals", min_minutes=900)

    assert list(result["player_id"]) == ["p1"]


def test_percentile_ranks_only_compares_within_the_same_position_group_and_season():
    player_seasons = _player_seasons([
        (2024, "p1", "Midfielder", "X", "M", 1000, 10),
        (2024, "p2", "Defender", "X", "D", 1000, 10),  # different position
        (2023, "p3", "Last season", "X", "M", 1000, 10),  # different season
    ])

    result = percentile_ranks(player_seasons, season=2024, position_group="M", metric="goals")

    assert list(result["player_id"]) == ["p1"]


def _player_seasons_full(rows):
    # rows: (season, player_id, player, team, position, minutes, goals, assists,
    #        shots, key_passes, xg, xa, npg, npxg, xg_chain, xg_buildup)
    columns = ["season", "player_id", "player", "team", "position", "minutes", "goals", "assists",
               "shots", "key_passes", "xg", "xa", "npg", "npxg", "xg_chain", "xg_buildup"]
    return pd.DataFrame(rows, columns=columns)


def test_similar_players_finds_the_closest_standardised_per_90_profile():
    # Target and "Twin" share the same per-90 shape (goals/assists scaled
    # together); "Different" has the opposite shape (high assists, no
    # goals) - Twin should come back as the single most similar player.
    player_seasons = _player_seasons_full([
        (2024, "target", "Target", "A", "M", 900, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        (2024, "twin", "Twin", "B", "M", 900, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        (2024, "scaled_twin", "Scaled Twin", "C", "M", 900, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        (2024, "different", "Different", "D", "M", 900, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0),
    ])

    result = similar_players(player_seasons, player_id="target", season=2024)

    assert "target" not in result["player_id"].to_list()  # never himself
    assert result.iloc[0]["player_id"] in ("twin", "scaled_twin")
    assert result.iloc[-1]["player_id"] == "different"


def test_similar_players_only_compares_within_the_same_position_group():
    player_seasons = _player_seasons_full([
        (2024, "target", "Target", "A", "M", 900, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        (2024, "same_shape_different_position", "Other", "B", "D", 900, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ])

    result = similar_players(player_seasons, player_id="target", season=2024)

    assert result.empty


def test_similar_players_excludes_players_below_the_minutes_threshold():
    player_seasons = _player_seasons_full([
        (2024, "target", "Target", "A", "M", 900, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        (2024, "bit_part", "Bit-part", "B", "M", 200, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ])

    result = similar_players(player_seasons, player_id="target", season=2024)

    assert result.empty
