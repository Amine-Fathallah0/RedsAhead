from datetime import date
import json
import math

from backend.ingest.understat import (
    fetch_league_data,
    fetch_match_data,
    league_url,
    match_url,
    parse_league_matches,
    parse_match_rosters,
    parse_match_shots,
    parse_player_seasons,
    parse_team_match_stats,
)
from pandas import DataFrame


class FakeFetcher:
    def __init__(self, pages):
        self.pages = pages

    def get(self, url):
        return self.pages[url]


def test_league_url_builds_the_understat_endpoint():
    assert (
        league_url("EPL", 2025)
        == "https://understat.com/getLeagueData/EPL/2025"
    )


def test_match_url_builds_the_understat_endpoint():
    assert match_url("28778") == "https://understat.com/getMatchData/28778"


def test_fetch_league_data_parses_the_json_response():
    payload = {"teams": {}, "players": [], "dates": [{"id": "1"}]}
    fetcher = FakeFetcher({league_url("EPL", 2025): json.dumps(payload)})

    result = fetch_league_data(fetcher, "EPL", 2025)

    assert result == payload

def test_parse_league_matches_keeps_only_played_matches():
    data = {
        "dates": [
            {
                "id": "28778", "isResult": True,
                "h": {"id": "87", "title": "Liverpool"},
                "a": {"id": "73", "title": "Bournemouth"},
                "goals": {"h": "4", "a": "2"},
                "xG": {"h": "2.33133", "a": "1.57"},
                "datetime": "2025-08-15 19:00:00",
            },
            {
                "id": "29200", "isResult": False,
                "h": {"id": "87", "title": "Liverpool"},
                "a": {"id": "80", "title": "Chelsea"},
                "goals": {"h": None, "a": None},
                "xG": {"h": None, "a": None},
                "datetime": "2026-12-05 15:00:00",
            },
        ]
    }

    matches = parse_league_matches(data)

    assert len(matches) == 1
    first = matches.iloc[0]
    assert first["understat_match_id"] == "28778"
    assert first["home_team"] == "Liverpool"
    assert first["home_goals"] == 4   # as an int, not a string
    assert first["home_xg"] == 2.33133      # as a float
    assert first["date"] == date(2025, 8, 15)


def history_entry(date_time, side, ppda_att, ppda_def):
    """One match in a team's history. Only the PPDA numbers vary; the rest is filler."""
    return {
        "h_a": side, "date": date_time,
        "xG": 2.0, "xGA": 1.0, "npxG": 1.8, "npxGA": 0.9,
        "ppda": {"att": ppda_att, "def": ppda_def},
        "ppda_allowed": {"att": 200, "def": 20},
        "deep": 9, "deep_allowed": 8,
        "scored": 3, "missed": 1, "xpts": 2.1, "result": "w",
    }


def test_parse_team_match_stats_has_one_row_per_team_per_match():
    data = {
        "teams": {
            "87": {"id": "87", "title": "Liverpool", "history": [
                history_entry("2025-08-15 19:00:00", "h", 149, 17),
                history_entry("2025-08-25 19:00:00", "a", 120, 0),
            ]},
            "73": {"id": "73", "title": "Bournemouth", "history": [
                history_entry("2025-08-15 19:00:00", "a", 90, 9),
            ]},
        }
    }

    stats = parse_team_match_stats(data)

    assert len(stats) == 3
    assert list(stats["team"]) == ["Liverpool", "Liverpool", "Bournemouth"]
    first = stats.iloc[0]
    assert first["date"] == date(2025, 8, 15)
    assert first["side"] == "h"
    assert first["xg"] == 2.0
    assert first["goals"] == 3
    assert first["deep_allowed"] == 8
    assert math.isclose(first["ppda"], 149 / 17)


def test_ppda_is_nan_when_there_are_no_defensive_actions():
    data = {
        "teams": {
            "87": {"id": "87", "title": "Liverpool", "history": [
                history_entry("2025-08-25 19:00:00", "a", 120, 0),
            ]},
        }
    }

    stats = parse_team_match_stats(data)

    assert math.isnan(stats.iloc[0]["ppda"])


def player_entry(player_id, name, team):
    """One player's season totals. Invented numbers; Understat sends them all as strings."""
    return {
        "id": player_id, "player_name": name, "games": "4", "time": "331",
        "goals": "1", "xG": "1.2", "assists": "1", "xA": "0.9", "shots": "10",
        "key_passes": "6", "yellow_cards": "1", "red_cards": "0",
        "position": "M S", "team_title": team,
        "npg": "1", "npxG": "1.1", "xGChain": "3.1", "xGBuildup": "1.4",
    }


def test_parse_player_seasons_types_the_fields_and_tags_the_season():
    data = {
        "players": [
            player_entry("9001", "Test Player One", "Liverpool"),
            player_entry("9002", "Test Player Two", "Liverpool"),
        ]
    }

    players = parse_player_seasons(data, 2025)

    assert len(players) == 2
    first = players.iloc[0]
    assert first["season"] == 2025
    assert first["player_id"] == "9001"
    assert first["player"] == "Test Player One"
    assert first["minutes"] == 331
    assert first["goals"] == 1
    assert first["xg"] == 1.2
    assert first["xg_buildup"] == 1.4


def test_a_player_who_changed_clubs_keeps_the_raw_team_string():
    data = {"players": [player_entry("9003", "Test Player Three", "Chelsea,Liverpool")]}

    players = parse_player_seasons(data, 2025)

    assert players.iloc[0]["team"] == "Chelsea,Liverpool"


def test_fetch_match_data_parses_the_json_response():
    payload = {"shots": {"h": [], "a": []}, "rosters": {"h": {}, "a": {}}}
    fetcher = FakeFetcher({match_url("28778"): json.dumps(payload)})

    result = fetch_match_data(fetcher, "28778")

    assert result == payload


def shot_entry(shot_id, minute, side, player, assisted):
    """One shot in a match between 'Home FC' and 'Away FC'. Invented values, sent as strings."""
    return {
        "id": shot_id, "match_id": "5000", "minute": minute, "h_a": side,
        "h_team": "Home FC", "a_team": "Away FC",
        "player_id": "70" + shot_id, "player": player, "player_assisted": assisted,
        "X": "0.885", "Y": "0.5", "xG": "0.31",
        "situation": "OpenPlay", "shotType": "RightFoot", "result": "Goal",
        "lastAction": "Pass",
    }


def test_parse_match_shots_combines_both_sides_and_names_the_shooting_team():
    data = {
        "shots": {
            "h": [shot_entry("1", "24", "h", "Home Striker", "Home Winger")],
            "a": [shot_entry("2", "40", "a", "Away Striker", None)],
        }
    }

    shots = parse_match_shots(data)

    assert len(shots) == 2
    assert list(shots["team"]) == ["Home FC", "Away FC"]
    assert list(shots["side"]) == ["h", "a"]
    first = shots.iloc[0]
    assert first["understat_match_id"] == "5000"
    assert first["minute"] == 24
    assert first["player"] == "Home Striker"
    assert first["x"] == 0.885
    assert first["xg"] == 0.31
    assert first["situation"] == "OpenPlay"
    assert first["shot_type"] == "RightFoot"


def test_an_unassisted_shot_has_no_assister():
    data = {"shots": {"h": [], "a": [shot_entry("2", "40", "a", "Away Striker", None)]}}

    shots = parse_match_shots(data)

    assert shots.iloc[0]["assister"] is None


def roster_entry(player_id, name, side, minutes, roster_in, roster_out):
    """One player's roster entry for a match. Invented values, sent as strings."""
    return {
        "id": "9" + player_id, "player_id": player_id, "player": name,
        "team_id": "87" if side == "h" else "80", "h_a": side,
        "position": "DMC", "positionOrder": "5",
        "time": minutes, "roster_in": roster_in, "roster_out": roster_out,
        "goals": "0", "own_goals": "0", "shots": "0", "key_passes": "0",
        "assists": "0", "xG": "0.0", "xA": "0.0", "xGChain": "0.3", "xGBuildup": "0.2",
        "yellow_card": "0", "red_card": "0",
    }


def test_parse_match_rosters_keeps_every_player_including_unused_subs():
    data = {
        "rosters": {
            "h": {
                "1": roster_entry("101", "Starter", "h", "90", "0", "0"),
                "2": roster_entry("102", "Unused Sub", "h", "0", "0", "0"),
            },
            "a": {
                "3": roster_entry("201", "Away Starter", "a", "78", "0", "1"),
            },
        }
    }

    rosters = parse_match_rosters(data)

    assert len(rosters) == 3
    assert set(rosters["player"]) == {"Starter", "Unused Sub", "Away Starter"}
    starter = rosters[rosters["player"] == "Starter"].iloc[0]
    assert starter["minutes"] == 90
    assert starter["position"] == "DMC"
    assert starter["team_id"] == "87"
    unused = rosters[rosters["player"] == "Unused Sub"].iloc[0]
    assert unused["minutes"] == 0


def test_parse_match_rosters_keeps_the_substitution_links():
    data = {
        "rosters": {
            "h": {},
            "a": {"3": roster_entry("201", "Away Starter", "a", "78", "0", "1")},
        }
    }

    rosters = parse_match_rosters(data)

    assert rosters.iloc[0]["roster_in"] == 0
    assert rosters.iloc[0]["roster_out"] == 1