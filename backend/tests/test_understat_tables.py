from datetime import date
import json
import math

from backend.ingest.understat import (
    fetch_league_data,
    league_url,
    match_url,
    parse_league_matches,
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