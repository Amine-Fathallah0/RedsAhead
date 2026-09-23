from datetime import date
import json

from backend.ingest.understat import fetch_league_data, league_url, match_url, parse_league_matches
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