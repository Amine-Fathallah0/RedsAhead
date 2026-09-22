import json

from backend.ingest.understat import fetch_league_data, league_url, match_url


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