from backend.ingest.understat import league_url, match_url

def test_league_url_builds_the_understat_endpoint():
    assert (
        league_url("EPL", 2025)
        == "https://understat.com/getLeagueData/EPL/2025"
    )
def test_match_url_builds_the_understat_endpoint():
    assert match_url("28778") == "https://understat.com/getMatchData/28778"