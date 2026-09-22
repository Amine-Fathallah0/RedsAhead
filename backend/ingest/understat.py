import json


def league_url(league: str, season: int) -> str:
    """Builds the Understat endpoint for a given league and season."""
    return f"https://understat.com/getLeagueData/{league}/{season}"
def match_url(match_id: str) -> str:
    """Builds the Understat endpoint for a given match."""
    return f"https://understat.com/getMatchData/{match_id}"


def fetch_league_data(fetcher, league: str, season: int) -> dict:
    """Fetches and parses one league season's data (teams, players, dates)."""
    url = league_url(league, season)
    text = fetcher.get(url)
    return json.loads(text)
