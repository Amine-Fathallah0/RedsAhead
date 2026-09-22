

def league_url(league: str, season: int) -> str:
    """Builds the Understat endpoint for a given league and season."""
    return f"https://understat.com/getLeagueData/{league}/{season}"
def match_url(match_id: str) -> str:
    """Builds the Understat endpoint for a given match."""
    return f"https://understat.com/getMatchData/{match_id}"
