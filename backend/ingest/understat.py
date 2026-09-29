import json

import pandas as pd


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


def fetch_match_data(fetcher, match_id: str) -> dict:
    """Fetches and parses one match's data (shots and rosters)."""
    url = match_url(match_id)
    text = fetcher.get(url)
    return json.loads(text)


def parse_league_matches(data: dict) -> pd.DataFrame:
    """One row per played match in a league season, from the 'dates' list."""
    rows = []
    for item in data["dates"]:
        if not item["isResult"]:
            continue
        rows.append({
            "understat_match_id": item["id"],
            "date": pd.Timestamp(item["datetime"]).date(),
            "home_team": item["h"]["title"],
            "away_team": item["a"]["title"],
            "home_goals": int(item["goals"]["h"]),
            "away_goals": int(item["goals"]["a"]),
            "home_xg": float(item["xG"]["h"]),
            "away_xg": float(item["xG"]["a"]),
        })
    return pd.DataFrame(rows)


def _ppda(attempts, actions) -> float:
    """Opponent passes per defensive action. NaN when there were no defensive actions."""
    actions = float(actions)
    return float(attempts) / actions if actions else float("nan")


def parse_team_match_stats(data: dict) -> pd.DataFrame:
    """One row per team per match, from the 'teams' dict's match histories."""
    rows = []
    for team in data["teams"].values():
        for entry in team["history"]:
            rows.append({
                "team": team["title"],
                "date": pd.Timestamp(entry["date"]).date(),
                "side": entry["h_a"],
                "xg": float(entry["xG"]),
                "xga": float(entry["xGA"]),
                "npxg": float(entry["npxG"]),
                "npxga": float(entry["npxGA"]),
                "ppda": _ppda(entry["ppda"]["att"], entry["ppda"]["def"]),
                "ppda_allowed": _ppda(
                    entry["ppda_allowed"]["att"], entry["ppda_allowed"]["def"]
                ),
                "deep": int(entry["deep"]),
                "deep_allowed": int(entry["deep_allowed"]),
                "goals": int(entry["scored"]),
                "goals_against": int(entry["missed"]),
                "xpts": float(entry["xpts"]),
            })
    return pd.DataFrame(rows)


def parse_player_seasons(data: dict, season: int) -> pd.DataFrame:
    """One row per player: their season totals, tagged with the season's start year.

    The payload doesn't say which season it belongs to, so the caller passes it in.
    `team` is kept exactly as Understat gives it (a player who moved mid-season
    appears as e.g. "Chelsea,Liverpool")."""
    rows = []
    for player in data["players"]:
        rows.append({
            "season": season,
            "player_id": player["id"],
            "player": player["player_name"],
            "team": player["team_title"],
            "position": player["position"],
            "games": int(player["games"]),
            "minutes": int(player["time"]),
            "goals": int(player["goals"]),
            "assists": int(player["assists"]),
            "shots": int(player["shots"]),
            "key_passes": int(player["key_passes"]),
            "xg": float(player["xG"]),
            "xa": float(player["xA"]),
            "npg": int(player["npg"]),
            "npxg": float(player["npxG"]),
            "xg_chain": float(player["xGChain"]),
            "xg_buildup": float(player["xGBuildup"]),
        })
    return pd.DataFrame(rows)


def parse_match_shots(data: dict) -> pd.DataFrame:
    """One row per shot, both teams together. `team` is the shooting team's name,
    so nobody has to remember what "h" and "a" stand for; `side` keeps home/away."""
    rows = []
    for side in ("h", "a"):
        for shot in data["shots"][side]:
            rows.append({
                "shot_id": shot["id"],
                "understat_match_id": shot["match_id"],
                "minute": int(shot["minute"]),
                "team": shot["h_team"] if shot["h_a"] == "h" else shot["a_team"],
                "side": shot["h_a"],
                "player_id": shot["player_id"],
                "player": shot["player"],
                "assister": shot.get("player_assisted") or None,
                "x": float(shot["X"]),
                "y": float(shot["Y"]),
                "xg": float(shot["xG"]),
                "situation": shot["situation"],
                "shot_type": shot["shotType"],
                "result": shot["result"],
                "last_action": shot.get("lastAction"),
            })
    return pd.DataFrame(rows)


def _substitution_link(value):
    """Understat writes 0 when a player has no substitution partner, otherwise the
    partner's roster id. Return the id as text, or None when there is no partner,
    so it can be compared directly with the `roster_id` column."""
    text = str(value)
    return None if text in ("", "0") else text


def parse_match_rosters(data: dict) -> pd.DataFrame:
    """One row per player Understat lists for a match. Nothing is filtered here; if
    a zero-minute player ever appears he is kept, and a minutes threshold is an
    analysis-layer decision (see config/settings.yaml's min_minutes).

    `roster_in` / `roster_out` are roster ids, not minutes: for a player who went
    off, `roster_in` is the roster id of the player who replaced him; for a player
    who came on, `roster_out` is the roster id of the player he replaced. The
    minute of the swap is the outgoing player's `minutes`."""
    rows = []
    for side in ("h", "a"):
        for entry in data["rosters"][side].values():
            rows.append({
                "roster_id": str(entry["id"]),
                "player_id": entry["player_id"],
                "player": entry["player"],
                "team_id": entry["team_id"],
                "side": entry["h_a"],
                "position": entry["position"],
                "position_order": int(entry["positionOrder"]),
                "minutes": int(entry["time"]),
                "roster_in": _substitution_link(entry["roster_in"]),
                "roster_out": _substitution_link(entry["roster_out"]),
                "goals": int(entry["goals"]),
                "own_goals": int(entry["own_goals"]),
                "shots": int(entry["shots"]),
                "key_passes": int(entry["key_passes"]),
                "assists": int(entry["assists"]),
                "xg": float(entry["xG"]),
                "xa": float(entry["xA"]),
                "xg_chain": float(entry["xGChain"]),
                "xg_buildup": float(entry["xGBuildup"]),
                "yellow_card": int(entry["yellow_card"]),
                "red_card": int(entry["red_card"]),
            })
    return pd.DataFrame(rows)