from pathlib import Path

import duckdb
import pandas as pd


def _read(db_path, table: str) -> pd.DataFrame:
    """Read one table from the database. The connection is read-only, so
    analysis can never change the data it's given."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.sql(f"SELECT * FROM {table}").df()
    finally:
        con.close()


def load_matches(db_path: Path | str) -> pd.DataFrame:
    return _read(db_path, "matches")


def load_league_matches(db_path: Path | str) -> pd.DataFrame:
    """Every team's league match list, not just Liverpool's - what the
    strength model is fitted on and the baselines are judged against."""
    return _read(db_path, "league_matches")


def load_shots(db_path: Path | str) -> pd.DataFrame:
    return _read(db_path, "shots")


def load_rosters(db_path: Path | str) -> pd.DataFrame:
    return _read(db_path, "rosters")


def load_player_seasons(db_path: Path | str) -> pd.DataFrame:
    return _read(db_path, "player_season_stats")
