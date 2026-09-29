import re
from pathlib import Path

import pandas as pd

_SCORE = re.compile(r"^(\d+)(?:\s*\((\d+)\))?$")

REQUIRED_COLUMNS = [
    "Date", "Comp", "Venue", "Result", "GF", "GA", "Opponent", "Poss",
    "Formation", "Opp Formation",
]


def parse_score(value: str) -> tuple[int, int | None]:
    """Split an FBref GF/GA value into (goals, penalties).

    Most matches are just "2". A match settled on penalties is written
    "2 (2)": the regulation-time score, then the shootout score in
    parentheses. `penalties` is None when there was no shootout, never 0,
    since 0 would wrongly claim "a shootout happened and nobody scored"."""
    found = _SCORE.match(value)
    if found is None:
        raise ValueError(f"unrecognised score: {value!r}")
    goals, penalties = found.groups()
    return int(goals), None if penalties is None else int(penalties)


def load_fixtures_csv(path, season: str) -> pd.DataFrame:
    """Read one manually exported FBref 'Scores & Fixtures' CSV. Only played
    matches are kept (a row with no Result is a fixture still to be played)."""
    path = Path(path)
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()   # FBref sometimes exports a leading
                                            # space before the first header
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")

    df = df[df["Result"].notna()].reset_index(drop=True)
    goals_for, penalties_for = zip(*(parse_score(str(v)) for v in df["GF"]))
    goals_against, penalties_against = zip(*(parse_score(str(v)) for v in df["GA"]))

    return pd.DataFrame({
        "season": season,
        "date": pd.to_datetime(df["Date"]).dt.date,
        "competition": df["Comp"],
        "venue": df["Venue"],
        "opponent": df["Opponent"],
        "result": df["Result"],
        "goals_for": goals_for,
        "goals_against": goals_against,
        "penalties_for": penalties_for,
        "penalties_against": penalties_against,
        "possession": pd.to_numeric(df["Poss"], errors="coerce"),
        "formation": df["Formation"],
        "opp_formation": df["Opp Formation"],
    })
