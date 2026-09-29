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
    # GF/GA must stay text: a season with no penalty shootouts has only plain
    # integers plus blanks (unplayed fixtures), which pandas would otherwise
    # read as a float64 column and turn "2" into "2.0", breaking parse_score.
    df = pd.read_csv(path, dtype={"GF": str, "GA": str})
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


_FILE_NAME = re.compile(r"^fbref_(\d{4}-\d{4})_fixtures\.csv$")


def load_all_fixtures(manual_dir) -> pd.DataFrame:
    """Read every manually exported season file in `manual_dir` and combine them.
    The season is read from the filename itself (fbref_<season>_fixtures.csv)."""
    frames = []
    for path in sorted(Path(manual_dir).glob("fbref_*_fixtures.csv")):
        found = _FILE_NAME.match(path.name)
        if found is None:
            raise ValueError(f"unexpected file name: {path.name}")
        frames.append(load_fixtures_csv(path, found.group(1)))
    if not frames:
        raise FileNotFoundError(f"no fbref_<season>_fixtures.csv files in {manual_dir}")
    return pd.concat(frames, ignore_index=True)
