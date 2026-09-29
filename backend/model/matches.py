import pandas as pd

from backend.model.eras import Era, era_for
from backend.model.names import NameMap

CLUB = "Liverpool"
LEAGUE = "Premier League"
TEAM_STAT_COLUMNS = ["xg", "xga", "npxg", "npxga", "ppda", "ppda_allowed", "deep", "deep_allowed"]


def build_matches(
    fbref: pd.DataFrame,
    understat_matches: pd.DataFrame,
    understat_team_stats: pd.DataFrame,
    name_map: NameMap,
    eras: list[Era],
) -> tuple[pd.DataFrame, list[str]]:
    """One row per Liverpool match, in every competition, from FBref's fixture
    list (the complete calendar). Understat's stats are attached on top, only
    for league matches, since Understat covers the Premier League alone -
    everything else (Champions League, cups) keeps its FBref basics with the
    Understat columns left missing. Every row is tagged with its manager era
    and with how many days separate it from Liverpool's previous match in any
    competition, since a cup match a few days earlier is exactly the kind of
    context a flat league performance needs.

    Nothing is ever silently dropped: an unmapped club name, or a match that
    exists in one source but not the other, is kept and reported instead."""
    problems: list[str] = []

    fb = fbref.copy()
    canonical_opponent = fb["opponent"].map(name_map.team)
    for raw in sorted(fb.loc[canonical_opponent.isna(), "opponent"].unique()):
        problems.append(f"FBref opponent not in name_map: {raw!r}")
    fb["opponent"] = canonical_opponent.fillna(fb["opponent"])

    us = understat_matches.copy()
    if not us.empty:
        us["home"] = us["home_team"].map(name_map.team)
        us["away"] = us["away_team"].map(name_map.team)
        unknown = pd.concat([
            us.loc[us["home"].isna(), "home_team"],
            us.loc[us["away"].isna(), "away_team"],
        ]).unique()
        for raw in sorted(unknown):
            problems.append(f"Understat team not in name_map: {raw!r}")
        us["home"] = us["home"].fillna(us["home_team"])
        us["away"] = us["away"].fillna(us["away_team"])

    ours = us[(us["home"] == CLUB) | (us["away"] == CLUB)].copy() if not us.empty else us
    if not ours.empty:
        ours["opponent"] = ours["away"].where(ours["home"] == CLUB, ours["home"])
        ours["competition"] = LEAGUE
    join_columns = ["understat_match_id", "date", "opponent", "competition"]
    ours_for_merge = ours[join_columns] if not ours.empty else pd.DataFrame(columns=join_columns)

    merged = fb.merge(ours_for_merge, how="left", on=["date", "opponent", "competition"])

    missing_us = merged[(merged["competition"] == LEAGUE) & merged["understat_match_id"].isna()]
    for _, row in missing_us.iterrows():
        problems.append(
            f"FBref league match {row['date']} vs {row['opponent']} has no Understat match"
        )
    matched_ids = set(merged["understat_match_id"].dropna())
    for _, row in ours.iterrows():
        if row["understat_match_id"] not in matched_ids:
            problems.append(
                f"Understat match {row['understat_match_id']} ({row['date']} vs "
                f"{row['opponent']}) has no FBref fixture"
            )

    if not understat_team_stats.empty:
        liverpool_stats = understat_team_stats[understat_team_stats["team"] == CLUB]
        merged = merged.merge(liverpool_stats[["date", *TEAM_STAT_COLUMNS]], how="left", on="date")
    else:
        for column in TEAM_STAT_COLUMNS:
            merged[column] = pd.NA
    no_stats = merged[merged["understat_match_id"].notna() & merged["xg"].isna()]
    for _, row in no_stats.iterrows():
        problems.append(f"no Understat team stats for {CLUB} on {row['date']}")

    merged["manager"] = merged["date"].map(lambda d: era_for(d, eras))
    for d in sorted(merged.loc[merged["manager"].isna(), "date"].unique()):
        problems.append(f"no manager era covers {d}")

    merged = merged.sort_values("date").reset_index(drop=True)
    merged["days_since_previous_match"] = pd.to_datetime(merged["date"]).diff().dt.days

    return merged, problems
