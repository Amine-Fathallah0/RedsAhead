"""Match diagnosis: how unusual was one match, compared to Liverpool's own
recent form under the same manager - not compared across eras, since a
different manager's tactics make a different baseline normal."""

import math

import pandas as pd

BASELINE_MATCHES = 10
MIN_BASELINE_MATCHES = 5
DIAGNOSIS_COLUMNS = ["xg", "ppda", "deep"]


def _match_on(matches: pd.DataFrame, date) -> pd.Series:
    cutoff = pd.Timestamp(date)
    todays = matches[pd.to_datetime(matches["date"]) == cutoff]
    if todays.empty:
        raise ValueError(f"no match on {date}")
    return todays.iloc[0]


def era_baseline(matches: pd.DataFrame, date) -> pd.DataFrame | None:
    """The last 10 matches strictly before `date`, under the same manager
    as the match on `date` itself - never a different era, no matter how
    recent. None if fewer than 5 such matches exist yet.

    Cup and European rows have no Understat stats at all (xg/ppda/deep
    are all NaN - Understat covers the Premier League only), so they're
    excluded before windowing. Including them wouldn't just add noise:
    pandas' mean/std silently skip NaN, so a window that looks like 10
    matches could rest on far fewer real data points than the minimum-5
    check believes it counted."""
    match = _match_on(matches, date)
    cutoff = pd.Timestamp(date)
    dates = pd.to_datetime(matches["date"])
    has_stats = matches[DIAGNOSIS_COLUMNS].notna().all(axis=1)
    same_era = matches[has_stats & (matches["manager"] == match["manager"]) & (dates < cutoff)].copy()
    same_era["_sort_date"] = dates[same_era.index]
    same_era = same_era.sort_values("_sort_date").tail(BASELINE_MATCHES).drop(columns="_sort_date")

    if len(same_era) < MIN_BASELINE_MATCHES:
        return None
    return same_era


def z_scores(matches: pd.DataFrame, date) -> pd.Series | None:
    """How many standard deviations the match on `date` sits from its era
    baseline, for xG, PPDA and deep completions - ranked by the size of
    the deviation, largest first, regardless of direction. None if there
    isn't yet a baseline to compare against."""
    baseline = era_baseline(matches, date)
    if baseline is None:
        return None
    match = _match_on(matches, date)

    scores = {}
    for column in DIAGNOSIS_COLUMNS:
        mean = baseline[column].mean()
        std = baseline[column].std()  # sample std (ddof=1): the baseline is a sample of the era, not the whole of it
        scores[column] = (match[column] - mean) / std if std > 0 else math.nan

    result = pd.Series(scores)
    return result.reindex(result.abs().sort_values(ascending=False).index)
