import math

import pandas as pd
import pytest

from backend.analysis.onoff import raw_onoff

PLAYER_X = 1


def _matches(rows):
    return pd.DataFrame(rows, columns=["understat_match_id", "venue", "xg"])


def _rosters(rows):
    return pd.DataFrame(rows, columns=["understat_match_id", "side", "player_id", "minutes"])


def test_raw_onoff_splits_matches_by_whether_the_player_played_and_reports_sample_sizes():
    # Player X starts 3 Home/Away matches (higher xG in all three) and is
    # absent from 3 others (lower xG). Liverpool's side is identified by
    # matching venue (Home/Away) to the roster's side (h/a) - rosters hold
    # both teams' players, not just Liverpool's.
    matches = _matches([
        (1, "Home", 2.0), (2, "Away", 2.2), (3, "Home", 1.8),
        (4, "Home", 1.0), (5, "Away", 0.8), (6, "Home", 1.2),
    ])
    rosters = _rosters([
        (1, "h", PLAYER_X, 90), (2, "a", PLAYER_X, 90), (3, "h", PLAYER_X, 90),
        # Player X has no roster row at all for matches 4-6: not in the squad.
        # A same-numbered opponent-side row must never be mistaken for Liverpool's own player.
        (4, "a", 999, 90),
    ])

    result = raw_onoff(PLAYER_X, matches, rosters, "xg")

    assert result["with_n"] == 3
    assert result["without_n"] == 3
    assert result["with_player_mean"] == pytest.approx(2.0)
    assert result["without_player_mean"] == pytest.approx(1.0)
    assert result["difference"] == pytest.approx(1.0)


def test_raw_onoff_excludes_matches_with_no_understat_data_from_both_groups():
    # A cup match has no understat_match_id and so no roster data either -
    # it's excluded entirely, not counted as "without the player" just
    # because there's no roster row proving otherwise.
    matches = _matches([
        (1, "Home", 2.0),
        (None, "Home", 99.0),  # cup match: no understat coverage at all
        (2, "Away", 1.0),
    ])
    rosters = _rosters([(1, "h", PLAYER_X, 90)])

    result = raw_onoff(PLAYER_X, matches, rosters, "xg")

    assert result["with_n"] == 1
    assert result["without_n"] == 1


def test_raw_onoff_is_flagged_below_threshold_when_the_player_has_little_playing_time():
    # 6 team matches -> display_threshold = min(270, 45*6) = 270 minutes.
    # Player X has only 90 minutes total: below the threshold.
    matches = _matches([(i, "Home", 1.5) for i in range(1, 7)])
    rosters = _rosters([(1, "h", PLAYER_X, 30), (2, "h", PLAYER_X, 30), (3, "h", PLAYER_X, 30)])

    result = raw_onoff(PLAYER_X, matches, rosters, "xg")

    assert result["below_threshold"] is True
    # Flagged, not removed: every number is still there to look at.
    assert result["with_n"] == 3
    assert not math.isnan(result["with_player_mean"])


def test_raw_onoff_is_not_flagged_once_the_player_clears_the_threshold():
    # 3 team matches -> display_threshold = min(270, 45*3) = 135 minutes.
    # Player X played all 270 minutes available: comfortably above it.
    matches = _matches([(i, "Home", 1.5) for i in range(1, 4)])
    rosters = _rosters([(1, "h", PLAYER_X, 90), (2, "h", PLAYER_X, 90), (3, "h", PLAYER_X, 90)])

    result = raw_onoff(PLAYER_X, matches, rosters, "xg")

    assert result["below_threshold"] is False
