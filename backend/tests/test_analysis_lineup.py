import pandas as pd
import pytest

from backend.analysis.lineup import entry_minute, reconstruct_score, score_state_at_entry


def _shots(rows):
    return pd.DataFrame(rows, columns=["minute", "side", "result"])


def _rosters(rows):
    return pd.DataFrame(rows, columns=["roster_id", "side", "minutes", "roster_out"])


def test_reconstruct_score_at_known_minutes():
    shots = _shots([
        (10, "h", "Goal"),
        (25, "h", "MissedShots"),  # not a goal: must not count
        (40, "a", "Goal"),
        (70, "h", "Goal"),
    ])

    assert reconstruct_score(shots, minute=5) == (0, 0)
    assert reconstruct_score(shots, minute=10) == (1, 0)  # inclusive of a goal scored exactly at the cutoff
    assert reconstruct_score(shots, minute=39) == (1, 0)
    assert reconstruct_score(shots, minute=40) == (1, 1)
    assert reconstruct_score(shots, minute=90) == (2, 1)


def test_entry_minute_is_zero_for_a_starter():
    rosters = _rosters([("A", "h", 90, None)])

    assert entry_minute(rosters, "A") == 0


def test_entry_minute_is_the_outgoing_starters_own_minutes():
    # A starts and plays 60 minutes before B comes on - a single, simple swap.
    rosters = _rosters([
        ("A", "h", 60, None),
        ("B", "h", 30, "A"),
    ])

    assert entry_minute(rosters, "B") == 60


def test_entry_minute_walks_a_substitution_chain_back_to_the_starter():
    # The real pattern found in match 7432: Wijnaldum starts, subbed off at 67
    # for Lallana, who plays only 5 minutes before being subbed off himself
    # for Lovren. Lovren's entry is 67 + 5 = 72, not Lallana's own minutes (5) -
    # the bug a naive single-hop lookup would produce.
    rosters = _rosters([
        ("wijnaldum", "a", 67, None),
        ("lallana", "a", 5, "wijnaldum"),
        ("lovren", "a", 18, "lallana"),
    ])

    assert entry_minute(rosters, "lallana") == 67
    assert entry_minute(rosters, "lovren") == 72


def test_score_state_at_entry_is_positive_when_the_players_team_is_leading():
    shots = _shots([(30, "h", "Goal")])
    rosters = _rosters([
        ("A", "h", 60, None),
        ("B", "h", 30, "A"),  # comes on at minute 60, team already 1-0 up
    ])

    assert score_state_at_entry(shots, rosters, "B") == 1


def test_score_state_at_entry_is_negative_when_the_players_team_is_trailing():
    shots = _shots([(30, "a", "Goal")])
    rosters = _rosters([
        ("A", "h", 60, None),
        ("B", "h", 30, "A"),
    ])

    assert score_state_at_entry(shots, rosters, "B") == -1


def test_score_state_at_entry_follows_the_substitution_chain_correctly():
    # Same chain as above: Lovren enters at minute 72. A goal at minute 70
    # (while Lallana was still on, before Lovren's actual entry at 72) must
    # count toward the score Lovren inherits.
    shots = _shots([(70, "a", "Goal")])
    rosters = _rosters([
        ("wijnaldum", "a", 67, None),
        ("lallana", "a", 5, "wijnaldum"),
        ("lovren", "a", 18, "lallana"),
    ])

    assert score_state_at_entry(shots, rosters, "lovren") == 1


def test_a_starter_always_enters_at_a_level_score():
    shots = _shots([(10, "h", "Goal")])
    rosters = _rosters([("A", "h", 90, None)])

    assert score_state_at_entry(shots, rosters, "A") == 0
