from datetime import date

import pandas as pd
import pytest

from backend.analysis.validation import (
    brier_score,
    held_out_predictions,
    match_outcome_probabilities,
    rmse,
)

HISTORY = pd.DataFrame(
    [
        (date(2025, 8, 1), "A", "B", 2, 0, 1.5, 0.5),
        (date(2025, 8, 8), "C", "D", 1, 1, 1.0, 1.0),
        (date(2025, 8, 15), "B", "C", 0, 3, 0.8, 2.0),
        (date(2025, 8, 22), "D", "A", 0, 2, 0.4, 1.6),
    ],
    columns=["date", "home_team", "away_team", "home_goals", "away_goals", "home_xg", "away_xg"],
)


def test_a_predictor_that_tries_to_peek_at_its_own_match_cannot_find_it():
    # This predictor tries to cheat by looking up its own answer in the
    # training history it's handed. In our fixture, every home/away pairing
    # appears exactly once — the one occurrence being the match in question —
    # so if the harness correctly withholds that match, the lookup always
    # comes up empty and the predictor is forced back onto a wrong guess.
    # If the harness ever leaked the current match in, every lookup would
    # succeed and this would score a suspicious, perfect zero.
    def peeking_predictor(training_history, home, away):
        row = training_history[
            (training_history["home_team"] == home) & (training_history["away_team"] == away)
        ]
        if row.empty:
            return 0.0, 0.0  # couldn't peek: falls back to an uninformed guess
        return row.iloc[0]["home_xg"], row.iloc[0]["away_xg"]

    predictions = held_out_predictions(HISTORY, peeking_predictor)

    assert rmse(predictions) > 0.1


def test_the_harness_only_passes_matches_dated_before_the_one_being_predicted():
    seen_histories = []

    def recording_predictor(training_history, home, away):
        seen_histories.append(training_history.copy())
        return 1.0, 1.0

    held_out_predictions(HISTORY, recording_predictor)

    # Four matches predicted, so four calls, each with strictly earlier matches.
    assert len(seen_histories) == 4
    for predicted_date, training_history in zip(HISTORY["date"], seen_histories):
        assert (training_history["date"] < predicted_date).all()
    # The first match has no history before it at all.
    assert seen_histories[0].empty
    # The last match has seen the three before it, never itself.
    assert len(seen_histories[-1]) == 3


def test_rmse_of_a_perfect_prediction_is_zero():
    predictions = pd.DataFrame({"actual_xg": [1.0, 2.0], "predicted_xg": [1.0, 2.0]})

    assert rmse(predictions) == pytest.approx(0.0)


def test_rmse_matches_a_hand_worked_example():
    # Errors of 1 and -1: squared errors 1 and 1, mean 1, root 1.
    predictions = pd.DataFrame({"actual_xg": [1.0, 2.0], "predicted_xg": [2.0, 1.0]})

    assert rmse(predictions) == pytest.approx(1.0)


def test_outcome_probabilities_sum_to_one():
    p_home, p_draw, p_away = match_outcome_probabilities(1.9, 0.8)

    assert p_home + p_draw + p_away == pytest.approx(1.0)


def test_the_stronger_side_gets_most_of_the_win_probability():
    p_home, p_draw, p_away = match_outcome_probabilities(1.9, 0.8)

    assert p_home > 0.5
    assert p_home > p_draw > p_away


def test_two_low_scoring_sides_make_the_draw_the_likeliest_outcome():
    p_home, p_draw, p_away = match_outcome_probabilities(0.3, 0.3)

    assert p_draw > p_home
    assert p_draw > p_away


def test_outcome_probabilities_are_symmetric():
    p_home, p_draw, p_away = match_outcome_probabilities(1.2, 0.6)
    q_home, q_draw, q_away = match_outcome_probabilities(0.6, 1.2)

    assert p_home == pytest.approx(q_away)
    assert p_away == pytest.approx(q_home)
    assert p_draw == pytest.approx(q_draw)


def test_brier_score_is_zero_for_a_perfectly_confident_correct_call():
    predictions = pd.DataFrame({
        "p_home": [1.0],
        "p_draw": [0.0],
        "p_away": [0.0],
        "result": ["home"],
    })

    assert brier_score(predictions) == pytest.approx(0.0)


def test_brier_score_penalises_confident_wrong_calls_more_than_unsure_ones():
    confident_and_wrong = pd.DataFrame({
        "p_home": [1.0], "p_draw": [0.0], "p_away": [0.0], "result": ["away"],
    })
    unsure_and_wrong = pd.DataFrame({
        "p_home": [0.4], "p_draw": [0.3], "p_away": [0.3], "result": ["away"],
    })

    assert brier_score(confident_and_wrong) > brier_score(unsure_and_wrong)
