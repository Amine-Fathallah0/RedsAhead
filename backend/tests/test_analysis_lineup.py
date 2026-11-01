import numpy as np
import pandas as pd
import pytest

from backend.analysis.lineup import (
    bootstrap_player_effects,
    build_lineup_design_matrix,
    counterfactual_effect,
    entry_minute,
    fit_ridge,
    opponent_net_strength,
    predict_lineup_outcome,
    reconstruct_score,
    score_state_at_entry,
    player_lineup_effects,
    score_state_control,
    select_ridge_alpha,
    _expanding_folds,
    _gated_player_effects,
)
from backend.analysis.strength import Ratings
from backend.model.names import NameMap


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


def test_fit_ridge_recovers_a_planted_player_effect_in_sign_and_rough_size():
    # Player A has a genuine positive effect of 0.8 on the xG difference,
    # on top of two controls that have their own, smaller influence. Ridge
    # shrinks every coefficient somewhat, so the planted 0.8 is not
    # expected back exactly - just the right sign, and in the right
    # neighbourhood rather than crushed to ~0 or wildly overshot.
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame({
        "player_a": rng.integers(0, 2, n).astype(float),
        "opponent_strength": rng.normal(0, 1, n),
        "is_home": rng.integers(0, 2, n).astype(float),
    })
    true_effect = 0.8
    noise = rng.normal(0, 0.3, n)
    y = (
        0.2 * X["opponent_strength"]
        + 0.3 * X["is_home"]
        + true_effect * X["player_a"]
        + noise
    )

    coef = fit_ridge(X, y, alpha=1.0)

    assert coef["player_a"] > 0
    assert 0.4 < coef["player_a"] < 1.2


def test_counterfactual_effect_equals_the_prediction_difference():
    # Deliberately NOT a tautology: counterfactual_effect must agree with
    # two separately-made predict_lineup_outcome calls, one per flag value -
    # if a future change (e.g. an interaction term) ever made the model
    # non-linear, this would catch counterfactual_effect drifting from
    # what actually flipping the flag produces.
    coef = pd.Series({"intercept": 1.0, "player_a": 0.8, "opponent_strength": 0.2})
    row = pd.Series({"player_a": 0.0, "opponent_strength": -0.5})

    with_player = row.copy()
    with_player["player_a"] = 1.0
    without_player = row.copy()
    without_player["player_a"] = 0.0
    expected = predict_lineup_outcome(coef, with_player) - predict_lineup_outcome(coef, without_player)

    assert counterfactual_effect(coef, row, "player_a") == pytest.approx(expected)
    assert counterfactual_effect(coef, row, "player_a") == pytest.approx(0.8)


def test_bootstrap_player_effects_gives_a_rare_player_a_wider_interval():
    # A regular player appears in roughly half of 150 matches; a rare
    # player appears in only 5. Both have the same true effect size, so
    # any width difference in the bootstrap intervals comes from how
    # little evidence backs the rare player's coefficient, not from a
    # different planted effect.
    rng = np.random.default_rng(1)
    n = 150
    regular = rng.integers(0, 2, n).astype(float)
    rare = np.zeros(n)
    rare[rng.choice(n, size=5, replace=False)] = 1.0
    X = pd.DataFrame({
        "regular_player": regular,
        "rare_player": rare,
        "opponent_strength": rng.normal(0, 1, n),
    })
    y = 0.5 * regular + 0.5 * rare + 0.2 * X["opponent_strength"] + rng.normal(0, 0.3, n)

    effects = bootstrap_player_effects(
        X, y, player_columns=["regular_player", "rare_player"],
        alpha=1.0, n_bootstrap=300, rng=np.random.default_rng(2),
    )

    regular_width = effects.loc["regular_player", "ci_high"] - effects.loc["regular_player", "ci_low"]
    rare_width = effects.loc["rare_player", "ci_high"] - effects.loc["rare_player", "ci_low"]
    assert rare_width > regular_width


def test_score_state_control_weighs_each_spell_by_its_own_duration():
    # Level for 30 minutes, ahead 1-0 for the next 30, level again (1-1)
    # for the final 30: the final score is level, but the match still
    # spent a third of its minutes ahead, which the control must reflect.
    shots = _shots([(30, "h", "Goal"), (60, "a", "Goal")])

    assert score_state_control(shots, side="h") == pytest.approx(30 / 90)


def test_score_state_control_is_positive_for_a_match_spent_mostly_ahead():
    # Leading 1-0 from minute 10 to the final whistle: level for 10 of 90
    # minutes, ahead for 80 - a minutes-weighted average close to +0.89,
    # not the naive final-score-only answer of +1.
    shots = _shots([(10, "h", "Goal")])

    assert score_state_control(shots, side="h") == pytest.approx(80 / 90)


def test_score_state_control_flips_sign_by_side():
    shots = _shots([(10, "h", "Goal")])

    assert score_state_control(shots, side="a") == pytest.approx(-80 / 90)


def test_score_state_control_caps_stoppage_time_goals_at_the_match_length():
    # A goal at minute 95 (second-half stoppage) must not count as if the
    # team led for -5 minutes of "extra" normal time - it's clipped to 90,
    # contributing a zero-length interval at the very end.
    shots = _shots([(95, "h", "Goal")])

    assert score_state_control(shots, side="h") == pytest.approx(0.0)


def test_opponent_net_strength_resolves_understat_aliases_through_the_name_map():
    # strength.fit keys its ratings by whatever name the Understat-sourced
    # league history used ("Tottenham"); the opponent name passed in here
    # is matches.py's canonical FBref name ("Tottenham Hotspur") - the
    # same mismatch build_matches() already resolves via the name map.
    name_map = NameMap({"Tottenham Hotspur": ["Tottenham", "Spurs"], "Liverpool": []})
    ratings = Ratings(base=0.0, home_advantage=0.0, attack={"Tottenham": 0.5}, defence={"Tottenham": -0.2})

    assert opponent_net_strength(ratings, "Tottenham Hotspur", name_map) == pytest.approx(0.7)


def test_opponent_net_strength_is_zero_for_a_team_with_no_rating_yet():
    name_map = NameMap({"Tottenham Hotspur": ["Tottenham"], "Liverpool": []})
    ratings = Ratings(base=0.0, home_advantage=0.0)

    assert opponent_net_strength(ratings, "Tottenham Hotspur", name_map) == 0.0


def test_expanding_folds_never_validates_on_a_row_before_its_training_rows():
    folds = list(_expanding_folds(n=100, n_folds=4, min_train=40))

    assert len(folds) == 4
    for train_slice, valid_slice in folds:
        assert train_slice.start == 0
        assert valid_slice.start == train_slice.stop  # no gap, and no overlap
    assert folds[-1][1].stop == 100  # the last fold's validation block reaches the end
    # each fold's training block grows - never shrinks, never repeats a split
    train_ends = [train_slice.stop for train_slice, _ in folds]
    assert train_ends == sorted(set(train_ends))


def test_select_ridge_alpha_prefers_a_larger_penalty_for_noisier_data():
    # With a real signal but heavy noise, a small alpha overfits every
    # fold's training noise; cross-validated held-out error should favour
    # a larger alpha here, not the smallest one in the grid.
    rng = np.random.default_rng(3)
    n = 200
    X = pd.DataFrame({
        "player_a": rng.integers(0, 2, n).astype(float),
        "opponent_strength": rng.normal(0, 1, n),
    })
    y = 0.3 * X["player_a"] + 0.2 * X["opponent_strength"] + rng.normal(0, 5.0, n)

    chosen = select_ridge_alpha(X, y, alphas=(0.01, 1.0, 1000.0), n_folds=4, min_train=60)

    assert chosen in (0.01, 1.0, 1000.0)
    assert chosen != 0.01


def test_build_lineup_design_matrix_assembles_the_expected_columns_and_rows():
    # Two Liverpool matches: p1 plays both, p2 only the first, p3 only the
    # second. Both matches share the same manager, so no era dummy should
    # be created at all - one category gives nothing to compare against.
    matches = pd.DataFrame([
        {
            "understat_match_id": "m1", "date": pd.Timestamp("2024-08-10"), "venue": "Home",
            "opponent": "Chelsea", "manager": "Klopp", "days_since_previous_match": 7.0,
            "xg": 2.0, "xga": 1.0,
        },
        {
            "understat_match_id": "m2", "date": pd.Timestamp("2024-08-17"), "venue": "Away",
            "opponent": "Tottenham Hotspur", "manager": "Klopp", "days_since_previous_match": 7.0,
            "xg": 1.0, "xga": 1.5,
        },
        {
            # No Understat coverage (a cup match) - must be excluded entirely.
            "understat_match_id": None, "date": pd.Timestamp("2024-08-13"), "venue": "Home",
            "opponent": "Everton", "manager": "Klopp", "days_since_previous_match": 3.0,
            "xg": 1.0, "xga": 1.0,
        },
    ])
    rosters = pd.DataFrame([
        {"understat_match_id": "m1", "side": "h", "player_id": "p1", "minutes": 90},
        {"understat_match_id": "m1", "side": "h", "player_id": "p2", "minutes": 90},
        {"understat_match_id": "m2", "side": "a", "player_id": "p1", "minutes": 90},
        {"understat_match_id": "m2", "side": "a", "player_id": "p3", "minutes": 90},
    ])
    shots = pd.DataFrame([
        {"understat_match_id": "m1", "minute": 10, "side": "h", "result": "Goal"},
    ], columns=["understat_match_id", "minute", "side", "result"])
    league_matches = pd.DataFrame([
        {"date": pd.Timestamp("2024-08-01"), "home_team": "Everton", "away_team": "Tottenham",
         "home_xg": 1.0, "away_xg": 1.2},
        {"date": pd.Timestamp("2024-08-05"), "home_team": "Tottenham", "away_team": "Chelsea",
         "home_xg": 1.5, "away_xg": 0.8},
    ])
    name_map = NameMap({"Chelsea": [], "Tottenham Hotspur": ["Tottenham"], "Everton": []})

    X, y, player_columns = build_lineup_design_matrix(matches, rosters, shots, league_matches, name_map)

    assert sorted(player_columns) == ["player_p1", "player_p2", "player_p3"]
    assert list(X["player_p1"]) == [1.0, 1.0]
    assert list(X["player_p2"]) == [1.0, 0.0]
    assert list(X["player_p3"]) == [0.0, 1.0]
    assert list(X["is_home"]) == [1.0, 0.0]
    assert list(X["rest_days"]) == [7.0, 7.0]
    assert not any(col.startswith("era_") for col in X.columns)  # one manager: no dummy needed
    assert X["opponent_strength"].notna().all()
    assert X["score_state_control"].notna().all()
    assert list(y) == [1.0, -0.5]
    assert len(X) == 2  # the cup match (no Understat id) is excluded


def test_gated_player_effects_reports_effects_when_the_player_model_wins():
    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame({
        "player_a": rng.integers(0, 2, n).astype(float),
        "opponent_strength": rng.normal(0, 1, n),
        "is_home": rng.integers(0, 2, n).astype(float),
    })
    y = 0.8 * X["player_a"] + 0.3 * X["opponent_strength"] + rng.normal(0, 0.3, n)

    result = _gated_player_effects(
        X, y, ["player_a"], alphas=(0.1, 1, 10, 100), n_folds=5, min_train=80,
        n_bootstrap=200, rng=np.random.default_rng(1),
    )

    assert result["beats_no_player_model"] is True
    assert result["full_model_rmse"] < result["no_player_model_rmse"]
    assert isinstance(result["effects"], pd.DataFrame)
    assert "player_a" in result["effects"].index


def test_gated_player_effects_withholds_effects_when_players_are_just_noise():
    # 15 player flags with no real relationship to y: the full model has
    # strictly more ways to overfit than the reduced one, with nothing
    # genuine to show for it, so it should not come out ahead.
    rng = np.random.default_rng(7)
    n = 150
    noise_players = {f"player_{i}": rng.integers(0, 2, n).astype(float) for i in range(15)}
    X = pd.DataFrame({**noise_players, "opponent_strength": rng.normal(0, 1, n)})
    y = 0.3 * X["opponent_strength"] + rng.normal(0, 1.0, n)

    result = _gated_player_effects(
        X, y, list(noise_players.keys()), alphas=(0.1, 1, 10, 100), n_folds=5, min_train=60,
        n_bootstrap=50, rng=np.random.default_rng(2),
    )

    assert result["beats_no_player_model"] is False
    assert result["effects"] is None


@pytest.mark.integration
def test_player_lineup_effects_on_real_data():
    # Slow (~a few minutes): opponent_strength refits the strength model
    # once per covered match (~340+), each on a growing slice of nine
    # seasons of league history - the one-time cost of building the
    # design matrix, not of the ridge fitting or bootstrap themselves.
    from backend.analysis.data import load_league_matches, load_matches, load_rosters, load_shots
    from backend.config import load_settings
    from backend.model.names import load_name_map

    db_path = load_settings().db_path
    matches = load_matches(db_path)
    rosters = load_rosters(db_path)
    shots = load_shots(db_path)
    league_matches = load_league_matches(db_path)
    name_map = load_name_map("config/name_map.yaml")

    result = player_lineup_effects(
        matches, rosters, shots, league_matches, name_map,
        n_folds=5, min_train=150, n_bootstrap=1000, rng=np.random.default_rng(0),
    )

    print(
        f"\nfull_model_rmse={result['full_model_rmse']:.4f} "
        f"no_player_model_rmse={result['no_player_model_rmse']:.4f} "
        f"beats_no_player_model={result['beats_no_player_model']} alpha={result['alpha']}"
    )
    assert result["full_model_rmse"] > 0
    assert result["no_player_model_rmse"] > 0
    if result["beats_no_player_model"]:
        assert isinstance(result["effects"], pd.DataFrame)
        print(result["effects"].sort_values("point_effect", ascending=False).head(10))
    else:
        assert result["effects"] is None
