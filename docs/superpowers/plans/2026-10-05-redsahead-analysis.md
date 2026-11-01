# RedsAhead Analysis Implementation Plan (Plan 2 of 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the DuckDB database into validated answers: a team-strength model that must beat simple baselines before anything depends on it, then match diagnosis, era comparison, player impact (raw on/off and adjusted lineup effects with a score-state control), and scouting.

**Architecture:** A new `backend/analysis/` package of pure functions. The only place that touches the database is `data.py`, which loads tables into pandas. Analysis never touches the network. The strength model is the foundation: form, match diagnosis, and player impact all read its ratings. The validation harness is a hard gate: no downstream task starts until the attack/defence model beats every baseline on held-out matches (Task 5).

**Tech Stack:** Python 3.13, pandas, numpy, scipy (for the fitting and tests), DuckDB (read-only), pytest.

**Spec:** `docs/superpowers/specs/2026-09-19-redsahead-design.md` (sections 4 and 5)

## Global Constraints

- Analysis never touches the network. It reads the DuckDB file or small fixtures.
- Every claim about a player's effect carries a "correlation, not causation" label and its sample size.
- Display thresholds are adaptive: `threshold = min(270, 45 × team matches played)` minutes. Below the threshold a result is greyed out with a warning, never hidden. The model keeps every player who appeared.
- Baseline window for match diagnosis: the last 10 matches in the same manager era, with a minimum of 5 before a baseline is shown.
- Form uses league matches only (Understat has xG for Premier League matches only).
- Every prediction uses only matches played before it. Any test that lets a model see the match it predicts is a bug.
- Tests run offline on small fixtures. Tests that read the real database are marked `integration`, as in Plan 1.
- Commits are made by the user, one logical change each, with messages proposed by Claude.

---

## File Structure

```
backend/analysis/
  __init__.py
  data.py          # the only database access: load_matches(), load_league_matches(), load_shots(), load_rosters(), load_player_seasons()
  thresholds.py    # display_threshold(matches_played) -> minutes
  baselines.py     # league_average, last_five_results, elo_ratings: the predictors the model must beat
  validation.py    # held_out_predictions(): predict each match using only earlier matches; error metrics
  strength.py      # attack/defence xG model: fit(history, half_life, penalty) -> ratings
  form.py          # form(team, date): points minus expected points over the last five league matches
  diagnosis.py     # era_baseline(), z_scores() for a single match
  era.py           # era_means(), bootstrap_ci(), permutation_test()
  onoff.py         # raw_onoff(): team numbers with and without a player, with sample sizes
  lineup.py        # score_state(), adjusted_lineup_effects(): ridge model with bootstrap intervals and counterfactuals
  scouting.py      # percentile_ranks(), similar_players(): cosine similarity across all 20 clubs
backend/tests/
  test_analysis_thresholds.py   test_analysis_baselines.py   test_analysis_validation.py
  test_analysis_strength.py     test_analysis_form.py        test_analysis_diagnosis.py
  test_analysis_era.py          test_analysis_onoff.py       test_analysis_lineup.py
  test_analysis_scouting.py     test_analysis_real_data.py   (integration)
```

---

### Task 1: Data access and the adaptive threshold

**Files:**
- Create: `backend/analysis/__init__.py`, `backend/analysis/data.py`, `backend/analysis/thresholds.py`
- Test: `backend/tests/test_analysis_thresholds.py`

**Interfaces:**
- Produces: `display_threshold(matches_played: int) -> int` returning `min(270, 45 * matches_played)`.
- Produces: `load_matches(db_path) -> DataFrame`, and the same pattern for `load_team_matches`, `load_shots`, `load_rosters`, `load_player_seasons`. Each opens the database read-only and returns a DataFrame.

- [ ] **Step 1: Write the failing threshold tests.** Cover: 0 matches gives 0; 1 match gives 45; 3 matches gives 135; 6 or more matches gives 270; 38 matches gives 270.
- [ ] **Step 2: Run them and confirm they fail** with `ModuleNotFoundError`.
- [ ] **Step 3: Implement `display_threshold`** as a one-line `min` expression.
- [ ] **Step 4: Write a failing integration test** that `load_matches` returns 510 rows from the real database.
- [ ] **Step 5: Implement the `data.py` loaders**, each reading the DuckDB file with `read_only=True`.
- [ ] **Step 6: Run the tests** and confirm green.
- [ ] **Step 7: Propose the commit message** `feat: analysis data access and adaptive display threshold`.

---

### Task 2: The baselines the model must beat

**Files:**
- Create: `backend/analysis/baselines.py`
- Test: `backend/tests/test_analysis_baselines.py`

**Interfaces:**
- Produces: `league_average(history) -> float`: the mean xG per team-match across all history.
- Produces: `last_five_results(team, history, date) -> float`: the mean points per game over that team's last five league matches before `date`.
- Produces: `elo_ratings(history) -> dict[str, float]` and `elo_expected(rating_a, rating_b) -> float`, a standard Elo update after each match.

- [ ] **Step 1: Write failing tests on a four-match hand-made history.** Check league average, the last-five mean on a team with five prior matches, and that Elo moves up after a win and down after a loss.
- [ ] **Step 2: Confirm the failures.**
- [ ] **Step 3: Implement the three functions.**
- [ ] **Step 4: Confirm green**, then propose the commit `feat: baseline predictors for the validation harness`.

---

### Task 3: The validation harness

**Files:**
- Create: `backend/analysis/validation.py`
- Test: `backend/tests/test_analysis_validation.py`

**Interfaces:**
- Produces: `held_out_predictions(history, predictor) -> DataFrame` with one row per match: the actual xG and the predicted xG, using only matches dated before it.
- Produces: `rmse(predictions) -> float`, the error metric for xG-predicting baselines (league average, the strength model).
- Produces: `match_outcome_probabilities(home_xg, away_xg) -> (p_home, p_draw, p_away)`: the Poisson scoreline conversion — treat each side's predicted xG as a Poisson rate, sum every scoreline's probability by who wins. This turns any xG prediction into the same kind of answer Elo gives, with no invented parameters.
- Produces: `brier_score(predictions) -> float`: how far a stated win/draw/loss probability was from the actual result. General-purpose 3-way scoring infrastructure (useful later, e.g. in match diagnosis) — not what Task 5 ends up using to judge Elo, since Elo's `elo_expected` is a single expected-score number (a draw already counts as half a win, chess-Elo style), not a 3-way split. Forcing one onto Elo would mean inventing a draw probability it was never built to produce.

- [ ] **Step 1: Write a leakage test.** Make a predictor that reads the match it is predicting. The harness must refuse it, or the test must show its error is not zero. This is what makes the gate honest.
- [ ] **Step 2: Write a test that the harness uses only earlier matches**, by checking the training history it passes to the predictor for one match.
- [ ] **Step 3: Write tests for `match_outcome_probabilities`**: probabilities sum to 1; a side with much higher xG gets most of the win probability; a 0-0-ish pairing (both low xG) gives draw the biggest share.
- [ ] **Step 4: Write a test for `brier_score`** on a hand-made set of predictions and actual results.
- [ ] **Step 5: Confirm failures, then implement** the harness, the Poisson conversion, and the Brier score.
- [ ] **Step 6: Confirm green**, then propose the commit `feat: held-out validation harness with outcome scoring`.

---

### Task 3b: Persist league-wide match history

**Why:** `build.py` already fetches every team's match list per season (`understat_matches`, from `parse_league_matches`) to join against Liverpool's own fixtures, then discards it. Task 4's fit and Task 5's gate both need real matches for every team Liverpool's opponents are drawn from, not just Liverpool's 510. No new scraping is needed — only persisting what's already fetched.

**Files:**
- Edit: `backend/model/build.py` (one more `CREATE OR REPLACE TABLE`)
- Edit: `backend/analysis/data.py` (one more loader)
- Test: `backend/tests/test_analysis_data.py` (one more integration case)

**Interfaces:**
- Produces: `load_league_matches(db_path) -> DataFrame`, the full-league match list (`date`, `home_team`, `away_team`, `home_goals`, `away_goals`, `home_xg`, `away_xg`) across every stored season — the same shape Task 2's baselines and Task 3's validation harness already use on fixtures.

- [ ] **Step 1: Add the table write** in `build.py`: `CREATE OR REPLACE TABLE league_matches AS SELECT * FROM understat_matches`.
- [ ] **Step 2: Rerun `python -m backend.model.build`** and record the resulting row count.
- [ ] **Step 3: Write the integration test** against that count, add `load_league_matches` to `data.py`, confirm green.
- [ ] **Step 4: Propose the commit** `feat: persist league-wide match history for the strength model`.

---

### Task 4: The attack/defence xG model

**Files:**
- Create: `backend/analysis/strength.py`
- Test: `backend/tests/test_analysis_strength.py`

**Interfaces:**
- Produces: `fit(history, half_life_days, penalty) -> Ratings`, where `Ratings` holds an attack number, a defence number, and a home-advantage adjustment for each team, plus the league-wide base and home-advantage terms.
- Produces: `predict_xg(ratings, home, away) -> (float, float)`: expected xG for the home and away side.

**Model:** the home side's expected xG is `exp(base + home_advantage + home_advantage_adjustment[home] + attack[home] - defence[away])`; the away side's is `exp(base + attack[away] - defence[home])` — no home terms apply to the side not playing at home. Each match is weighted by `0.5 ** (age_in_days / half_life_days)`, measured in calendar days (not matches) so the summer gap between seasons - when squads actually change - counts as the long stretch it is, rather than as "one match" of staleness. Attack, defence, and the per-team home-advantage adjustment are all fitted with the same L2 penalty, toward zero. A team with few matches gets pulled toward an adjustment of zero — it just gets the league-wide `home_advantage` like everyone else — while a team with enough matches and a genuinely lopsided split (strong at home, poor away, the way Carrick's Manchester United were) can show a large adjustment, because the data earns it rather than the model assuming it.

Short-term fixture congestion (a team playing again two days after its last match) is a different effect - how tired a team was for one specific match, not how stale old data is - and stays out of this model. It's already captured as `days_since_previous_match` in the `matches` table, for a later task to use.

- [x] **Step 1: Write a recovery test on synthetic data.** Build a league where team A's attack is known to be higher, simulate xG from the model, and check the fit recovers the ordering.
- [x] **Step 2: Write a recovery test for the home-advantage adjustment.** Build a team with an exaggerated home/away split in synthetic data and two teams with a normal one; check the fit clearly separates the exaggerated team from the other two, and the two normal teams - built identically - land close to each other. (`home_advantage` and the per-team adjustment trade off freely against each other, so only their *relative* values are identifiable, not each team's absolute adjustment against a true zero - confirmed on real numbers before writing the assertion this way.)
- [x] **Step 3: Write a test that the recency weighting works.** A team that improves midway should rate higher after the change than a flat-weighted fit says.
- [x] **Step 4: Write a test that the penalty shrinks ratings.** A larger penalty must move every rating — attack, defence, and the home-advantage adjustment alike — toward zero.
- [x] **Step 5: Confirm failures, implement the fit** using a Poisson-style regression with weights and penalty (scipy's optimiser).
- [x] **Step 6: Confirm green**, then propose the commit `feat: attack/defence xG strength model with per-team home advantage`.

---

### Task 5: THE GATE (real data)

**Files:**
- Script (one-time, not part of the pytest suite): the grid search over every `(half_life_days, penalty)` combination — this is a multi-hour run, not something CI or a normal integration-test sweep should ever execute.
- Create: `backend/tests/test_analysis_real_data.py` (integration) — a **fast** permanent test, fixed to the hyperparameters the grid search chose, so the gate stays checkable without re-running the search.

**Procedure:**
- Held-out matches: every league match from 2024/25 onward (August 2024 by date, no separate season tag needed — there's a clean close season with no Premier League matches in June/July, so "August or later → that year's season" is unambiguous from the date alone).
- Refit once per distinct match **date** in that window, not once per match — several Premier League matches share a date, and (per the leakage fix above) all of them correctly get the same training history anyway. Cheap baselines (`league_average`, `elo_ratings`, `last_five_results`) are computed once per date too, independent of the grid, since they don't depend on `half_life_days`/`penalty`.
- Grid-search the half-life in days (14, 30, 60, 120, unlimited) and the penalty, choosing the pair with the lowest held-out xG error. Record the chosen values.
- Convert the chosen model's predicted xG to win/draw/loss probabilities with `match_outcome_probabilities`, then reduce that to whichever single number each baseline is being judged against: an **expected score** (`p_home + 0.5×p_draw`) for the Elo comparison — Elo's own native currency, not a 3-way split — and **expected points** (`3×p_home + 1×p_draw`) for the last-five comparison, matching the scale `last_five_results` already returns.

**Gate rule, in two parts — each baseline is judged on the question it actually answers:**
1. **xG accuracy:** the attack/defence model's held-out RMSE against actual xG must be lower than `league_average`'s.
2. **Outcome accuracy:** the model's expected-score RMSE (reusing `validation.rmse`) against the actual result (1 win / 0.5 draw / 0 loss) must be lower than Elo's own expected-score RMSE from `elo_expected`. The model's expected-points RMSE against points actually earned must also be lower than `last_five_results`'s.

If any comparison fails, stop. Do not proceed to Task 6. Report the numbers to the user and revisit the model.

- [x] **Step 1: Write and run the grid-search script** (not a pytest test) over every half-life/penalty combination, in the background given the expected runtime.
- [x] **Step 2: From its output**, write `test_analysis_real_data.py` as a fast integration test fixed to the winning hyperparameters, asserting all three comparisons pass.
- [x] **Step 3: The gate passes.** Numbers recorded below; continuing to Task 6.
- [x] **Step 4: Propose the commit** `test: validate the strength model against baselines on real seasons`.

**Chosen values and results**, from a two-round grid search on 810 held-out Premier League matches (2024/25 onward, 240 distinct dates):

- Round 1 (`half_life_days` ∈ {14, 30, 60, 120, unlimited}, `penalty` ∈ {0.01, 0.1, 1.0, 5.0}): best was (120, 5.0) at xG RMSE 0.8253 — but penalty kept improving all the way to 5.0, its upper boundary, so the search wasn't actually done.
- Round 2 (`half_life_days` ∈ {120, 180, 250}, `penalty` ∈ {5.0, 10.0, 20.0, 50.0}): found a genuine interior optimum on both axes this time — penalty peaked at 10 at every half-life tried (worse on either side), and 180 days beat both 120 and 250. Winner: **`half_life_days=180`, `penalty=10.0`**, xG RMSE 0.8224 (a marginal improvement over round 1 — not worth a third round).

| | Model | Baseline | |
|---|---|---|---|
| xG accuracy | 0.8224 | `league_average`: 0.9113 | PASS |
| Outcome vs Elo (expected-score RMSE) | 0.3924 | `elo_ratings`: 0.3988 | PASS |
| Points vs last-five (expected-points RMSE) | 1.1977 | `last_five_results`: 1.3778 | PASS |

---

### Task 6: Form, checked against your examples

**Files:**
- Create: `backend/analysis/form.py`
- Test: `backend/tests/test_analysis_form.py`, plus real-data checks

**Definition:** `form(team, date) = sum over the team's last five league matches before date of (points won − expected points)`, where expected points come from the attack/defence model's predicted xG for that match, run through `validation.match_outcome_probabilities` (the same Poisson conversion the gate uses).

- [x] **Step 1: Write fixture tests:** a win against a strong opponent scores above one against a weak opponent with the same result, and a shutout draw at home to a weak side scores negative.
- [x] **Step 2: Write real-data checks against the examples we agreed** — only Fulham 0-0 (12 Sep 2026) turned out to be checkable: querying the real `matches` table showed Iraola has played just seven matches so far (through 2026-09-20, the data's current edge), none against Manchester City, Coventry, Aston Villa, or Real Madrid. Those were hypothetical illustrations from the planning conversation ("if we beat City 1-0..."), not matches that have happened yet. Real Madrid is a Champions League opponent regardless and could never be part of `form()` - it only uses league matches, per the plan's own global constraints. Confirmed: the Fulham draw scores negative, matching the agreed intuition. **Revisit the other three examples once those fixtures are actually played and ingested.**
- [x] **Step 3: Implement** `form.py` and confirm.
- [x] **Step 4: Propose the commit** `feat: opponent-adjusted form from the strength model`.

Along the way: `fit()` crashed with a confusing low-level numpy error on an empty history (hit when computing form for a match whose own training window has nothing before it) instead of failing clearly - fixed to raise `ValueError`, matching `league_average`'s existing precedent.

---

### Task 7: Match Diagnosis

**Files:**
- Create: `backend/analysis/diagnosis.py`
- Test: `backend/tests/test_analysis_diagnosis.py`

- [x] **Step 1: Test the rolling era baseline:** the last 10 same-era matches, nothing when fewer than 5 exist.
- [x] **Step 2: Test z-scores** for PPDA, xG and deep completions against that baseline, with the largest deviations ranked first.
- [x] **Step 3: Implement and confirm**, then propose the commit `feat: match diagnosis against the rolling era baseline`.

A real-data sanity check caught a genuine bug before it shipped: cup and European rows have no Understat stats (NaN for xg/ppda/deep - Understat covers the Premier League only), and the "last 10" window was pulling them in raw. pandas' mean/std silently skip NaN, so a window padded with cup matches could rest on far fewer real data points than its row count - and the "minimum 5" count checked raw rows, not valid ones, so it could wrongly pass. Fixed: `era_baseline` now filters to rows with real stats before windowing. Confirmed against real data too: Iraola's Bournemouth match (20 Sep 2026) correctly shows no baseline yet - only 4 real Premier League matches exist before it once the two cup/European rows are excluded.

---

### Task 8: Era comparison

**Files:**
- Create: `backend/analysis/era.py`
- Test: `backend/tests/test_analysis_era.py`

- [x] **Step 1: Test the era means, bootstrap confidence intervals and the permutation test** on a fixture where two eras clearly differ and where they do not.
- [x] **Step 2: Implement and confirm**, then propose the commit `feat: era comparison with confidence intervals and a permutation test`.

Applied Task 7's lesson up front this time: `era_means` drops NaN (cup/European) rows before computing anything, with a test for it, rather than waiting to rediscover the same bug. Real-data sanity check (Klopp vs Slot): possession is a real, significant difference (61.1% vs 57.8%, p=0.0065); PPDA and xG are not (p=0.34, p=0.74) - a believable result, not a red flag.

---

### Task 9: Raw on/off

**Files:**
- Create: `backend/analysis/onoff.py`
- Test: `backend/tests/test_analysis_onoff.py`

- [x] **Step 1: Test that the raw split counts matches with and without a player**, and that the sample size is reported alongside every difference.
- [x] **Step 2: Test that results below the adaptive threshold are flagged as greyed out**, not removed.
- [x] **Step 3: Implement and confirm**, then propose the commit `feat: raw on/off splits with sample sizes`.

Note for later readers: `rosters` holds both teams' players per match, not just Liverpool's, so `raw_onoff` identifies Liverpool's own rows by matching `matches`' `venue` (Home/Away) to the roster's `side` (h/a) - a non-obvious join, tested explicitly (including that a same-numbered opponent-side player is never mistaken for Liverpool's own). Cup/European matches have no `understat_match_id` and so no roster data; they're excluded from both groups rather than guessed into "without the player." Real-data check (Salah, xG): 2.17 with him on the pitch (315 matches) vs 1.77 without (32) - a real attacking boost, while xGA barely moves (1.12 vs 1.16), as expected for an attacking player.

---

### Task 10: The score-state control

**Files:**
- Create: `backend/analysis/lineup.py` (score-state section)
- Test: `backend/tests/test_analysis_lineup.py`

**Definition:** from the shots table's goals (minute, side), rebuild the score at every minute of every match. For each player who came on, record the score when he entered, so the model can separate the player from the game state he came on in.

- [x] **Step 1: Test the score reconstruction** on a fixture with goals at known minutes.
- [x] **Step 2: Test the score state at a substitution minute** for a player who came on while his team was leading.
- [x] **Step 3: Implement and confirm**, then propose the commit `feat: score-state reconstruction for lineup controls`.

Checking real data before trusting the naive approach caught a genuine subtlety: `parse_match_rosters`'s own docstring (Plan 1, already merged) claims "the minute of the swap is the outgoing player's minutes" - true only when the outgoing player was an original starter. For a substitute who is later subbed off himself, his `minutes` is how long *he* was on the pitch, not his clock-exit position. Verified on real match 7432: Lallana replaced Wijnaldum at minute 67, played 5 minutes, then was replaced by Lovren - Lovren's real entry minute is 72 (67+5), not 5. 14 such substitution chains exist in the real data, so this wasn't a rare edge case. `entry_minute` walks `roster_out` back to a starter instead of trusting one hop; the misleading comment in `understat.py` is corrected to point at it.

---

### Task 11: Adjusted lineup effects (ridge)

**Files:**
- Modify: `backend/analysis/lineup.py`
- Test: `backend/tests/test_analysis_lineup.py`

**Model:** one row per Liverpool match. Predictors: a 0/1 flag for each player who appeared, plus the opponent's strength from the gated model, home or away, rest days, era, and the score-state control. Target: xG difference. Ridge penalty chosen by time-aware cross-validation. Intervals come from a bootstrap.

**Outputs:** per-player effects with intervals, and "with and without" counterfactuals: the predicted difference for a match with a player added to the lineup, which equals his estimated effect.

**Validation:** the model must beat a version without player flags on held-out matches. If it doesn't, the player effects are not reported.

- [x] **Step 1: Test on synthetic data** that a planted player effect is recovered in sign and rough size.
- [x] **Step 2: Test that the counterfactual** equals the prediction with the player added minus the prediction without him.
- [x] **Step 3: Test that a player with few minutes gets a wider interval** than a regular starter.
- [x] **Step 4: Implement, confirm, and run the held-out comparison** against the no-player version.
- [x] **Step 5: Propose the commit** `feat: adjusted lineup effects with score-state controls and counterfactuals`.

**Findings:**

- Ridge is a closed-form fit (standardize columns, penalize everything but the intercept, rescale back) rather than a new dependency - `sklearn` isn't installed in this project, and the whole model is a direct enough calculation that writing it out matches `strength.py`'s own style (custom Poisson-deviance optimization, no GLM library either).
- `score_state_control` generalizes `lineup.py`'s existing per-entry reconstruction to a minutes-weighted average goal difference across a whole match, so a team "cruising at 3-0" has a legitimate place to put that variance instead of it landing on whichever substitute was on the pitch.
- Joining a Liverpool match's opponent name (`matches.opponent`, FBref's full names - "Tottenham Hotspur") against the strength model's own ratings (keyed by Understat's short names from `league_matches` - "Tottenham") silently returned 0.0 for ~10 clubs before this was caught: `opponent_net_strength` now routes both sides through the existing `backend/model/names.py` `NameMap` (the same one `build_matches()` already uses), rather than inventing a second, parallel name table.
- Era dummies are one-hot by manager with the **most-common** era dropped as the baseline, not an arbitrary (e.g. alphabetical) one - the first implementation accidentally made the ~5-match Iraola caretaker spell the implicit baseline, which every other era's coefficient would then be measured against despite it being the era ridge has the least evidence about.
- Alpha selection uses expanding-window (time-ordered) cross-validation, the same anti-leakage shape as `validation.held_out_predictions` - never validating a fold on a match using coefficients fitted partly on later ones.
- **The validation gate failed on real data, honestly**: full model (with player flags) RMSE 1.4774 vs. the no-player-flags model RMSE 1.4057, at their respective best cross-validated alphas (3000 and 300 - each confirmed a genuine interior optimum, not a search-boundary artifact, by scanning a much wider alpha grid first, the same check Task 5's grid search needed). The player-flag model does not win, so per the plan's own rule, `player_lineup_effects` returns `effects=None` rather than reporting unproven per-player numbers. This is a believable result, not a bug: one match's xG difference is a very noisy target, and ~79 Liverpool players (many with only a handful of appearances) across 346 Understat-covered matches is a hard identification problem for an additive, linear per-player effect - the same difficulty well-documented for "plus-minus"-style sports models generally. The gate did its job.
- **Follow-up: minutes-weighted presence instead of a flat 0/1 flag.** A late substitute's column is now his minutes played divided by 90 (0.0 never played, ~0.33 a 30-minute cameo, 1.0 a full match) rather than an all-or-nothing flag, so `counterfactual_effect`'s "flag at 1 vs 0" reads as "played the whole match" vs. "didn't play" instead of "was in the squad" vs. "wasn't." This is strictly a more honest representation either way, but it did **not** flip the result: true optimum RMSE 1.4907 (alpha 100000, also re-confirmed against a search-boundary artifact - this one took two rounds of widening before settling) vs. the same 1.4057 reduced-model baseline, actually a slightly wider loss than the 0/1 version's 0.0717. Kept the minutes-weighted version anyway since it's the more defensible design on its own merits, independent of this result - but it's further evidence the ceiling here is sample size and per-match noise, not the specific feature encoding.

---

### Task 12: Scouting

**Files:**
- Create: `backend/analysis/scouting.py`
- Test: `backend/tests/test_analysis_scouting.py`

**Scope:** all 20 clubs, from the player-season table, filtered by minutes and position.

- [ ] **Step 1: Test percentile ranks** on a fixture with known ordering.
- [ ] **Step 2: Test cosine similarity** on standardised per-90 profiles, and that a player is not returned as similar to himself.
- [ ] **Step 3: Implement and confirm**, then propose the commit `feat: scouting by percentile and similarity across all 20 clubs`.

---

### Task 13: Close-out

- [ ] **Step 1: Integration tests** for each view's output on the real database.
- [ ] **Step 2: Comprehension checkpoint** with the user, on the gate and the score-state control in particular.
- [ ] **Step 3: Pull request** from `feat/analysis` into `main`, merged with a merge commit, as with Plan 1.

---

## Decisions still open

- Half-life and penalty for the strength model are chosen by Task 5, not fixed here.
- Scouting uses the same adaptive threshold as the rest of the analysis.
