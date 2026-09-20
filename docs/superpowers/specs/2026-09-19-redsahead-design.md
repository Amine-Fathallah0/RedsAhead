# RedsAhead — Design Spec

*One step ahead, by the numbers.*

Unofficial fan project. Not affiliated with Liverpool FC. No club crest or official branding in the UI.

## 1. Goal

An interactive dashboard that combines player analysis, tactical analysis and scouting for Liverpool FC, and relates players and tactics to match outcomes. Example question it should help answer: *"How did an opponent break Liverpool's press in this match, and which players/lineups coincide with that weakness?"* From those weaknesses it suggests scouting targets.

Managerial eras (Klopp, Slot, Iraola) are a first-class dimension so tactical styles can be compared directly.

## 2. Constraints

- **Free data only, two sources with different roles** (verified 2026-09-19 by inspection):
  - **Understat = analytical backbone.** Per-match team data: xG, xGA, npxG, npxGA, xPTS, PPDA (for and against, raw attempts/defensive actions), deep completions (for/against). Player data: xG, xA, key passes, xGChain, xGBuildup. Shot-level data is confirmed (seen on a player shot map and tooltip): pitch location, minute, situation, assister, shot type, result, xG, plus match and date, per player and season. Seasons 2014/15 to 2026/27. Data is embedded in page JSON; no bot check observed. Terms of use not yet checked.
  - **FBref = context layer.** Basic stats only: lineups, starts, minutes, positions, possession, own and opponent formation, result, and basic player actions (shots, SoT, fouls, fouled, offsides, crosses, tackles won, interceptions, cards). The advanced Opta-based tables (pressures, xG, progressive passes, Scouting Report) are **absent for 2025-26 and 2026-27**; older seasons are unchecked and are not relied on. FBref sits behind a Cloudflare bot check, so automated scraping is not assumed to work.
- **Understat covers domestic league matches only (Premier League).** xG, PPDA, deep completions and shots exist only for league matches; European and cup matches carry FBref basics only (result, possession, formations). Era Comparison and Match Diagnosis therefore run on league matches.
- **Consequence:** the dashboard cannot attribute a specific mistimed press to a specific player at a specific moment. Pressing is measured at team level (PPDA, deep completions). All player-to-outcome links are statistical correlations and are labelled as such.
- **Coverage:** Understat from 2017/18 through the current season (aligned with FBref lineups where available). Era boundaries live in `config/eras.yaml` and can be corrected without code changes.
- **Stack:** Python backend (FastAPI, DuckDB, pandas), React frontend (Plotly for charts).
- **Open verification items:** Understat terms of use; whether FBref lineup/formation data exists for every season between 2017/18 and now (confirmed present for 2017/18 and 2026/27: possession, own and opponent formation); whether FBref data can be obtained (manual CSV export vs. scraping).

## 3. Architecture

```
Sources -> ingest (cached) -> model (clean schema, DuckDB) -> analysis (pure functions) -> api (FastAPI, JSON) -> web (React)
```

| Unit | Job | Depends on |
|---|---|---|
| `backend/ingest/` | Understat: fetch and cache embedded JSON (rate-limited). FBref: read manually exported CSVs dropped in `backend/ingest/manual/` (a scraper only if terms and bot check allow). No parsing beyond storage | network, cache dir, manual CSVs |
| `backend/model/` | Clean raw files into a fixed schema (`matches`, `team_match_stats`, `player_match_stats`, `shots`, `lineups`); tag each match with manager era from `eras.yaml`. Joins the two sources: matches are keyed on date + home/away teams; players and teams are matched through `config/name_map.yaml` (accents, name variants, club naming). Unmatched names are reported, never silently dropped. Positions are grouped into one consistent scheme, goalkeepers handled separately from outfielders | raw cache, manual CSVs, config |
| `backend/analysis/` | Pure functions over model tables: `era.py`, `diagnosis.py`, `onoff.py`, `scouting.py`. No web or network code | DuckDB tables |
| `backend/api/` | Thin FastAPI layer exposing analysis outputs as validated JSON; no logic | analysis |
| `web/` | React dashboard, four views | api |

Rules: analysis never touches the network; every panel making a player-outcome claim shows sample size and a "correlation, not causation" label.

## 4. Version 1 scope (Tier 1)

1. **Era Comparison** — Klopp vs Slot vs Iraola on pressing intensity (PPDA, deep completions allowed), possession, xG/xGA and npxG for/against, and formation usage. Includes opponent-formation vs outcome (which opposing shapes hurt Liverpool under each manager). Means with confidence intervals and significance tests.
2. **Match Diagnosis** — for a chosen match, compare PPDA, xG and deep completions to Liverpool's rolling baseline for that era; show largest deviations, opponent formation, lineup and minutes, an xG timeline by minute and a shot map with shooter, assister, situation and shot type. Flags matches where the press dropped off or the structure likely broke, as a proxy, not a proven cause.
3. **Player Profiles and Player Impact** — percentile radars from Understat (xG, xA, key passes, xGChain, xGBuildup) plus FBref basic actions. Player impact has two layers, shown side by side:
   - **Raw on/off:** team metrics (goals, xG, PPDA) with and without each player, using FBref lineups/minutes and Understat match data, with sample sizes. Confounded by opponent strength and teammates; the UI says so.
   - **Adjusted lineup effects (the v1 ML component):** ridge-regression adjusted plus-minus. Each match (or lineup stint) is a row; predictors are player-minutes indicators plus opponent strength, home/away and era; the target is xG difference (goal difference as a check). The penalty is chosen by time-aware cross-validation. Outputs a per-player effect with uncertainty (e.g. bootstrap intervals). Validated against a no-player baseline on held-out matches and by stability across resamples; players with too few minutes are shrunk to zero or greyed out. The UI reports it as an association given the data, not a causal effect.
4. **Scouting (simple)** — rank Liverpool's weakest metrics per era; for a chosen weakness, rank players at other clubs by percentile on related stats and cosine similarity, filtered by minutes and position. No fitted model. **Limitation:** other clubs' players only have Understat attacking/build-up metrics and FBref basic actions, so defensive and pressing scouting is thin and the UI says so.

Small-sample handling: thresholds (min matches / minutes) are config values in `config/settings.yaml`; below-threshold results are greyed out with a warning.

## 5. Explicitly out of scope for v1

- Role-aware player embeddings for scouting (replacing percentile ranking), opponent-style clustering, and outcome models with SHAP (planned for v2; schema and API leave room but nothing is built). Adjusted lineup effects were promoted into v1 (see Player Impact above).
- Press-break detection from shot sequences and changepoint analysis (Tier 3).
- Any paid/richer data source and any plugin/adapter system for sources. Ingestion targets FBref and Understat directly.

## 6. Error handling

- **Scraping:** rate-limited and cached, so a failure never re-fetches what succeeded. Runs report which seasons/tables failed rather than continuing silently.
- **Missing data:** panels show "not available for this period"; no interpolation. Cleaning records what it dropped.
- **Small samples:** enforced in analysis via config thresholds, so the UI cannot display an unguarded number.
- **API:** validated response models; clear 4xx errors for unknown matches/players.

## 7. Testing

- Analysis: unit tests on small fixture files, no network.
- API: contract tests against fixture data.
- Frontend: one smoke test per view.
- Scrapers: light tests against saved HTML samples to catch site layout changes.

## 8. Project layout

```
Liverpoool/
  config/         eras.yaml, settings.yaml
  backend/
    ingest/       fbref.py, understat.py, cache/
    model/
    analysis/     era.py, diagnosis.py, onoff.py, scouting.py
    api/
    tests/        fixtures/
  web/
  docs/superpowers/specs/
```

## 9. Build order

1. Finish the data-availability check: Understat shot-level data and terms of use, FBref lineup/formation coverage back to 2017/18, and how FBref data will be obtained (manual CSV export vs. scraper). Adjust views accordingly.
2. Ingest + model (cached scrapers -> clean DuckDB schema, era tagging).
3. Analysis: `era.py`, `diagnosis.py`, `onoff.py`, `scouting.py`, each unit-tested.
4. API routes.
5. Frontend: Era Comparison, Match Diagnosis, Player Profiles/On-Off, Scouting.
6. Polish: correlation-not-causation labels, sample-size warnings, README (including the unofficial-fan-project note).

## 10. Data licensing and handling

Source terms are informal or restrictive (per a third-party summary, not the sources' own terms): Understat has no formal licence and expects attribution and polite, non-commercial use; FBref data is Opta's, its terms restrict systematic scraping, and raw numbers must not be republished.

- Never commit raw data: `backend/ingest/cache/`, `backend/ingest/manual/` and the DuckDB file are in `.gitignore`. The repo holds code only, so it can go public later without leaking data.
- The UI shows derived analysis (charts, percentiles, model outputs), not raw FBref tables. Understat and FBref are credited in the UI footer and README.
- Fetching is polite: cache every response, pull each season once, at most one request per three seconds, never fetch on dashboard load.
- Personal, non-commercial use only.

## 11. Naming

Display name `RedsAhead`; package/repo/folder identifiers `redsahead`. The name appears only in config and the UI title, so renaming is cheap.
