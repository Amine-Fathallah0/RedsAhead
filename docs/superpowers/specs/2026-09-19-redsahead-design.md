# RedsAhead — Design Spec

*One step ahead, by the numbers.*

Unofficial fan project. Not affiliated with Liverpool FC. No club crest or official branding in the UI.

## 1. Goal

An interactive dashboard that combines player analysis, tactical analysis and scouting for Liverpool FC, and relates players and tactics to match outcomes. Example question it should help answer: *"How did an opponent break Liverpool's press in this match, and which players/lineups coincide with that weakness?"* From those weaknesses it suggests scouting targets.

Managerial eras (Klopp, Slot, Iraola) are a first-class dimension so tactical styles can be compared directly.

## 2. Constraints

- **Free data only:** FBref (aggregated team/player stats incl. pressures) and Understat (shot-level xG with minute and location). No event/tracking data.
- **Consequence:** the dashboard cannot attribute a specific mistimed press to a specific player at a specific moment. All player-to-outcome links are statistical correlations and are labelled as such.
- **Coverage:** from 2017/18 (start of FBref advanced stats) through the current season. Era boundaries live in `config/eras.yaml` and can be corrected without code changes.
- **Stack:** Python backend (FastAPI, DuckDB, pandas), React frontend (Plotly for charts).
- **Data availability is unverified.** FBref's data provider may have changed. The first task is to confirm what is actually obtainable; views that depend on missing metrics are cut or reworked.

## 3. Architecture

```
Sources -> ingest (cached) -> model (clean schema, DuckDB) -> analysis (pure functions) -> api (FastAPI, JSON) -> web (React)
```

| Unit | Job | Depends on |
|---|---|---|
| `backend/ingest/` | Download and cache raw FBref/Understat data to disk; rate-limited; no parsing beyond storage | network, cache dir |
| `backend/model/` | Clean raw files into a fixed schema (`matches`, `team_match_stats`, `player_match_stats`, `shots`, `lineups`); tag each match with manager era from `eras.yaml` | raw cache, config |
| `backend/analysis/` | Pure functions over model tables: `era.py`, `diagnosis.py`, `onoff.py`, `scouting.py`. No web or network code | DuckDB tables |
| `backend/api/` | Thin FastAPI layer exposing analysis outputs as validated JSON; no logic | analysis |
| `web/` | React dashboard, four views | api |

Rules: analysis never touches the network; every panel making a player-outcome claim shows sample size and a "correlation, not causation" label.

## 4. Version 1 scope (Tier 1)

1. **Era Comparison** — Klopp vs Slot vs Iraola on pressing intensity (pressures by third, PPDA-style ratio), possession, shot quality for/against, progressive passing. Means with confidence intervals and significance tests.
2. **Match Diagnosis** — for a chosen match, compare metrics to Liverpool's rolling baseline for that era; show largest deviations, an xG timeline by minute, and a shot map. Flags where the structure likely broke, as a proxy, not a proven cause.
3. **Player Profiles and On/Off** — percentile radars; team metrics with and without each player, with sample sizes. Raw on/off is confounded by opponent strength and teammates; the UI says so.
4. **Scouting (simple)** — rank Liverpool's weakest metrics per era; for a chosen weakness, rank players at other clubs by percentile on related stats and cosine similarity, filtered by minutes and position. No fitted model.

Small-sample handling: thresholds (min matches / minutes) are config values in `config/settings.yaml`; below-threshold results are greyed out with a warning.

## 5. Explicitly out of scope for v1

- Adjusted lineup effects (regularised regression), opponent-style clustering, outcome models with SHAP (Tier 2, planned for v2; schema and API leave room but nothing is built).
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

1. Data-availability check (what FBref/Understat actually serve, 2017/18 onward); adjust views accordingly.
2. Ingest + model (cached scrapers -> clean DuckDB schema, era tagging).
3. Analysis: `era.py`, `diagnosis.py`, `onoff.py`, `scouting.py`, each unit-tested.
4. API routes.
5. Frontend: Era Comparison, Match Diagnosis, Player Profiles/On-Off, Scouting.
6. Polish: correlation-not-causation labels, sample-size warnings, README (including the unofficial-fan-project note).

## 10. Naming

Display name `RedsAhead`; package/repo/folder identifiers `redsahead`. The name appears only in config and the UI title, so renaming is cheap.
