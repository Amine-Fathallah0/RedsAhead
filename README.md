# RedsAhead

*One step ahead, by the numbers.*

[![Tests](https://github.com/Amine-Fathallah0/RedsAhead/actions/workflows/tests.yml/badge.svg)](https://github.com/Amine-Fathallah0/RedsAhead/actions/workflows/tests.yml)

An interactive dashboard combining player analysis, tactical analysis and scouting for Liverpool FC — built on free data, with the goal of relating players and tactics to match outcomes (e.g. *"which lineup, and which tactical shift, coincided with this result?"*).

**Unofficial fan project.** Not affiliated with, endorsed by, or connected to Liverpool Football Club in any way. Personal, non-commercial use only.

## Status

This repo currently holds the **data foundation**: ingestion, cleaning, and a DuckDB database joining Understat and FBref data across multiple seasons, all test-covered. The analysis layer, API and frontend dashboard described in the [design spec](docs/superpowers/specs/2026-09-19-redsahead-design.md) are not built yet — see the [implementation plan](docs/superpowers/plans/2026-09-20-redsahead-data-foundation.md) for what this covers and what comes next.

## Data sources and credit

- **[Understat](https://understat.com)** — the analytical backbone: match-level xG, PPDA (pressing intensity), deep completions, shots, and full match lineups with minutes played. Covers Premier League matches only. Fetched politely (cached on disk, one request every 3 seconds) from Understat's own data endpoints.
- **[FBref](https://fbref.com)** — context: fixtures, results, possession, and formations, across every competition (league, cups, Champions League). FBref sits behind a bot-detection check, so its data is exported by hand rather than scraped (see below).

Neither source publishes a formal API or licence for this data. Usage here follows the norms both sites are built around: personal and non-commercial, with attribution, and without republishing raw data — see [the spec's data-handling section](docs/superpowers/specs/2026-09-19-redsahead-design.md#10-data-licensing-and-handling) for the full reasoning. Raw data is never committed to this repo (see `.gitignore`); the dashboard, once built, will show derived analysis rather than raw tables.

### A note on how this was built

An early version of the ingestion code assumed Understat embedded its data as JSON inside each page's HTML — true when this project started, but Understat has since moved that data behind separate JSON endpoints (`getLeagueData`, `getMatchData`). The old approach and the discovery that replaced it are preserved in this repo's git history rather than kept in the working tree; the [spec's revision note](docs/superpowers/specs/2026-09-19-redsahead-design.md) has the short version, and the commit log has the rest.

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -e ".[dev]"
```

## Loading FBref data

FBref's bot check means its data has to be exported by hand, once per season:

1. Open Liverpool's "Scores & Fixtures" page for a season on FBref, e.g.
   `https://fbref.com/en/squads/822bd0ba/2025-2026/all_comps/Liverpool-Stats-All-Competitions`
2. On the **Scores & Fixtures** table, choose **Share & Export → Get table as CSV**. This reveals the CSV as plain text on the page — it isn't a file download.
3. Select the text starting at the real header row (`Date,Time,Comp,...`) — **don't** include FBref's citation notice above it — and copy it.
4. Save it as `backend/ingest/manual/fbref_<season>_fixtures.csv`, e.g. `fbref_2025-2026_fixtures.csv`. Create the `manual/` folder if it doesn't exist yet.

Repeat for each season you want (2017-18 through the current season, per the spec). This folder is gitignored — nothing here is ever committed.

## Building the database

```bash
python -m backend.model.build
```

This fetches each Understat season and match (cached afterwards, so reruns are fast), reads every `fbref_*_fixtures.csv` you've exported, joins the two sources, tags every match with its manager era, and writes `data/redsahead.duckdb` with four tables: `matches`, `shots`, `player_season_stats`, and `rosters`. Any unmatched club name or fixture is printed as a problem rather than silently dropped — add it to `config/name_map.yaml` and rerun.

## Running the tests

```bash
pytest                 # unit tests: offline, run in CI on every push
pytest -m integration  # also checks the real Understat site (needs internet)
```

## Project layout

```
config/           eras.yaml, settings.yaml, name_map.yaml
backend/
  ingest/         fbref.py, understat.py, polite_fetcher.py
                  cache/, manual/   (gitignored — raw data lives here only)
  model/          eras.py, names.py, matches.py, build.py
  tests/          unit tests + integration tests (pytest -m integration)
data/             redsahead.duckdb  (gitignored)
docs/superpowers/ design spec and implementation plans
.github/workflows/ CI (runs the unit test suite on every push and pull request)
```
