# RedsAhead Data Foundation Implementation Plan (Plan 1 of 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the ingest and model layers that turn Understat pages and manually exported FBref CSVs into a clean DuckDB database (`matches`, `shots`, `player_season_stats`) with each match tagged by manager era.

**Architecture:** `ingest/` fetches and caches Understat pages politely and reads FBref CSVs that the user exports by hand. `model/` parses the raw payloads into pandas tables, joins the two sources (matches keyed on date + opponent, teams matched through `config/name_map.yaml`), tags each match with a manager era, and writes DuckDB tables. Nothing in this plan touches analysis, the API or the frontend. All unit tests run offline on small synthetic fixtures.

**Tech Stack:** Python 3.11+ (the user's venv is `.Reds`, Python 3.13), pandas, DuckDB, PyYAML, requests, pytest.

**Spec:** `docs/superpowers/specs/2026-09-19-redsahead-design.md`

**Roadmap:** Plan 1 (this) data foundation. Plan 2: analysis (era comparison, match diagnosis, on/off, ridge adjusted lineup effects, scouting). Plan 3: FastAPI layer. Plan 4: React frontend. Plan 2 starts with a lineup-source decision (Task 4 probes whether Understat's `rostersData` provides lineups and minutes; otherwise FBref match logs are needed).

## Global Constraints

- Free data only. Understat is the analytical backbone; FBref is the context layer and comes from **manual CSV exports** (FBref sits behind a Cloudflare bot check; never automate it).
- Never commit raw data: `backend/ingest/cache/`, `backend/ingest/manual/` and the DuckDB file are in `.gitignore`. Test fixtures are small **synthetic** payloads written by hand, never captured real data.
- Polite fetching: cache every response, pull each season once, at most one request per three seconds, never fetch on dashboard load.
- Analysis never touches the network; tests run offline on fixtures.
- Unmatched names and rows are reported, never silently dropped.
- Era boundaries live in `config/eras.yaml`; thresholds (min matches / minutes) live in `config/settings.yaml`.
- Coverage: 2017/18 through the current season. Understat covers **Premier League matches only**, so xG/PPDA exist only for league matches; European and cup matches carry FBref basics only.
- Personal, non-commercial use. Credit Understat and FBref in the README (and later the UI footer).
- Python 3.11+. Commands below use the user's venv: `.Reds/Scripts/python` (Windows Git Bash).

---

## File Structure

```
pyproject.toml
.gitignore
README.md
config/
  settings.yaml        # intervals, paths, small-sample thresholds
  eras.yaml            # manager eras
  name_map.yaml        # club name aliases between sources
backend/
  __init__.py
  config.py            # Settings + load_settings
  ingest/
    __init__.py
    understat.py       # embedded-JSON extraction, PoliteFetcher, URL helpers
    fbref.py           # fixtures CSV loader (manual exports)
  model/
    __init__.py
    eras.py            # Era, load_eras, era_for
    names.py           # normalise, NameMap, load_name_map
    understat_tables.py# league/match page parsers -> DataFrames
    matches.py         # build_matches: join FBref + Understat, tag eras, report problems
    build.py           # build_database + CLI entry point
  tests/
    __init__.py
    fixtures.py        # synthetic Understat league page
    fixtures_match.py  # synthetic Understat match pages
    fixtures_fbref.py  # synthetic FBref fixtures CSV
    fixtures_model.py  # shared NAME_MAP / ERAS for model tests
    test_config.py  test_eras.py  test_understat_fetch.py
    test_understat_tables.py  test_understat_shots.py  test_fbref.py
    test_names.py  test_matches.py  test_build.py
```

---

### Task 1: Project scaffolding, settings and eras

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `config/settings.yaml`, `config/eras.yaml`
- Create: `backend/__init__.py`, `backend/model/__init__.py`, `backend/ingest/__init__.py`, `backend/tests/__init__.py` (all empty)
- Create: `backend/config.py`, `backend/model/eras.py`
- Test: `backend/tests/test_config.py`, `backend/tests/test_eras.py`

**Interfaces:**
- Produces: `backend.config.ROOT: Path`, `Settings` (fields `min_request_interval_seconds: float`, `cache_dir: Path`, `manual_dir: Path`, `db_path: Path`, `min_matches: int`, `min_minutes: int`), `load_settings(path: Path | None = None) -> Settings`.
- Produces: `backend.model.eras.Era(manager: str, start: date, end: date | None)`, `load_eras(path) -> list[Era]`, `era_for(d: date, eras: list[Era]) -> str | None`.

- [ ] **Step 1: Create the packaging and ignore files**

<!-- file: pyproject.toml -->
```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "redsahead"
version = "0.1.0"
description = "RedsAhead: one step ahead, by the numbers."
requires-python = ">=3.11"
dependencies = [
  "pandas>=2.1",
  "duckdb>=1.0",
  "pyyaml>=6.0",
  "requests>=2.31",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[tool.setuptools.packages.find]
include = ["backend*"]

[tool.pytest.ini_options]
testpaths = ["backend/tests"]
```

<!-- file: .gitignore -->
```gitignore
# environments and caches
.Reds/
.venv/
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/

# raw data must never be committed (spec section 10)
data/
backend/ingest/cache/
backend/ingest/manual/

# frontend (Plan 4)
node_modules/
web/dist/
```

Create the four empty `__init__.py` files listed above. Then install:

```bash
.Reds/Scripts/python -m pip install -e ".[dev]"
```
Expected: `Successfully installed ... redsahead-0.1.0`.

- [ ] **Step 2: Write the failing tests**

<!-- file: backend/tests/test_config.py -->
```python
from backend.config import ROOT, load_settings


def test_load_settings_resolves_paths_against_repo_root(tmp_path):
    path = tmp_path / "settings.yaml"
    path.write_text(
        "min_request_interval_seconds: 3\n"
        "cache_dir: backend/ingest/cache\n"
        "manual_dir: backend/ingest/manual\n"
        "db_path: data/redsahead.duckdb\n"
        "min_matches: 10\n"
        "min_minutes: 900\n",
        encoding="utf-8",
    )

    settings = load_settings(path)

    assert settings.min_request_interval_seconds == 3.0
    assert settings.cache_dir == ROOT / "backend" / "ingest" / "cache"
    assert settings.db_path == ROOT / "data" / "redsahead.duckdb"
    assert settings.min_matches == 10
    assert settings.min_minutes == 900
```

<!-- file: backend/tests/test_eras.py -->
```python
from datetime import date

import pytest

from backend.model.eras import era_for, load_eras

GOOD = """
eras:
  - manager: Klopp
    start: 2015-10-08
    end: 2024-05-31
  - manager: Slot
    start: 2024-06-01
    end: null
"""


def write(tmp_path, body):
    path = tmp_path / "eras.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_era_for_respects_inclusive_boundaries_and_open_end(tmp_path):
    eras = load_eras(write(tmp_path, GOOD))

    assert era_for(date(2024, 5, 31), eras) == "Klopp"
    assert era_for(date(2024, 6, 1), eras) == "Slot"
    assert era_for(date(2031, 1, 1), eras) == "Slot"
    assert era_for(date(2010, 1, 1), eras) is None


def test_overlapping_eras_are_rejected(tmp_path):
    body = """
eras:
  - manager: A
    start: 2020-01-01
    end: 2020-12-31
  - manager: B
    start: 2020-06-01
    end: null
"""
    with pytest.raises(ValueError, match="overlap"):
        load_eras(write(tmp_path, body))


def test_end_before_start_is_rejected(tmp_path):
    body = """
eras:
  - manager: A
    start: 2020-06-01
    end: 2020-01-01
"""
    with pytest.raises(ValueError, match="before start"):
        load_eras(write(tmp_path, body))
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.Reds/Scripts/python -m pytest backend/tests/test_config.py backend/tests/test_eras.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.config'` (and `backend.model.eras`).

- [ ] **Step 4: Write the implementation**

<!-- file: backend/config.py -->
```python
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    min_request_interval_seconds: float
    cache_dir: Path
    manual_dir: Path
    db_path: Path
    min_matches: int
    min_minutes: int


def load_settings(path: Path | None = None) -> Settings:
    path = path or ROOT / "config" / "settings.yaml"
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Settings(
        min_request_interval_seconds=float(raw["min_request_interval_seconds"]),
        cache_dir=ROOT / raw["cache_dir"],
        manual_dir=ROOT / raw["manual_dir"],
        db_path=ROOT / raw["db_path"],
        min_matches=int(raw["min_matches"]),
        min_minutes=int(raw["min_minutes"]),
    )
```

<!-- file: backend/model/eras.py -->
```python
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Era:
    manager: str
    start: date
    end: date | None


def load_eras(path: Path) -> list[Era]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    eras = [
        Era(
            manager=item["manager"],
            start=date.fromisoformat(str(item["start"])),
            end=None if item.get("end") is None else date.fromisoformat(str(item["end"])),
        )
        for item in raw["eras"]
    ]
    eras.sort(key=lambda era: era.start)
    for era in eras:
        if era.end is not None and era.end < era.start:
            raise ValueError(f"{era.manager}: end before start")
    for prev, nxt in zip(eras, eras[1:]):
        if prev.end is None or prev.end >= nxt.start:
            raise ValueError(f"eras overlap: {prev.manager} and {nxt.manager}")
    return eras


def era_for(d: date, eras: list[Era]) -> str | None:
    for era in eras:
        if era.start <= d and (era.end is None or d <= era.end):
            return era.manager
    return None
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.Reds/Scripts/python -m pytest backend/tests/test_config.py backend/tests/test_eras.py -v`
Expected: 4 passed.

- [ ] **Step 6: Write the real config files**

<!-- file: config/settings.yaml -->
```yaml
# Seconds between network requests to any source (spec section 10).
min_request_interval_seconds: 3
cache_dir: backend/ingest/cache
manual_dir: backend/ingest/manual
db_path: data/redsahead.duckdb
# Small-sample thresholds used by analysis in Plan 2.
min_matches: 10
min_minutes: 900
```

<!-- file: config/eras.yaml -->
```yaml
# Manager eras. Boundaries are inclusive; a null end means "still in charge".
# Dates are chosen so no match falls in a gap. Confirm them before trusting era comparisons.
eras:
  - manager: Klopp
    start: 2015-10-08
    end: 2024-05-31
  - manager: Slot
    start: 2024-06-01
    end: null
```

- [ ] **Step 7: Ask the user for the Iraola start date, then update `config/eras.yaml`**

Ask the user (in chat): "On what date did Iraola take charge?" Do not guess. Set Slot's `end` to the day before that date and add:

```yaml
  - manager: Iraola
    start: <date the user gives, YYYY-MM-DD>
    end: null
```
Also ask the user to confirm the Klopp and Slot dates above. Verify the file loads:

```bash
.Reds/Scripts/python -c "from backend.model.eras import load_eras; from pathlib import Path; print(load_eras(Path('config/eras.yaml')))"
```
Expected: three `Era(...)` entries in date order, no error.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml .gitignore config backend
git commit -m "feat: project scaffolding, settings and manager eras"
```

---

### Task 2: Understat page extraction and polite fetcher

**Files:**
- Create: `backend/ingest/understat.py`
- Test: `backend/tests/test_understat_fetch.py`

**Interfaces:**
- Produces: `extract_embedded_json(html: str, var_name: str) -> Any` (raises `KeyError` if the variable is absent).
- Produces: `league_url(season_start_year: int) -> str`, `match_url(match_id: str) -> str`.
- Produces: `PoliteFetcher(cache_dir, min_interval_seconds, session=None, sleep=time.sleep, clock=time.monotonic)` with `.get(url: str) -> str`.

- [ ] **Step 1: Write the failing tests**

<!-- file: backend/tests/test_understat_fetch.py -->
```python
import pytest

from backend.ingest.understat import (
    PoliteFetcher,
    extract_embedded_json,
    league_url,
    match_url,
)


def page(var, literal):
    return f"<html><script>var {var} = JSON.parse('{literal}');</script></html>"


def test_extract_decodes_hex_escapes_like_understat_emits():
    html = page("datesData", r"[\x7B\x22title\x22\x3A\x22Liverpool\x22\x7D]")

    assert extract_embedded_json(html, "datesData") == [{"title": "Liverpool"}]


def test_extract_handles_unicode_escapes_and_literal_accents():
    escaped = page("playersData", r"[\x7B\x22n\x22\x3A\x22Ødegaard\x22\x7D]")
    literal = page("playersData", r"[\x7B\x22n\x22\x3A\x22Jérémy\x22\x7D]")

    assert extract_embedded_json(escaped, "playersData") == [{"n": "Ødegaard"}]
    assert extract_embedded_json(literal, "playersData") == [{"n": "Jérémy"}]


def test_extract_raises_key_error_when_variable_missing():
    with pytest.raises(KeyError, match="teamsData"):
        extract_embedded_json("<html></html>", "teamsData")


def test_url_helpers():
    assert league_url(2025) == "https://understat.com/league/EPL/2025"
    assert match_url("1001") == "https://understat.com/match/1001"


class FakeResponse:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


class FakeSession:
    def __init__(self, pages):
        self.pages = pages
        self.calls = []

    def get(self, url, timeout, headers):
        self.calls.append(url)
        return FakeResponse(self.pages[url])


def make_fetcher(tmp_path, session, sleeps):
    return PoliteFetcher(
        tmp_path,
        3.0,
        session=session,
        sleep=sleeps.append,
        clock=lambda: 100.0,
    )


def test_second_fetch_of_same_url_comes_from_cache(tmp_path):
    session = FakeSession({"u1": "<html>one</html>"})
    fetcher = make_fetcher(tmp_path, session, [])

    assert fetcher.get("u1") == "<html>one</html>"
    assert fetcher.get("u1") == "<html>one</html>"
    assert session.calls == ["u1"]


def test_cache_survives_a_new_fetcher_instance(tmp_path):
    make_fetcher(tmp_path, FakeSession({"u1": "x"}), []).get("u1")
    second = FakeSession({})

    assert make_fetcher(tmp_path, second, []).get("u1") == "x"
    assert second.calls == []


def test_uncached_requests_are_spaced_by_the_minimum_interval(tmp_path):
    sleeps = []
    fetcher = make_fetcher(tmp_path, FakeSession({"a": "A", "b": "B"}), sleeps)

    fetcher.get("a")
    assert sleeps == []
    fetcher.get("b")
    assert sleeps == [3.0]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.Reds/Scripts/python -m pytest backend/tests/test_understat_fetch.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.ingest.understat'`.

- [ ] **Step 3: Write the implementation**

<!-- file: backend/ingest/understat.py -->
```python
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import requests

USER_AGENT = "RedsAhead/0.1 (personal research project)"
_JSON_PARSE = r"var\s+{name}\s*=\s*JSON\.parse\('(.*?)'\)"


def league_url(season_start_year: int) -> str:
    return f"https://understat.com/league/EPL/{season_start_year}"


def match_url(match_id: str) -> str:
    return f"https://understat.com/match/{match_id}"


def extract_embedded_json(html: str, var_name: str) -> Any:
    """Return the JSON Understat embeds as `var NAME = JSON.parse('...')`."""
    found = re.search(_JSON_PARSE.format(name=re.escape(var_name)), html, re.DOTALL)
    if found is None:
        raise KeyError(f"{var_name} not found in page")
    # The payload uses \xNN / \uNNNN escapes; the latin-1 round trip keeps any
    # literal non-ASCII characters intact while unicode_escape decodes the rest.
    decoded = found.group(1).encode("latin-1", "backslashreplace").decode("unicode_escape")
    return json.loads(decoded)


class PoliteFetcher:
    """Fetch pages one at a time, cache every response on disk, and wait
    between network requests."""

    def __init__(
        self,
        cache_dir,
        min_interval_seconds,
        session=None,
        sleep=time.sleep,
        clock=time.monotonic,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = min_interval_seconds
        self.session = session or requests.Session()
        self._sleep = sleep
        self._clock = clock
        self._last_request = None

    def get(self, url: str) -> str:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        path = self.cache_dir / f"{digest}.html"
        if path.exists():
            return path.read_text(encoding="utf-8")
        if self._last_request is not None:
            wait = self._last_request + self.min_interval - self._clock()
            if wait > 0:
                self._sleep(wait)
        response = self.session.get(url, timeout=30, headers={"User-Agent": USER_AGENT})
        response.raise_for_status()
        self._last_request = self._clock()
        path.write_text(response.text, encoding="utf-8")
        return response.text
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.Reds/Scripts/python -m pytest backend/tests/test_understat_fetch.py -v`
Expected: 7 passed.

- [ ] **Step 5: Probe one real league page (nothing from it is committed)**

This makes one request, cached under the gitignored cache dir. It confirms the real payloads have the fields the parsers in Task 3 expect.

```bash
.Reds/Scripts/python - <<'EOF'
from backend.config import load_settings
from backend.ingest.understat import PoliteFetcher, extract_embedded_json, league_url

settings = load_settings()
fetcher = PoliteFetcher(settings.cache_dir, settings.min_request_interval_seconds)
html = fetcher.get(league_url(2025))
for name in ("datesData", "teamsData", "playersData"):
    data = extract_embedded_json(html, name)
    first = data[0] if isinstance(data, list) else next(iter(data.values()))
    print(name, type(data).__name__, len(data), sorted(first))
EOF
```
Expected keys: `datesData` items contain `id, isResult, h, a, goals, xG, datetime`; `teamsData` items contain `id, title, history`; `playersData` items contain `id, player_name, games, time, goals, xG, assists, xA, shots, key_passes, position, team_title, npg, npxG, xGChain, xGBuildup`. **If any key differs, stop and fix the Task 3 fixtures and parsers to match before continuing.**

- [ ] **Step 6: Commit**

```bash
git add backend/ingest/understat.py backend/tests/test_understat_fetch.py
git commit -m "feat: understat embedded-json extraction and polite cached fetcher"
```

---

### Task 3: Understat league-page parsers

**Files:**
- Create: `backend/tests/fixtures.py`, `backend/model/understat_tables.py`
- Test: `backend/tests/test_understat_tables.py`

**Interfaces:**
- Consumes: `extract_embedded_json` from Task 2.
- Produces: `parse_league_matches(html) -> DataFrame` with columns `understat_match_id, date, home_team, away_team, home_goals, away_goals, home_xg, away_xg` (played matches only; `date` is a `datetime.date`).
- Produces: `parse_team_match_stats(html) -> DataFrame` with columns `team, date, side, xg, xga, npxg, npxga, ppda, ppda_allowed, deep, deep_allowed, goals, goals_against, xpts` (`ppda` is NaN when the defensive-action count is 0).
- Produces: `parse_player_seasons(html, season_start_year) -> DataFrame` with columns `season, player_id, player, team, position, games, minutes, goals, assists, shots, key_passes, xg, xa, npg, npxg, xg_chain, xg_buildup`.
- Produces test helpers: `understat_page(**variables) -> str`, `league_page() -> str`, constants `DATES`, `TEAMS`, `PLAYERS` in `backend/tests/fixtures.py`.

- [ ] **Step 1: Write the synthetic fixtures and the failing tests**

All numbers below are invented for testing; they are not real match data.

<!-- file: backend/tests/fixtures.py -->
```python
"""Synthetic Understat payloads. Invented numbers, structure mirrors the real pages."""
import json


def understat_page(**variables) -> str:
    scripts = []
    for name, obj in variables.items():
        payload = json.dumps(obj).replace("\\", "\\\\").replace("'", "\\'")
        scripts.append(f"<script>var {name} = JSON.parse('{payload}');</script>")
    return "<html>" + "".join(scripts) + "</html>"


def _team(team_id, title, short):
    return {"id": team_id, "title": title, "short_title": short}


LIVERPOOL = _team("87", "Liverpool", "LIV")
BOURNEMOUTH = _team("73", "Bournemouth", "BOU")
NEWCASTLE = _team("78", "Newcastle United", "NEW")
PALACE = _team("80", "Crystal Palace", "CRY")

DATES = [
    {
        "id": "1001", "isResult": True, "h": LIVERPOOL, "a": BOURNEMOUTH,
        "goals": {"h": "4", "a": "2"}, "xG": {"h": "2.33", "a": "1.57"},
        "datetime": "2025-08-15 19:00:00",
    },
    {
        "id": "1002", "isResult": True, "h": NEWCASTLE, "a": LIVERPOOL,
        "goals": {"h": "2", "a": "3"}, "xG": {"h": "1.10", "a": "2.40"},
        "datetime": "2025-08-25 19:00:00",
    },
    {
        "id": "1003", "isResult": False, "h": LIVERPOOL, "a": PALACE,
        "goals": {"h": None, "a": None}, "xG": {"h": None, "a": None},
        "datetime": "2025-09-27 15:00:00",
    },
]


def _history(date, side, xg, xga, att, deff, att_a, def_a, deep, deep_a, scored, missed, xpts, result):
    return {
        "h_a": side, "xG": xg, "xGA": xga, "npxG": xg, "npxGA": xga,
        "ppda": {"att": att, "def": deff}, "ppda_allowed": {"att": att_a, "def": def_a},
        "deep": deep, "deep_allowed": deep_a, "scored": scored, "missed": missed,
        "xpts": xpts, "result": result, "date": date,
    }


TEAMS = {
    "87": {
        "id": "87", "title": "Liverpool",
        "history": [
            _history("2025-08-15 19:00:00", "h", 2.33, 1.57, 149, 17, 278, 24, 9, 8, 4, 2, 1.877, "w"),
            _history("2025-08-25 19:00:00", "a", 2.40, 1.10, 120, 0, 200, 20, 7, 5, 3, 2, 2.1, "w"),
        ],
    },
    "73": {
        "id": "73", "title": "Bournemouth",
        "history": [
            _history("2025-08-15 19:00:00", "a", 1.57, 2.33, 278, 24, 149, 17, 8, 9, 2, 4, 0.4, "l"),
        ],
    },
}


def _player(pid, name, team, position):
    return {
        "id": pid, "player_name": name, "games": "4", "time": "331", "goals": "1",
        "xG": "1.2", "assists": "1", "xA": "0.9", "shots": "10", "key_passes": "6",
        "yellow_cards": "1", "red_cards": "0", "position": position, "team_title": team,
        "npg": "1", "npxG": "1.1", "xGChain": "3.1", "xGBuildup": "1.4",
    }


PLAYERS = [
    _player("9001", "Florian Wirtz", "Liverpool", "M S"),
    _player("9002", "Dominik Szoboszlai", "Liverpool", "M"),
]


def league_page() -> str:
    return understat_page(datesData=DATES, teamsData=TEAMS, playersData=PLAYERS)
```

<!-- file: backend/tests/test_understat_tables.py -->
```python
import math
from datetime import date

from backend.model.understat_tables import (
    parse_league_matches,
    parse_player_seasons,
    parse_team_match_stats,
)
from backend.tests.fixtures import league_page


def test_league_matches_keeps_only_played_matches_with_typed_values():
    matches = parse_league_matches(league_page())

    assert list(matches["understat_match_id"]) == ["1001", "1002"]
    first = matches.iloc[0]
    assert first["date"] == date(2025, 8, 15)
    assert first["home_team"] == "Liverpool"
    assert first["away_team"] == "Bournemouth"
    assert first["home_goals"] == 4
    assert first["home_xg"] == 2.33


def test_team_match_stats_has_one_row_per_team_per_match():
    stats = parse_team_match_stats(league_page())

    assert len(stats) == 3
    liverpool = stats[stats["team"] == "Liverpool"].reset_index(drop=True)
    assert liverpool.loc[0, "date"] == date(2025, 8, 15)
    assert liverpool.loc[0, "side"] == "h"
    assert math.isclose(liverpool.loc[0, "ppda"], 149 / 17)
    assert liverpool.loc[0, "deep_allowed"] == 8


def test_ppda_is_nan_when_there_are_no_defensive_actions():
    stats = parse_team_match_stats(league_page())
    liverpool = stats[stats["team"] == "Liverpool"].reset_index(drop=True)

    assert math.isnan(liverpool.loc[1, "ppda"])


def test_player_seasons_are_typed_and_tagged_with_the_season():
    players = parse_player_seasons(league_page(), 2025)

    assert len(players) == 2
    wirtz = players[players["player"] == "Florian Wirtz"].iloc[0]
    assert wirtz["season"] == 2025
    assert wirtz["minutes"] == 331
    assert wirtz["xg"] == 1.2
    assert wirtz["xg_buildup"] == 1.4
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.Reds/Scripts/python -m pytest backend/tests/test_understat_tables.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.model.understat_tables'`.

- [ ] **Step 3: Write the implementation**

<!-- file: backend/model/understat_tables.py -->
```python
import pandas as pd

from backend.ingest.understat import extract_embedded_json

MATCH_COLUMNS = [
    "understat_match_id", "date", "home_team", "away_team",
    "home_goals", "away_goals", "home_xg", "away_xg",
]
TEAM_STAT_COLUMNS = [
    "team", "date", "side", "xg", "xga", "npxg", "npxga", "ppda", "ppda_allowed",
    "deep", "deep_allowed", "goals", "goals_against", "xpts",
]
PLAYER_COLUMNS = [
    "season", "player_id", "player", "team", "position", "games", "minutes", "goals",
    "assists", "shots", "key_passes", "xg", "xa", "npg", "npxg", "xg_chain", "xg_buildup",
]


def _ratio(attempts, actions) -> float:
    actions = float(actions)
    return float(attempts) / actions if actions else float("nan")


def parse_league_matches(html: str) -> pd.DataFrame:
    rows = []
    for item in extract_embedded_json(html, "datesData"):
        if not item["isResult"]:
            continue
        rows.append({
            "understat_match_id": str(item["id"]),
            "date": pd.Timestamp(item["datetime"]).date(),
            "home_team": item["h"]["title"],
            "away_team": item["a"]["title"],
            "home_goals": int(item["goals"]["h"]),
            "away_goals": int(item["goals"]["a"]),
            "home_xg": float(item["xG"]["h"]),
            "away_xg": float(item["xG"]["a"]),
        })
    return pd.DataFrame(rows, columns=MATCH_COLUMNS)


def parse_team_match_stats(html: str) -> pd.DataFrame:
    rows = []
    for team in extract_embedded_json(html, "teamsData").values():
        for entry in team["history"]:
            rows.append({
                "team": team["title"],
                "date": pd.Timestamp(entry["date"]).date(),
                "side": entry["h_a"],
                "xg": float(entry["xG"]),
                "xga": float(entry["xGA"]),
                "npxg": float(entry["npxG"]),
                "npxga": float(entry["npxGA"]),
                "ppda": _ratio(entry["ppda"]["att"], entry["ppda"]["def"]),
                "ppda_allowed": _ratio(entry["ppda_allowed"]["att"], entry["ppda_allowed"]["def"]),
                "deep": int(entry["deep"]),
                "deep_allowed": int(entry["deep_allowed"]),
                "goals": int(entry["scored"]),
                "goals_against": int(entry["missed"]),
                "xpts": float(entry["xpts"]),
            })
    return pd.DataFrame(rows, columns=TEAM_STAT_COLUMNS)


def parse_player_seasons(html: str, season_start_year: int) -> pd.DataFrame:
    rows = []
    for p in extract_embedded_json(html, "playersData"):
        rows.append({
            "season": season_start_year,
            "player_id": str(p["id"]),
            "player": p["player_name"],
            "team": p["team_title"],
            "position": p["position"],
            "games": int(p["games"]),
            "minutes": int(p["time"]),
            "goals": int(p["goals"]),
            "assists": int(p["assists"]),
            "shots": int(p["shots"]),
            "key_passes": int(p["key_passes"]),
            "xg": float(p["xG"]),
            "xa": float(p["xA"]),
            "npg": int(p["npg"]),
            "npxg": float(p["npxG"]),
            "xg_chain": float(p["xGChain"]),
            "xg_buildup": float(p["xGBuildup"]),
        })
    return pd.DataFrame(rows, columns=PLAYER_COLUMNS)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.Reds/Scripts/python -m pytest backend/tests/test_understat_tables.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/tests/fixtures.py backend/tests/test_understat_tables.py backend/model/understat_tables.py
git commit -m "feat: parse understat league page into match, team and player tables"
```

---

### Task 4: Understat match-page shot parser and lineup-source probe

**Files:**
- Create: `backend/tests/fixtures_match.py`
- Modify: `backend/model/understat_tables.py` (append `parse_match_shots` and `SHOT_COLUMNS`)
- Test: `backend/tests/test_understat_shots.py`
- Create (notes only, no raw data): `docs/superpowers/notes/2026-09-20-lineup-source.md`

**Interfaces:**
- Consumes: `extract_embedded_json`, `understat_page` (Task 3 fixtures).
- Produces: `parse_match_shots(html) -> DataFrame` with columns `shot_id, understat_match_id, minute, team, player_id, player, assister, x, y, xg, situation, shot_type, result, last_action` (one row per shot, both teams; `assister` is `None` when unassisted).
- Produces test helper: `match_page(match_id: str) -> str` in `backend/tests/fixtures_match.py` (supports ids `"1001"` and `"1002"`).

- [ ] **Step 1: Write the synthetic fixture and the failing test**

<!-- file: backend/tests/fixtures_match.py -->
```python
"""Synthetic Understat match pages (invented shots)."""
from backend.tests.fixtures import understat_page


def _shot(shot_id, match_id, minute, side, home, away, player_id, player, assisted, x, y, xg,
          situation, shot_type, result, last_action):
    return {
        "id": shot_id, "match_id": match_id, "minute": str(minute), "h_a": side,
        "h_team": home, "a_team": away, "player_id": player_id, "player": player,
        "player_assisted": assisted, "X": str(x), "Y": str(y), "xG": str(xg),
        "situation": situation, "shotType": shot_type, "result": result,
        "lastAction": last_action,
    }


_SHOTS = {
    "1001": {
        "h": [_shot("5001", "1001", 24, "h", "Liverpool", "Bournemouth", "9002",
                    "Dominik Szoboszlai", "Florian Wirtz", 0.885, 0.5, 0.31,
                    "OpenPlay", "RightFoot", "Goal", "Pass")],
        "a": [_shot("5002", "1001", 40, "a", "Liverpool", "Bournemouth", "7001",
                    "Some Bournemouth Player", None, 0.8, 0.4, 0.05,
                    "SetPiece", "Head", "MissedShots", "Cross")],
    },
    "1002": {
        "h": [_shot("5003", "1002", 10, "h", "Newcastle United", "Liverpool", "7002",
                    "Some Newcastle Player", None, 0.9, 0.6, 0.12,
                    "OpenPlay", "LeftFoot", "SavedShot", "None")],
        "a": [_shot("5004", "1002", 70, "a", "Newcastle United", "Liverpool", "9001",
                    "Florian Wirtz", "Dominik Szoboszlai", 0.86, 0.45, 0.22,
                    "OpenPlay", "RightFoot", "Goal", "Pass")],
    },
}


def match_page(match_id: str) -> str:
    return understat_page(shotsData=_SHOTS[match_id])
```

<!-- file: backend/tests/test_understat_shots.py -->
```python
from backend.model.understat_tables import parse_match_shots
from backend.tests.fixtures_match import match_page


def test_shots_from_both_teams_are_returned_with_the_shooting_team():
    shots = parse_match_shots(match_page("1001"))

    assert list(shots["shot_id"]) == ["5001", "5002"]
    assert list(shots["team"]) == ["Liverpool", "Bournemouth"]
    assert set(shots["understat_match_id"]) == {"1001"}


def test_shot_fields_are_typed_and_mapped():
    first = parse_match_shots(match_page("1001")).iloc[0]

    assert first["minute"] == 24
    assert first["xg"] == 0.31
    assert first["x"] == 0.885
    assert first["assister"] == "Florian Wirtz"
    assert first["situation"] == "OpenPlay"
    assert first["shot_type"] == "RightFoot"
    assert first["result"] == "Goal"


def test_unassisted_shot_has_no_assister():
    second = parse_match_shots(match_page("1001")).iloc[1]

    assert second["assister"] is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.Reds/Scripts/python -m pytest backend/tests/test_understat_shots.py -v`
Expected: FAIL with `ImportError: cannot import name 'parse_match_shots'`.

- [ ] **Step 3: Append the implementation to `backend/model/understat_tables.py`**

<!-- append: backend/model/understat_tables.py -->
```python
SHOT_COLUMNS = [
    "shot_id", "understat_match_id", "minute", "team", "player_id", "player", "assister",
    "x", "y", "xg", "situation", "shot_type", "result", "last_action",
]


def parse_match_shots(html: str) -> pd.DataFrame:
    data = extract_embedded_json(html, "shotsData")
    rows = []
    for side in ("h", "a"):
        for shot in data[side]:
            rows.append({
                "shot_id": str(shot["id"]),
                "understat_match_id": str(shot["match_id"]),
                "minute": int(shot["minute"]),
                "team": shot["h_team"] if shot["h_a"] == "h" else shot["a_team"],
                "player_id": str(shot["player_id"]),
                "player": shot["player"],
                "assister": shot.get("player_assisted") or None,
                "x": float(shot["X"]),
                "y": float(shot["Y"]),
                "xg": float(shot["xG"]),
                "situation": shot["situation"],
                "shot_type": shot["shotType"],
                "result": shot["result"],
                "last_action": shot.get("lastAction"),
            })
    return pd.DataFrame(rows, columns=SHOT_COLUMNS)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.Reds/Scripts/python -m pytest backend/tests/test_understat_shots.py -v`
Expected: 3 passed.

- [ ] **Step 5: Probe one real match page for shot fields and lineup data (nothing raw is committed)**

```bash
.Reds/Scripts/python - <<'EOF'
from backend.config import load_settings
from backend.ingest.understat import PoliteFetcher, extract_embedded_json, league_url, match_url

settings = load_settings()
fetcher = PoliteFetcher(settings.cache_dir, settings.min_request_interval_seconds)
dates = extract_embedded_json(fetcher.get(league_url(2025)), "datesData")
match_id = next(d["id"] for d in dates if d["isResult"])
html = fetcher.get(match_url(match_id))

try:
    shots = extract_embedded_json(html, "shotsData")
    print("shotsData keys:", sorted(shots["h"][0]))
except KeyError:
    print("shotsData NOT PRESENT")

try:
    rosters = extract_embedded_json(html, "rostersData")
    player = next(iter(rosters["h"].values()))
    print("rostersData keys:", sorted(player))
except KeyError:
    print("rostersData NOT PRESENT")
EOF
```
Expected shot keys include `X, Y, h_a, h_team, a_team, id, lastAction, match_id, minute, player, player_assisted, player_id, result, shotType, situation, xG`. **If a key the parser uses is missing or renamed, fix `parse_match_shots` and `fixtures_match.py` before continuing.**

- [ ] **Step 6: Record the lineup-source finding**

Write `docs/superpowers/notes/2026-09-20-lineup-source.md` with: whether `rostersData` is present; if so, its keys (in particular anything giving minutes played, position, starting/substitute status); and one line of recommendation for Plan 2 — use Understat rosters for lineups if they include minutes and positions, otherwise plan for FBref per-match exports. Do not paste raw payloads, only key names.

- [ ] **Step 7: Commit**

```bash
git add backend/tests/fixtures_match.py backend/tests/test_understat_shots.py backend/model/understat_tables.py docs/superpowers/notes
git commit -m "feat: parse understat match shots; note lineup-source findings"
```

---

### Task 5: FBref fixtures CSV loader

**Files:**
- Create: `backend/tests/fixtures_fbref.py`, `backend/ingest/fbref.py`
- Test: `backend/tests/test_fbref.py`

**Interfaces:**
- Produces: `load_fixtures_csv(path, season: str) -> DataFrame` with columns `season, date, competition, venue, opponent, result, goals_for, goals_against, possession, formation, opp_formation` (played matches only; `season` like `"2025-2026"`; goals are nullable `Int64`).
- Produces: `load_all_fixtures(manual_dir) -> DataFrame` reading every `fbref_YYYY-YYYY_fixtures.csv` in the folder.
- Produces test fixture: `FIXTURES_CSV: str` in `backend/tests/fixtures_fbref.py`.

- [ ] **Step 1: Write the fixture and the failing tests**

The fixture uses the column headers FBref shows on the Scores & Fixtures table. Values are invented.

<!-- file: backend/tests/fixtures_fbref.py -->
```python
"""Synthetic FBref 'Scores & Fixtures' CSV export (invented values)."""

FIXTURES_CSV = (
    "Date,Time,Comp,Round,Day,Venue,Result,GF,GA,Opponent,Poss,Attendance,Captain,"
    "Formation,Opp Formation,Referee,Match Report,Notes\n"
    '2025-08-15,20:00,Premier League,Matchweek 1,Fri,Home,W,4,2,Bournemouth,63,"60,000",'
    "Captain A,4-2-3-1,4-2-3-1,Referee A,Match Report,\n"
    '2025-08-25,20:00,Premier League,Matchweek 2,Mon,Away,W,3,2,Newcastle,42,"52,000",'
    "Captain A,4-2-3-1,4-3-3,Referee B,Match Report,\n"
    '2025-09-17,20:00,Champions Lg,League phase,Wed,Home,W,3,2,Atletico Madrid,45,"59,000",'
    "Captain A,4-2-3-1,3-5-2,Referee C,Match Report,\n"
    "2025-09-27,15:00,Premier League,Matchweek 6,Sat,Away,,,,Crystal Palace,,,,,,,Head-to-Head,\n"
)
```

<!-- file: backend/tests/test_fbref.py -->
```python
from datetime import date

import pytest

from backend.ingest.fbref import load_all_fixtures, load_fixtures_csv
from backend.tests.fixtures_fbref import FIXTURES_CSV


def write_csv(tmp_path, name="fbref_2025-2026_fixtures.csv", text=FIXTURES_CSV):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_unplayed_fixtures_are_dropped_and_columns_renamed(tmp_path):
    fixtures = load_fixtures_csv(write_csv(tmp_path), "2025-2026")

    assert len(fixtures) == 3
    assert list(fixtures.columns) == [
        "season", "date", "competition", "venue", "opponent", "result",
        "goals_for", "goals_against", "possession", "formation", "opp_formation",
    ]
    first = fixtures.iloc[0]
    assert first["season"] == "2025-2026"
    assert first["date"] == date(2025, 8, 15)
    assert first["goals_for"] == 4
    assert first["possession"] == 63
    assert first["opp_formation"] == "4-2-3-1"


def test_missing_required_column_is_reported_with_the_file_name(tmp_path):
    path = write_csv(tmp_path, text="Date,Comp\n2025-08-15,Premier League\n")

    with pytest.raises(ValueError, match="fbref_2025-2026_fixtures.csv.*Formation"):
        load_fixtures_csv(path, "2025-2026")


def test_load_all_fixtures_reads_every_season_file(tmp_path):
    write_csv(tmp_path, "fbref_2025-2026_fixtures.csv")
    write_csv(tmp_path, "fbref_2024-2025_fixtures.csv")

    fixtures = load_all_fixtures(tmp_path)

    assert set(fixtures["season"]) == {"2024-2025", "2025-2026"}
    assert len(fixtures) == 6


def test_load_all_fixtures_fails_clearly_when_no_files_exist(tmp_path):
    with pytest.raises(FileNotFoundError, match="fbref_"):
        load_all_fixtures(tmp_path)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.Reds/Scripts/python -m pytest backend/tests/test_fbref.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.ingest.fbref'`.

- [ ] **Step 3: Write the implementation**

<!-- file: backend/ingest/fbref.py -->
```python
import re
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    "Date", "Comp", "Venue", "Result", "GF", "GA", "Opponent", "Poss",
    "Formation", "Opp Formation",
]
_FILE_NAME = re.compile(r"^fbref_(\d{4})-(\d{4})_fixtures\.csv$")


def load_fixtures_csv(path, season: str) -> pd.DataFrame:
    """Read one manually exported FBref 'Scores & Fixtures' CSV (played matches only)."""
    path = Path(path)
    df = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    df = df[df["Result"].notna()]
    return pd.DataFrame({
        "season": season,
        "date": pd.to_datetime(df["Date"]).dt.date,
        "competition": df["Comp"],
        "venue": df["Venue"],
        "opponent": df["Opponent"],
        "result": df["Result"],
        "goals_for": pd.to_numeric(df["GF"], errors="coerce").astype("Int64"),
        "goals_against": pd.to_numeric(df["GA"], errors="coerce").astype("Int64"),
        "possession": pd.to_numeric(df["Poss"], errors="coerce"),
        "formation": df["Formation"],
        "opp_formation": df["Opp Formation"],
    }).reset_index(drop=True)


def load_all_fixtures(manual_dir) -> pd.DataFrame:
    frames = []
    for path in sorted(Path(manual_dir).glob("fbref_*_fixtures.csv")):
        found = _FILE_NAME.match(path.name)
        if found is None:
            raise ValueError(f"unexpected file name: {path.name}")
        frames.append(load_fixtures_csv(path, f"{found.group(1)}-{found.group(2)}"))
    if not frames:
        raise FileNotFoundError(f"no fbref_YYYY-YYYY_fixtures.csv files in {manual_dir}")
    return pd.concat(frames, ignore_index=True)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.Reds/Scripts/python -m pytest backend/tests/test_fbref.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/tests/fixtures_fbref.py backend/tests/test_fbref.py backend/ingest/fbref.py
git commit -m "feat: load manually exported fbref fixtures csv files"
```

---

### Task 6: Club name mapping and the match join

**Files:**
- Create: `config/name_map.yaml`, `backend/model/names.py`, `backend/model/matches.py`, `backend/tests/fixtures_model.py`
- Test: `backend/tests/test_names.py`, `backend/tests/test_matches.py`

**Interfaces:**
- Consumes: `Era`, `era_for` (Task 1); `parse_league_matches`, `parse_team_match_stats` (Task 3); FBref fixtures frame (Task 5).
- Produces: `normalise(name: str) -> str`; `NameMap(teams: dict[str, list[str]])` with `.team(name) -> str | None`; `load_name_map(path) -> NameMap`.
- Produces: `CLUB = "Liverpool"`; `build_matches(fbref, us_matches, us_team_stats, name_map, eras) -> tuple[DataFrame, list[str]]`. The frame has the FBref columns plus `understat_match_id, xg, xga, npxg, npxga, ppda, ppda_allowed, deep, deep_allowed, manager`; the list holds human-readable problems.
- Produces test constants: `NAME_MAP`, `ERAS` in `backend/tests/fixtures_model.py`.

- [ ] **Step 1: Write the failing name tests**

<!-- file: backend/tests/test_names.py -->
```python
import pytest

from backend.model.names import NameMap, normalise


def test_normalise_strips_accents_case_and_punctuation():
    assert normalise("Atlético  Madrid") == "atletico madrid"
    assert normalise("Nott'ham Forest") == "nott ham forest"


def test_aliases_resolve_to_the_canonical_name():
    names = NameMap({"Manchester United": ["Man Utd", "Manchester Utd"]})

    assert names.team("Man Utd") == "Manchester United"
    assert names.team("manchester utd") == "Manchester United"
    assert names.team("Manchester United") == "Manchester United"


def test_unknown_team_returns_none():
    assert NameMap({"Arsenal": []}).team("Nowhere FC") is None


def test_an_alias_claimed_by_two_teams_is_rejected():
    with pytest.raises(ValueError, match="alias"):
        NameMap({"Leeds United": ["Leeds"], "Leeds City": ["Leeds"]})
```

- [ ] **Step 2: Run to verify failure**

Run: `.Reds/Scripts/python -m pytest backend/tests/test_names.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.model.names'`.

- [ ] **Step 3: Write `names.py` and the real name map**

<!-- file: backend/model/names.py -->
```python
import re
import unicodedata
from pathlib import Path

import yaml


def normalise(name: str) -> str:
    text = unicodedata.normalize("NFKD", name)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


class NameMap:
    """Maps the club names used by different sources to one canonical name."""

    def __init__(self, teams: dict[str, list[str]]):
        self._lookup: dict[str, str] = {}
        for canonical, aliases in teams.items():
            for alias in [canonical, *aliases]:
                key = normalise(alias)
                if self._lookup.get(key, canonical) != canonical:
                    raise ValueError(
                        f"alias {alias!r} claimed by both {self._lookup[key]!r} and {canonical!r}"
                    )
                self._lookup[key] = canonical

    def team(self, name: str) -> str | None:
        return self._lookup.get(normalise(name))


def load_name_map(path) -> NameMap:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return NameMap(raw["teams"])
```

<!-- file: config/name_map.yaml -->
```yaml
# Canonical club name -> aliases used by Understat and FBref.
# Unmatched names are reported by the build; add them here.
teams:
  Arsenal: []
  Aston Villa: []
  Bournemouth: [AFC Bournemouth]
  Brentford: []
  Brighton: [Brighton & Hove Albion, Brighton and Hove Albion]
  Burnley: []
  Cardiff City: [Cardiff]
  Chelsea: []
  Coventry City: [Coventry]
  Crystal Palace: []
  Everton: []
  Fulham: []
  Huddersfield Town: [Huddersfield]
  Hull City: [Hull]
  Ipswich Town: [Ipswich]
  Leeds United: [Leeds]
  Leicester City: [Leicester]
  Liverpool: []
  Luton Town: [Luton]
  Manchester City: [Man City]
  Manchester United: [Man Utd, Manchester Utd, Man United]
  Newcastle United: [Newcastle, Newcastle Utd]
  Norwich City: [Norwich]
  Nottingham Forest: [Nott'ham Forest, Nott'm Forest, Nottingham]
  Sheffield United: [Sheffield Utd]
  Southampton: []
  Stoke City: [Stoke]
  Sunderland: []
  Swansea City: [Swansea]
  Tottenham Hotspur: [Tottenham, Spurs]
  Watford: []
  West Bromwich Albion: [West Brom]
  West Ham United: [West Ham]
  Wolverhampton Wanderers: [Wolves, Wolverhampton]
```

- [ ] **Step 4: Run to verify the name tests pass**

Run: `.Reds/Scripts/python -m pytest backend/tests/test_names.py -v`
Expected: 4 passed.

- [ ] **Step 5: Write the shared model fixtures and the failing join tests**

<!-- file: backend/tests/fixtures_model.py -->
```python
from datetime import date

from backend.model.eras import Era
from backend.model.names import NameMap

NAME_MAP = NameMap({
    "Liverpool": [],
    "Bournemouth": [],
    "Newcastle United": ["Newcastle"],
    "Crystal Palace": [],
})

ERAS = [Era("Slot", date(2024, 6, 1), None)]
```

<!-- file: backend/tests/test_matches.py -->
```python
import math

from backend.model.matches import build_matches
from backend.model.understat_tables import parse_league_matches, parse_team_match_stats
from backend.ingest.fbref import load_fixtures_csv
from backend.tests.fixtures import league_page
from backend.tests.fixtures_fbref import FIXTURES_CSV
from backend.tests.fixtures_model import ERAS, NAME_MAP


def build(tmp_path, csv_text=FIXTURES_CSV):
    path = tmp_path / "fbref_2025-2026_fixtures.csv"
    path.write_text(csv_text, encoding="utf-8")
    fbref = load_fixtures_csv(path, "2025-2026")
    html = league_page()
    return build_matches(
        fbref, parse_league_matches(html), parse_team_match_stats(html), NAME_MAP, ERAS
    )


def test_league_matches_get_understat_stats_and_cup_matches_do_not(tmp_path):
    matches, problems = build(tmp_path)

    assert problems == []
    assert len(matches) == 3
    league = matches[matches["competition"] == "Premier League"]
    assert list(league["understat_match_id"]) == ["1001", "1002"]
    assert league.iloc[0]["xg"] == 2.33
    assert math.isclose(league.iloc[0]["ppda"], 149 / 17)
    cup = matches[matches["competition"] == "Champions Lg"].iloc[0]
    assert math.isnan(cup["xg"])
    assert cup["opponent"] == "Atletico Madrid"


def test_every_match_is_tagged_with_its_manager_and_opponents_are_canonical(tmp_path):
    matches, _ = build(tmp_path)

    assert set(matches["manager"]) == {"Slot"}
    assert "Newcastle United" in set(matches["opponent"])


def test_fbref_league_match_missing_from_understat_is_reported(tmp_path):
    extra = FIXTURES_CSV + (
        '2025-10-04,15:00,Premier League,Matchweek 7,Sat,Away,D,1,1,Crystal Palace,55,'
        '"25,000",Captain A,4-2-3-1,4-4-2,Referee D,Match Report,\n'
    )

    _, problems = build(tmp_path, extra)

    assert any("Crystal Palace" in p and "no Understat match" in p for p in problems)


def test_understat_match_missing_from_fbref_is_reported(tmp_path):
    only_first = "\n".join(FIXTURES_CSV.splitlines()[:2]) + "\n"

    _, problems = build(tmp_path, only_first)

    assert any("Understat match 1002" in p for p in problems)


def test_unknown_fbref_opponent_is_reported_not_dropped(tmp_path):
    renamed = FIXTURES_CSV.replace(",Bournemouth,", ",Bornemouth FC,")

    matches, problems = build(tmp_path, renamed)

    assert any("Bornemouth FC" in p for p in problems)
    assert "Bornemouth FC" in set(matches["opponent"])
```

- [ ] **Step 6: Run to verify failure**

Run: `.Reds/Scripts/python -m pytest backend/tests/test_matches.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.model.matches'`.

- [ ] **Step 7: Write the implementation**

<!-- file: backend/model/matches.py -->
```python
import pandas as pd

from backend.model.eras import Era, era_for
from backend.model.names import NameMap

CLUB = "Liverpool"
LEAGUE = "Premier League"
TEAM_STAT_COLUMNS = ["xg", "xga", "npxg", "npxga", "ppda", "ppda_allowed", "deep", "deep_allowed"]


def build_matches(
    fbref: pd.DataFrame,
    us_matches: pd.DataFrame,
    us_team_stats: pd.DataFrame,
    name_map: NameMap,
    eras: list[Era],
) -> tuple[pd.DataFrame, list[str]]:
    """One row per Liverpool match (all competitions, from FBref), with Understat
    stats attached to league matches and a manager era on every row.
    Returns the table and a list of human-readable problems."""
    problems: list[str] = []

    fb = fbref.copy()
    is_league = fb["competition"] == LEAGUE
    raw_opponents = fb.loc[is_league, "opponent"]
    canonical = raw_opponents.map(name_map.team)
    for raw in raw_opponents[canonical.isna()].unique():
        problems.append(f"FBref opponent not in name_map: {raw!r}")
    fb.loc[is_league, "opponent"] = canonical.fillna(raw_opponents)

    us = us_matches.copy()
    us["home"] = us["home_team"].map(name_map.team)
    us["away"] = us["away_team"].map(name_map.team)
    unknown = pd.concat([
        us.loc[us["home"].isna(), "home_team"],
        us.loc[us["away"].isna(), "away_team"],
    ]).unique()
    for raw in unknown:
        problems.append(f"Understat team not in name_map: {raw!r}")
    ours = us[(us["home"] == CLUB) | (us["away"] == CLUB)].copy()
    ours["opponent"] = ours["away"].where(ours["home"] == CLUB, ours["home"])
    ours["competition"] = LEAGUE
    ours = ours[["understat_match_id", "date", "opponent", "competition"]]

    merged = fb.merge(ours, how="left", on=["date", "opponent", "competition"])

    matched_ids = set(merged["understat_match_id"].dropna())
    for _, row in ours.iterrows():
        if row["understat_match_id"] not in matched_ids:
            problems.append(
                f"Understat match {row['understat_match_id']} ({row['date']} vs "
                f"{row['opponent']}) has no FBref fixture"
            )
    missing = merged[(merged["competition"] == LEAGUE) & merged["understat_match_id"].isna()]
    for _, row in missing.iterrows():
        problems.append(
            f"FBref league match {row['date']} vs {row['opponent']} has no Understat match"
        )

    stats = us_team_stats[us_team_stats["team"] == CLUB][["date", *TEAM_STAT_COLUMNS]]
    merged = merged.merge(stats, how="left", on="date")
    no_stats = merged[merged["understat_match_id"].notna() & merged["xg"].isna()]
    for _, row in no_stats.iterrows():
        problems.append(f"no Understat team stats for {CLUB} on {row['date']}")

    merged["manager"] = merged["date"].map(lambda d: era_for(d, eras))
    for d in merged.loc[merged["manager"].isna(), "date"]:
        problems.append(f"no manager era covers {d}")

    return merged.sort_values("date").reset_index(drop=True), problems
```

- [ ] **Step 8: Run to verify the join tests pass**

Run: `.Reds/Scripts/python -m pytest backend/tests/test_matches.py backend/tests/test_names.py -v`
Expected: 9 passed.

- [ ] **Step 9: Commit**

```bash
git add config/name_map.yaml backend/model/names.py backend/model/matches.py backend/tests/fixtures_model.py backend/tests/test_names.py backend/tests/test_matches.py
git commit -m "feat: club name mapping and fbref/understat match join with problem report"
```

---

### Task 7: DuckDB build pipeline and CLI

**Files:**
- Create: `backend/model/build.py`
- Test: `backend/tests/test_build.py`

**Interfaces:**
- Consumes: everything above. The fetcher argument only needs a `.get(url) -> str` method.
- Produces: `season_start_year(season: str) -> int`; `build_database(*, fetcher, fixtures, name_map, eras, con) -> list[str]` (creates tables `matches`, `shots`, `player_season_stats` in the given DuckDB connection and returns the problem list); `main() -> int` (CLI; exit code 1 if there are problems).

- [ ] **Step 1: Write the failing tests**

<!-- file: backend/tests/test_build.py -->
```python
import duckdb

from backend.ingest.fbref import load_all_fixtures
from backend.ingest.understat import league_url, match_url
from backend.model.build import build_database, season_start_year
from backend.tests.fixtures import league_page
from backend.tests.fixtures_fbref import FIXTURES_CSV
from backend.tests.fixtures_match import match_page
from backend.tests.fixtures_model import ERAS, NAME_MAP


class FakeFetcher:
    def __init__(self, pages):
        self.pages = pages
        self.requested = []

    def get(self, url):
        self.requested.append(url)
        return self.pages[url]


def test_season_start_year():
    assert season_start_year("2025-2026") == 2025


def test_build_database_writes_all_tables(tmp_path):
    (tmp_path / "fbref_2025-2026_fixtures.csv").write_text(FIXTURES_CSV, encoding="utf-8")
    fetcher = FakeFetcher({
        league_url(2025): league_page(),
        match_url("1001"): match_page("1001"),
        match_url("1002"): match_page("1002"),
    })
    con = duckdb.connect(":memory:")

    problems = build_database(
        fetcher=fetcher,
        fixtures=load_all_fixtures(tmp_path),
        name_map=NAME_MAP,
        eras=ERAS,
        con=con,
    )

    assert problems == []
    assert con.sql("SELECT count(*) FROM matches").fetchone()[0] == 3
    assert con.sql("SELECT count(*) FROM matches WHERE xg IS NULL").fetchone()[0] == 1
    assert con.sql("SELECT count(*) FROM shots").fetchone()[0] == 4
    assert con.sql("SELECT count(*) FROM player_season_stats").fetchone()[0] == 2
    assert con.sql("SELECT DISTINCT manager FROM matches").fetchall() == [("Slot",)]


def test_match_pages_are_only_requested_for_matched_league_matches(tmp_path):
    (tmp_path / "fbref_2025-2026_fixtures.csv").write_text(FIXTURES_CSV, encoding="utf-8")
    fetcher = FakeFetcher({
        league_url(2025): league_page(),
        match_url("1001"): match_page("1001"),
        match_url("1002"): match_page("1002"),
    })

    build_database(
        fetcher=fetcher,
        fixtures=load_all_fixtures(tmp_path),
        name_map=NAME_MAP,
        eras=ERAS,
        con=duckdb.connect(":memory:"),
    )

    assert sorted(fetcher.requested) == sorted(
        [league_url(2025), match_url("1001"), match_url("1002")]
    )
```

- [ ] **Step 2: Run to verify failure**

Run: `.Reds/Scripts/python -m pytest backend/tests/test_build.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.model.build'`.

- [ ] **Step 3: Write the implementation**

<!-- file: backend/model/build.py -->
```python
import duckdb
import pandas as pd

from backend.config import ROOT, load_settings
from backend.ingest.fbref import load_all_fixtures
from backend.ingest.understat import PoliteFetcher, league_url, match_url
from backend.model.eras import load_eras
from backend.model.matches import build_matches
from backend.model.names import load_name_map
from backend.model.understat_tables import (
    parse_league_matches,
    parse_match_shots,
    parse_player_seasons,
    parse_team_match_stats,
)


def season_start_year(season: str) -> int:
    return int(season.split("-")[0])


def _write(con, name: str, frame: pd.DataFrame) -> None:
    con.register("_frame", frame.convert_dtypes())
    con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _frame")
    con.unregister("_frame")


def build_database(*, fetcher, fixtures, name_map, eras, con) -> list[str]:
    """Fetch Understat data for every season present in `fixtures`, join it with
    the FBref fixtures, and write `matches`, `shots` and `player_season_stats`.
    Returns the list of problems found (unmatched names or matches)."""
    us_matches, us_team_stats, players = [], [], []
    for season in sorted(fixtures["season"].unique()):
        year = season_start_year(season)
        html = fetcher.get(league_url(year))
        us_matches.append(parse_league_matches(html))
        us_team_stats.append(parse_team_match_stats(html))
        players.append(parse_player_seasons(html, year))

    matches, problems = build_matches(
        fixtures,
        pd.concat(us_matches, ignore_index=True),
        pd.concat(us_team_stats, ignore_index=True),
        name_map,
        eras,
    )

    shot_frames = [
        parse_match_shots(fetcher.get(match_url(match_id)))
        for match_id in matches["understat_match_id"].dropna().unique()
    ]
    shots = pd.concat(shot_frames, ignore_index=True) if shot_frames else pd.DataFrame()

    _write(con, "matches", matches)
    _write(con, "shots", shots)
    _write(con, "player_season_stats", pd.concat(players, ignore_index=True))
    return problems


def main() -> int:
    settings = load_settings()
    eras = load_eras(ROOT / "config" / "eras.yaml")
    name_map = load_name_map(ROOT / "config" / "name_map.yaml")
    fixtures = load_all_fixtures(settings.manual_dir)
    fetcher = PoliteFetcher(settings.cache_dir, settings.min_request_interval_seconds)

    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(settings.db_path))
    problems = build_database(
        fetcher=fetcher, fixtures=fixtures, name_map=name_map, eras=eras, con=con
    )
    con.close()

    print(f"Wrote {settings.db_path}")
    if problems:
        print(f"{len(problems)} problem(s) found:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("No problems found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run to verify the tests pass**

Run: `.Reds/Scripts/python -m pytest backend/tests/test_build.py -v`
Expected: 3 passed. If DuckDB rejects a pandas dtype produced by `convert_dtypes()`, fix `_write` (not the test) so the same assertions hold, including `xg IS NULL` counting the non-league match.

- [ ] **Step 5: Run the whole suite**

Run: `.Reds/Scripts/python -m pytest -v`
Expected: 34 passed.

- [ ] **Step 6: Commit**

```bash
git add backend/model/build.py backend/tests/test_build.py
git commit -m "feat: build duckdb tables from understat and fbref data"
```

---

### Task 8: README and the first real build

**Files:**
- Create: `README.md`

This task needs the user's hands: FBref cannot be automated.

- [ ] **Step 1: Write the README**

<!-- file: README.md -->
````markdown
# RedsAhead

*One step ahead, by the numbers.*

Interactive dashboard combining player analysis, tactical analysis and scouting for Liverpool FC, built on free data. Unofficial fan project, not affiliated with Liverpool FC. Personal, non-commercial use.

Design: `docs/superpowers/specs/2026-09-19-redsahead-design.md`.

## Data sources and credit

- **Understat** (https://understat.com): xG, PPDA, deep completions and shot-level data. Fetched politely (cached, one request per three seconds).
- **FBref** (https://fbref.com): fixtures, possession and formations. Exported **by hand** as CSV.

Raw data is never committed (see `.gitignore`).

## Setup

```bash
.Reds/Scripts/python -m pip install -e ".[dev]"
.Reds/Scripts/python -m pytest
```

## Loading FBref data

For each season from 2017-2018 to the current one:

1. Open the Liverpool "All Competitions" page for that season on FBref.
2. On the **Scores & Fixtures** table choose *Share & Export → Get table as CSV* and copy the text.
3. Save it as `backend/ingest/manual/fbref_YYYY-YYYY_fixtures.csv`, for example `fbref_2017-2018_fixtures.csv`. Create the folder if it does not exist.

The header row must include: Date, Comp, Venue, Result, GF, GA, Opponent, Poss, Formation, Opp Formation.

## Building the database

```bash
.Reds/Scripts/python -m backend.model.build
```

This fetches each Understat season page and each Liverpool league match page once (cached afterwards), joins them with your FBref CSVs and writes `data/redsahead.duckdb`. It lists any club names or matches it could not match. Add missing club names to `config/name_map.yaml` and rerun; cached pages are reused.
````

- [ ] **Step 2: Ask the user to export the FBref fixtures CSVs**

Ask the user to save one `fbref_YYYY-YYYY_fixtures.csv` per season in `backend/ingest/manual/` (start with just 2025-2026 to prove the pipeline, then add the rest).

- [ ] **Step 3: Run the real build**

Run: `.Reds/Scripts/python -m backend.model.build`
Expected: `Wrote .../data/redsahead.duckdb`. With one season this makes about 40 requests at three seconds each (roughly two minutes). A problem list is normal on the first run.

- [ ] **Step 4: Fix reported problems**

For each `not in name_map` line, add the alias to `config/name_map.yaml` under the right club and rerun (cache makes reruns fast). For `no Understat match` / `has no FBref fixture` lines, check dates and opponent names for that row in the CSV. Repeat until the run prints `No problems found.` for the season.

- [ ] **Step 5: Sanity-check the database**

```bash
.Reds/Scripts/python - <<'EOF'
import duckdb
con = duckdb.connect("data/redsahead.duckdb", read_only=True)
print(con.sql("SELECT manager, count(*) AS matches, round(avg(ppda), 2) AS avg_ppda, round(avg(xg), 2) AS avg_xg FROM matches GROUP BY manager").df())
print(con.sql("SELECT count(*) AS shots FROM shots").fetchone())
EOF
```
Expected: one row per manager present in the season, plausible values (PPDA in single digits to the teens, xG per match around 1 to 3), and a non-zero shot count. Show the output to the user.

- [ ] **Step 6: Add the remaining seasons**

With the user, add the other seasons' CSVs and rerun step 3 and step 4 until clean.

- [ ] **Step 7: Commit**

```bash
git add README.md config/name_map.yaml
git commit -m "docs: readme with data setup; name map fixes from first real build"
```

---

## Self-Review

**Spec coverage** (spec sections against tasks):
- §2 sources and roles: Tasks 2-5. Understat backbone and FBref manual CSV are covered.
- §3 `ingest/`: Tasks 2, 4, 5. `model/`: Tasks 1, 3, 4, 6, 7. Cross-source join via `name_map.yaml` and unmatched names reported: Task 6.
- §10 licensing: `.gitignore`, polite fetcher, synthetic fixtures, README credit (Tasks 1, 2, 8).
- Eras from `eras.yaml`, thresholds in `settings.yaml`: Task 1.
- Not in this plan (belongs to later plans): `analysis/` (era comparison, match diagnosis, on/off, ridge lineup effects, scouting), `api/`, `web/`, the UI labels and warnings.
- Known gap carried forward: lineups and minutes (needed by on/off and the ridge model) are decided by the Task 4 probe and handled at the start of Plan 2.

**Placeholder scan:** the only unspecified value is the Iraola start date, which Task 1 Step 7 obtains from the user by design. No other TBD/TODO.

**Type consistency:** `build_matches` returns `(DataFrame, list[str])` and is called that way in Tasks 6 and 7. `PoliteFetcher.get(url)` is the only fetcher interface used by `build_database` and `FakeFetcher`. Column names `understat_match_id`, `xg`, `ppda`, `manager` are identical across parsers, join and tests.
