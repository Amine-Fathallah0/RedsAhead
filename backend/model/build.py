import duckdb
import pandas as pd

from backend.ingest.fbref import load_all_fixtures
from backend.ingest.understat import fetch_league_data, parse_league_matches, fetch_match_data, parse_team_match_stats,parse_player_seasons, parse_match_shots, parse_match_rosters

from backend.model.matches import build_matches
from backend.model.eras import Era, load_eras
from backend.model.names import load_name_map

from backend.ingest.polite_fetcher import PoliteFetcher
from backend.config import load_settings

#---- building matches table from build matches 
settings = load_settings()
fetcher = PoliteFetcher(settings.cache_dir, settings.min_request_interval_seconds)
FBref = load_all_fixtures(settings.manual_dir)
match_frames, stat_frames, player_frames = [], [], []
for season in (2024, 2025, 2026):
    data = fetch_league_data(fetcher, "EPL", season)
    match_frames.append(parse_league_matches(data))
    stat_frames.append(parse_team_match_stats(data))
    player_frames.append(parse_player_seasons(data, season))

understat_matches = pd.concat(match_frames, ignore_index=True)
understat_team_stats = pd.concat(stat_frames, ignore_index=True)
understat_player_stats = pd.concat(player_frames, ignore_index=True)

name_map = load_name_map("config/name_map.yaml")
eras = load_eras("config/eras.yaml")

dfBuildMatches, problems = build_matches(FBref, understat_matches, understat_team_stats, name_map, eras)

#matches done ---

# ---- shots and rosters: one fetch per Liverpool league match ----
# dfBuildMatches already narrowed this down to Liverpool's own matches (the join
# in build_matches only ever attaches an understat_match_id to Liverpool's rows),
# so this is 81 requests (one per Premier League match), not all 380 in the league.
liverpool_match_ids = dfBuildMatches["understat_match_id"].dropna().unique()
shot_frames, roster_frames = [], []
for match_id in liverpool_match_ids:
    match_data = fetch_match_data(fetcher, match_id)
    shot_frames.append(parse_match_shots(match_data))
    roster_frames.append(parse_match_rosters(match_data, match_id))

shots_df = pd.concat(shot_frames, ignore_index=True)
rosters_df = pd.concat(roster_frames, ignore_index=True)

# ---- write everything to the database ----
settings.db_path.parent.mkdir(parents=True, exist_ok=True)
conn = duckdb.connect(database=str(settings.db_path))
conn.execute("CREATE OR REPLACE TABLE matches AS SELECT * FROM dfBuildMatches")
conn.execute("CREATE OR REPLACE TABLE shots AS SELECT * FROM shots_df")
conn.execute("CREATE OR REPLACE TABLE player_season_stats AS SELECT * FROM understat_player_stats")
conn.execute("CREATE OR REPLACE TABLE rosters AS SELECT * FROM rosters_df")
conn.close()

print(problems)
print(
    f"Built {settings.db_path}: {len(dfBuildMatches)} matches, {len(shots_df)} shots, "
    f"{len(understat_player_stats)} player-seasons, {len(rosters_df)} roster entries."
)
