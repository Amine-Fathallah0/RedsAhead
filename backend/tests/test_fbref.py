from datetime import date

from backend.ingest.fbref import load_all_fixtures, load_fixtures_csv, parse_score

# A few real rows exported from FBref's Scores & Fixtures table (2025-2026), kept
# small on purpose: one shootout match, one normal win, one unplayed fixture.
FIXTURES_CSV = (
    "Date,Time,Comp,Round,Day,Venue,Result,GF,GA,Opponent,Poss,Attendance,Captain,"
    "Formation,Opp Formation,Referee,Match Report,Notes\n"
    "2025-08-10,15:00,FA Community Shield,FA Community Shield,Sun,Neutral,D,2 (2),"
    "2 (3),Crystal Palace,59,82645,Virgil van Dijk,4-2-3-1,3-4-3,Chris Kavanagh,"
    "Match Report,Crystal Palace won a penalty shoot-out following normal time\n"
    "2025-08-15,20:00,Premier League,Matchweek 1,Fri,Home,W,4,2,Bournemouth,61,"
    "60315,Virgil van Dijk,4-2-3-1,4-1-4-1,Anthony Taylor,Match Report,\n"
    "2026-05-30,16:00,Premier League,Matchweek 38,Sat,Away,,,,Bournemouth,,,,,,,"
    "Head-to-Head,\n"
)


# A season with no shootout at all: every GF/GA value looks like a plain
# integer, plus one blank pair for an unplayed fixture. This is the exact shape
# that made pandas silently read GF/GA as float64 and turn "2" into 2.0.
NO_SHOOTOUT_CSV = (
    "Date,Time,Comp,Round,Day,Venue,Result,GF,GA,Opponent,Poss,Attendance,Captain,"
    "Formation,Opp Formation,Referee,Match Report,Notes\n"
    "2026-08-23,16:30,Premier League,Matchweek 1,Sun,Away,D,2,2,Newcastle,61,52630,"
    "Virgil van Dijk,4-2-3-1,4-2-3-1,Stuart Attwell,Match Report,\n"
    "2027-05-30,16:00,Premier League,Matchweek 38,Sun,Home,,,,Bournemouth,,,,,,,"
    "Head-to-Head,\n"
)


def test_parse_score_on_a_normal_result():
    goals, penalties = parse_score("2")

    assert goals == 2
    assert penalties is None


def test_parse_score_on_a_penalty_shootout_result():
    goals, penalties = parse_score("2 (2)")

    assert goals == 2
    assert penalties == 2


def write_csv(tmp_path, text=FIXTURES_CSV, name="fbref_2025-2026_fixtures.csv"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_load_fixtures_csv_drops_unplayed_matches(tmp_path):
    fixtures = load_fixtures_csv(write_csv(tmp_path), "2025-2026")

    assert len(fixtures) == 2   # the third row (no Result yet) is dropped


def test_load_fixtures_csv_reads_a_normal_match(tmp_path):
    fixtures = load_fixtures_csv(write_csv(tmp_path), "2025-2026")

    bournemouth = fixtures[fixtures["opponent"] == "Bournemouth"].iloc[0]
    assert bournemouth["season"] == "2025-2026"
    assert bournemouth["date"] == date(2025, 8, 15)
    assert bournemouth["competition"] == "Premier League"
    assert bournemouth["venue"] == "Home"
    assert bournemouth["goals_for"] == 4
    assert bournemouth["goals_against"] == 2
    assert bournemouth["possession"] == 61
    assert bournemouth["formation"] == "4-2-3-1"
    assert bournemouth["opp_formation"] == "4-1-4-1"


def test_load_fixtures_csv_splits_out_a_penalty_shootout(tmp_path):
    fixtures = load_fixtures_csv(write_csv(tmp_path), "2025-2026")

    shootout = fixtures[fixtures["opponent"] == "Crystal Palace"].iloc[0]
    assert shootout["goals_for"] == 2
    assert shootout["goals_against"] == 2
    assert shootout["penalties_for"] == 2
    assert shootout["penalties_against"] == 3


def test_load_fixtures_csv_leaves_penalties_missing_for_normal_matches(tmp_path):
    from pandas import isna

    fixtures = load_fixtures_csv(write_csv(tmp_path), "2025-2026")

    bournemouth = fixtures[fixtures["opponent"] == "Bournemouth"].iloc[0]
    assert isna(bournemouth["penalties_for"])
    assert isna(bournemouth["penalties_against"])


def test_load_all_fixtures_reads_every_season_file_in_the_folder(tmp_path):
    write_csv(tmp_path, name="fbref_2024-2025_fixtures.csv")
    write_csv(tmp_path, name="fbref_2025-2026_fixtures.csv")

    fixtures = load_all_fixtures(tmp_path)

    assert set(fixtures["season"]) == {"2024-2025", "2025-2026"}
    assert len(fixtures) == 4   # 2 played matches per file


def test_load_all_fixtures_fails_clearly_when_the_folder_is_empty(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError, match="fbref_"):
        load_all_fixtures(tmp_path)


def test_load_fixtures_csv_handles_a_season_with_no_shootouts(tmp_path):
    fixtures = load_fixtures_csv(write_csv(tmp_path, text=NO_SHOOTOUT_CSV), "2026-2027")

    assert len(fixtures) == 1
    assert fixtures.iloc[0]["goals_for"] == 2


def test_load_all_fixtures_rejects_a_malformed_season_in_the_filename(tmp_path):
    import pytest

    write_csv(tmp_path, name="fbref_25-26_fixtures.csv")   # two-digit year: not fbref_YYYY-YYYY_fixtures.csv

    with pytest.raises(ValueError, match="fbref_25-26_fixtures.csv"):
        load_all_fixtures(tmp_path)
