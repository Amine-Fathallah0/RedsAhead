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
