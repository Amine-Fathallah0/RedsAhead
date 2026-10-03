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
