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
