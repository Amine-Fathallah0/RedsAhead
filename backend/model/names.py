import re
import unicodedata
from pathlib import Path

import yaml


_FLAG_CODE_PREFIX = re.compile(r"^[a-z]{2} (?=.)")


def normalise(name: str) -> str:
    """Lowercase, strip accents and punctuation, so different spellings of the
    same club compare equal (e.g. "Atlético Madrid" and "Atletico  Madrid").

    Also strips a leading two-letter country code (e.g. "es Atlético Madrid"):
    FBref shows a small flag icon before European opponents, and copy-pasting
    the table leaves that icon's alt-text behind as plain text."""
    text = unicodedata.normalize("NFKD", name)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = _FLAG_CODE_PREFIX.sub("", text.lower())
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


class NameMap:
    """Resolves any spelling FBref or Understat use for a club to one canonical
    name. Unknown names return None rather than a guess."""

    def __init__(self, teams: dict[str, list[str]]):
        self._lookup: dict[str, str] = {}
        for canonical, aliases in teams.items():
            for alias in [canonical, *aliases]:
                key = normalise(alias)
                if self._lookup.get(key, canonical) != canonical:
                    raise ValueError(
                        f"alias {alias!r} claimed by both "
                        f"{self._lookup[key]!r} and {canonical!r}"
                    )
                self._lookup[key] = canonical

    def team(self, name: str) -> str | None:
        return self._lookup.get(normalise(name))


def load_name_map(path) -> NameMap:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return NameMap(raw["teams"])
