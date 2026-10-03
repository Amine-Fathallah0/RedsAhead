import re
import unicodedata
from pathlib import Path

import yaml


_FLAG_CODE_PREFIX = re.compile(r"^[a-z]{2,3} (?=.)")


def normalise(name: str) -> str:
    """Lowercase, strip accents and punctuation, so different spellings of the
    same club compare equal (e.g. "Atlético Madrid" and "Atletico  Madrid").

    Deliberately does *not* touch a leading two-letter word: that's meaningful
    for some real clubs ("RB Leipzig", "AC Milan"), so stripping it here would
    silently corrupt those names. See NameMap.team() for where a genuine
    leftover flag-code prefix (e.g. "es Atlético Madrid") is handled instead,
    as a fallback that only fires when nothing else matched."""
    text = unicodedata.normalize("NFKD", name)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


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
        found = self._lookup.get(normalise(name))
        if found is not None:
            return found
        # Fallback only: a genuine leftover flag-code prefix (e.g. FBref's
        # "es Atlético Madrid"). Tried second, so a real club name that
        # happens to start with two letters and a space (e.g. "RB Leipzig")
        # is never affected unless the direct lookup above already failed.
        stripped = _FLAG_CODE_PREFIX.sub("", normalise(name))
        return self._lookup.get(stripped)


def load_name_map(path) -> NameMap:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return NameMap(raw["teams"])
