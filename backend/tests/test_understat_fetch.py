from pathlib import Path

import pytest
import os
import time


from backend.ingest.understat import extract_embedded_json
from backend.ingest.polite_fetcher import PoliteFetcher
class FakeResponse:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


class FakeSession:
    """Stands in for the internet: records which addresses were requested."""

    def __init__(self, pages):
        self.pages = pages
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append(url)
        return FakeResponse(self.pages[url])

def test_extract_decodes_hex_escapes():
    payload = r"[\x7B\x22title\x22\x3A\x22Liverpool\x22\x7D]"
    html = f"<html><script>var datesData = JSON.parse('{payload}');</script></html>"

    result = extract_embedded_json(html, "datesData")

    assert result == [{"title": "Liverpool"}]

def test_extract_raises_key_error_when_variable_missing():
    with pytest.raises(KeyError, match="teamsData"):
        extract_embedded_json("<html><body><script>var falsedata = JSON.parse('{\"name\": \"Liverpool\"}');</script></body></html>", "teamsData")

def test_extract_keeps_literal_accented_characters():
    payload=r"[\x7B\x22name\x22\x3A\x22Jérémy\x22\x7D]"
    html = f"<html><script>var playersData = JSON.parse('{payload}');</script></html>"
    result = extract_embedded_json(html, "playersData")
    assert result == [{"name": "Jérémy"}]

def test_extract_handles_characters_outside_latin1():
    payload = r"[\x7B\x22name\x22\x3A\x22Çağlar Söyüncü\x22\x7D]"
    html = f"<html><script>var playersData = JSON.parse('{payload}');</script></html>"

    result = extract_embedded_json(html, "playersData")

    assert result == [{"name": "Çağlar Söyüncü"}]

def test_second_fetch_of_same_url_comes_from_cache(tmp_path: Path):
    session = FakeSession({"u1": "<html>one</html>"})
    fetcher = PoliteFetcher(tmp_path, 3.0, session=session)

    first = fetcher.get("u1")
    second = fetcher.get("u1")

    assert first == "<html>one</html>"
    assert second == "<html>one</html>"
    assert session.calls == ["u1"]


def test_uncached_requests_are_spaced_by_the_minimum_interval(tmp_path: Path):
    sleeps = []
    session = FakeSession({"a": "A", "b": "B"})
    fetcher = PoliteFetcher(
        tmp_path, 3.0, session=session, sleep=sleeps.append, clock=lambda: 100.0
    )

    fetcher.get("a")
    assert sleeps == []

    fetcher.get("b")
    assert sleeps == [3.0]


def test_cached_pages_do_not_trigger_a_wait(tmp_path: Path):
    sleeps = []
    session = FakeSession({"a": "A"})
    fetcher = PoliteFetcher(
        tmp_path, 3.0, session=session, sleep=sleeps.append, clock=lambda: 100.0
    )

    fetcher.get("a")
    fetcher.get("a")

    assert sleeps == []  



def test_expired_cache_entry_is_downloaded_again(tmp_path: Path):
    session = FakeSession({"u1": "<html>one</html>"})
    fetcher = PoliteFetcher(tmp_path, 0, session=session, sleep=lambda seconds: None)

    fetcher.get("u1")
    cached_file = next(tmp_path.glob("*.html"))
    two_days_ago = time.time() - 2 * 24 * 3600
    os.utime(cached_file, (two_days_ago, two_days_ago))

    fetcher.get("u1", max_age_seconds=3600)

    assert session.calls == ["u1", "u1"]

def test_fresh_cache_entry_is_not_downloaded_again(tmp_path):
    session= FakeSession({"u1": "<html>one</html>"})
    fetcher=PoliteFetcher(tmp_path, 0, session=session, sleep=lambda seconds: None)

    fetcher.get("u1")
    cached_file=next(tmp_path.glob("*.html"))
    less_than_one_hour_ago=time.time() - 3400
    os.utime(cached_file, (less_than_one_hour_ago, less_than_one_hour_ago))

    fetcher.get("u1", max_age_seconds=3600)
    assert session.calls==["u1"]