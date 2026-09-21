
import pytest

from backend.ingest.understat import extract_embedded_json

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