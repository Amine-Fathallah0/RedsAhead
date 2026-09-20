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
