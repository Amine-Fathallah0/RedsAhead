import pytest

from backend.model.names import NameMap, normalise


def test_normalise_strips_accents_case_and_punctuation():
    assert normalise("Atlético  Madrid") == "atletico madrid"
    assert normalise("Nott'ham Forest") == "nott ham forest"


def test_normalise_strips_a_leftover_country_flag_code():
    # FBref's little flag icon doesn't survive copy-paste; its two-letter
    # country code is left behind as plain text in front of the club name.
    assert normalise("es Atlético Madrid") == normalise("Atlético Madrid")
    assert normalise("az Qarabağ") == normalise("Qarabağ")
    assert normalise("tr Galatasaray") == normalise("Galatasaray")


def test_normalise_does_not_strip_a_real_two_letter_word():
    # "PSV" has no prefix to strip; make sure the rule only fires on an
    # actual "xx " pattern in front of more text, not real short club names.
    assert normalise("PSV") == "psv"


def test_aliases_resolve_to_the_canonical_name():
    names = NameMap({"Manchester United": ["Man Utd", "Manchester Utd"]})

    assert names.team("Man Utd") == "Manchester United"
    assert names.team("manchester utd") == "Manchester United"
    assert names.team("Manchester United") == "Manchester United"


def test_unknown_team_returns_none():
    assert NameMap({"Arsenal": []}).team("Nowhere FC") is None


def test_an_alias_claimed_by_two_teams_is_rejected():
    with pytest.raises(ValueError, match="alias"):
        NameMap({"Leeds United": ["Leeds"], "Leeds City": ["Leeds"]})
