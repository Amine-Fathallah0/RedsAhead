import pytest

from backend.model.names import NameMap, normalise


def test_normalise_strips_accents_case_and_punctuation():
    assert normalise("Atlético  Madrid") == "atletico madrid"
    assert normalise("Nott'ham Forest") == "nott ham forest"


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


def test_a_leftover_country_flag_code_is_stripped_as_a_fallback():
    # FBref's little flag icon doesn't survive copy-paste; its two-letter
    # country code is left behind as plain text in front of the club name.
    names = NameMap({"Atlético Madrid": [], "Qarabağ": [], "Galatasaray": []})

    assert names.team("es Atlético Madrid") == "Atlético Madrid"
    assert names.team("az Qarabağ") == "Qarabağ"
    assert names.team("tr Galatasaray") == "Galatasaray"


def test_a_three_letter_country_code_is_stripped_as_a_fallback():
    # Some European opponents carry a three-letter code ("eng", "sct").
    names = NameMap({"Chelsea": [], "Rangers": []})

    assert names.team("eng Chelsea") == "Chelsea"
    assert names.team("sct Rangers") == "Rangers"


def test_a_real_short_prefix_in_a_canonical_name_is_not_mistaken_for_a_flag_code():
    # "RB" and "AC" are real, meaningful parts of these clubs' names, not a
    # leftover flag code; they must resolve directly, with nothing stripped.
    names = NameMap({"RB Leipzig": [], "AC Milan": []})

    assert names.team("RB Leipzig") == "RB Leipzig"
    assert names.team("AC Milan") == "AC Milan"
    # and a real flag code in front of one of these must still work
    assert names.team("de RB Leipzig") == "RB Leipzig"
