from backend.analysis.thresholds import display_threshold


def test_threshold_grows_with_matches_then_caps_at_270():
    assert display_threshold(0) == 0
    assert display_threshold(1) == 45
    assert display_threshold(3) == 135
    assert display_threshold(6) == 270
    assert display_threshold(38) == 270
