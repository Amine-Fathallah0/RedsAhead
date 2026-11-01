MINUTES_PER_MATCH = 45
CAP_MINUTES = 270


def display_threshold(matches_played: int) -> int:
    """Minutes a player needs before his numbers are shown as reliable.

    It grows with the matches the team has played, so early in a season a
    regular starter isn't greyed out for lack of time, and it caps at three
    full matches' worth. This only controls display; the model keeps everyone."""
    return min(CAP_MINUTES, MINUTES_PER_MATCH * matches_played)
