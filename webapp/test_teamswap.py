"""Self-check for the cross-team swap guard: a track must split when sustained
evidence shows the OTHER team's jersey, but must NOT split on classifier noise.
Run: python3 test_teamswap.py"""
import app


def _idf():
    # Home = dark (black), Away = light (white) — the usual broadcast pairing.
    return app.PlayerIdentifier(
        {"Home": ["black"], "Away": ["white"]},
        {"14": "Ivanov"}, {"4": "Petrov"},
        app.BASIC_COLORS, number_model=None)


def test_sustained_other_team_is_a_swap():
    idf = _idf()
    idf._est_team[7] = "Home"
    for _ in range(idf.TEAM_SWITCH_CONFIRM):
        idf._push_team_evidence(7, "Away", 1.0)
    assert idf._team_swap_target(7) == "Away"


def test_not_enough_reads_no_swap():
    idf = _idf()
    idf._est_team[7] = "Home"
    for _ in range(idf.TEAM_SWITCH_CONFIRM - 1):
        idf._push_team_evidence(7, "Away", 1.0)
    assert idf._team_swap_target(7) is None     # too little evidence yet


def test_noise_does_not_split():
    idf = _idf()
    idf._est_team[7] = "Home"
    # Mostly the established team with a couple of flickering reads -> no swap.
    for i in range(idf.TEAM_WINDOW):
        idf._push_team_evidence(7, "Away" if i % 5 == 0 else "Home", 1.0)
    assert idf._team_swap_target(7) is None


def test_swap_resets_stale_identity():
    idf = _idf()
    idf._est_team[7] = "Home"
    idf._team_votes[7] = [0.0, 50.0]        # heavy accumulated "dark/Home" history
    idf._est_number[7] = 14
    idf._cache[7] = {"name": "Ivanov", "number": 14, "team": "Home", "last_frame": 1}
    idf._apply_team_swap(7, "Away")
    assert idf._est_team[7] == "Away"
    assert 7 not in idf._est_number and 7 not in idf._cache
    # the sticky tally must now favour the NEW team, not the old 50 units
    v = idf._team_votes[7]
    light_first = idf._light_team == "Away"
    assert (v[0] > v[1]) == light_first, (v, idf._light_team)


if __name__ == "__main__":
    test_sustained_other_team_is_a_swap()
    test_not_enough_reads_no_swap()
    test_noise_does_not_split()
    test_swap_resets_stale_identity()
    print("team swap guard OK")
