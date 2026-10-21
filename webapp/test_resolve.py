"""Self-check for TrackMerger.resolve_number: confidence-weighted voting + the
single-digit noise gate. Run: python3 test_resolve.py"""
import app
R = app.TrackMerger.resolve_number


def test_frequency_fallback():
    # No confs given -> plain frequency voting (backward compatible).
    assert R([24, 24, 24, 9, 9]) == 24
    assert R([47, 47, 14]) == 47


def test_single_digit_gate():
    assert R([9]) is None            # 1 obs < min_count
    assert R([9, 9]) is None         # 2 obs < SINGLE_MIN(3)
    assert R([9, 9, 9]) == 9         # 3 obs, 100% share -> ok
    assert R([9, 9, 9, 4, 4, 4]) is None  # single-digit, share 0.5 < 0.60


def test_confidence_flips_frequency():
    # Frequency says 14 (6 vs 4), but 47 is read far more confidently -> 47.
    votes = [14] * 6 + [47] * 4
    confs = [0.35] * 6 + [0.95] * 4
    assert R(votes) == 14                      # frequency alone
    assert R(votes, confs=confs) == 47         # confidence-weighted


def test_shaky_single_digit_loses():
    # A frequent but low-confidence single "9" must not beat a confident number;
    # when truly ambiguous it returns None (reviewer decides) rather than a wrong 9.
    votes = [9] * 8 + [27] * 4
    confs = [0.45] * 8 + [0.9] * 4
    assert R(votes, confs=confs) != 9


if __name__ == "__main__":
    test_frequency_fallback()
    test_single_digit_gate()
    test_confidence_flips_frequency()
    test_shaky_single_digit_loses()
    print("resolve_number OK")
