"""Self-check for clip_confidence: the % shown next to every clip in review.
Run: python3 test_confidence.py"""
import app
C = app.clip_confidence


def test_unanimous_shrinks_with_sample_size():
    # Same 100% agreement, different evidence: 3 reads must NOT claim certainty.
    few, _ = C([9, 9, 9], [1.0] * 3, 9)
    many, _ = C([9] * 60, [1.0] * 60, 9)
    assert few == 80 and many >= 96, (few, many)
    assert few < many


def test_weighted_by_recognition_confidence():
    # 6 shaky reads of 14 vs 4 confident reads of 47 -> 47 is what we label,
    # and the % must reflect the confident evidence, not the head count.
    nums = [14] * 6 + [47] * 4
    confs = [0.35] * 6 + [0.95] * 4
    assert app.TrackMerger.resolve_number(nums, confs=confs) == 47
    pct, n = C(nums, confs, 47)
    assert n == 10 and 45 <= pct <= 70, (pct, n)     # contested -> mid band


def test_contested_track_is_not_confident():
    pct, _ = C([9] * 10 + [6] * 10, [1.0] * 20, 9)
    assert 45 <= pct <= 55, pct                       # a coin flip reads as one


def test_no_number_no_percent():
    assert C([], [], None) == (None, 0)
    assert C([None, None], [], None) == (None, 0)
    assert C([9, 9], [1.0, 1.0], None)[0] is None     # nothing chosen -> no claim


def test_missing_confs_fall_back_to_frequency():
    pct, n = C([9] * 20, None, 9)
    assert n == 20 and pct >= 90


def test_bands_cover_every_percent():
    assert [app.conf_band(p) for p in (100, 90, 89, 75, 74, 60, 59, 0)] == [
        "90–100%", "90–100%", "75–90%", "75–90%", "60–75%", "60–75%", "<60%", "<60%"]


if __name__ == "__main__":
    test_unanimous_shrinks_with_sample_size()
    test_weighted_by_recognition_confidence()
    test_contested_track_is_not_confident()
    test_no_number_no_percent()
    test_missing_confs_fall_back_to_frequency()
    test_bands_cover_every_percent()
    print("clip_confidence OK")
