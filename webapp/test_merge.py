"""Self-check for TrackMerger's number-less absorption guard.

A track that never earned a confident number of its own must NOT be stitched
into a numbered identity by spatial proximity — it stays its own (Unknown)
group. Same-number tracks still merge; two number-less tracks still merge.
Run: python3 test_merge.py
"""
import app


def _trk(team, nums, t0, t1, pos0, pos1):
    span = max(t1 - t0, 0.1)
    return {
        "player_team": [team] * 8,
        "player_number": list(nums),
        "player_name": [],
        "frames": [],
        "first_seen": t0,
        "last_seen": t1,
        "first_position": pos0,
        "last_position": pos1,
        "time_on_ice": span,
        "total_distance": span * 4,  # ~14 km/h, never "stationary"
    }


def _merge(data):
    _, mapping = app.TrackMerger(fps=60).merge(data)
    # number of distinct final identities
    return len(set(mapping.values())), mapping


def test_numberless_not_absorbed():
    # A: strong #4 ; B: adjacent but no number at all -> must stay separate
    data = {
        1: _trk("Home", [4] * 10, 0.0, 5.0, (100, 100), (100, 100)),
        2: _trk("Home", [], 5.2, 8.0, (108, 104), (108, 104)),
    }
    n, mapping = _merge(data)
    assert n == 2, f"number-less track was absorbed: {mapping}"


def test_same_number_still_merges():
    # Two fragments both reading #4 -> one identity (regression guard)
    data = {
        1: _trk("Home", [4] * 10, 0.0, 5.0, (100, 100), (100, 100)),
        2: _trk("Home", [4] * 10, 5.2, 8.0, (108, 104), (108, 104)),
    }
    n, _ = _merge(data)
    assert n == 1


def test_two_numberless_merge_but_stay_unknown():
    # Two adjacent number-less fragments may stitch (both Unknown) ...
    data = {
        1: _trk("Home", [], 0.0, 5.0, (100, 100), (100, 100)),
        2: _trk("Home", [], 5.2, 8.0, (108, 104), (108, 104)),
    }
    merged, mapping = app.TrackMerger(fps=60).merge(data)
    canon = set(mapping.values())
    assert len(canon) == 1
    # ... and the merged identity carries no number
    g = merged[next(iter(canon))]
    assert app.TrackMerger.resolve_number(g["player_number"]) is None


if __name__ == "__main__":
    test_numberless_not_absorbed()
    test_same_number_still_merges()
    test_two_numberless_merge_but_stay_unknown()
    print("track merge OK")
