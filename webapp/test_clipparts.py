"""Self-check for clip_parts: manual review cuts. The keys must match clipParts()
in the review UI, or finalize would look up decisions it never wrote.
Run: python3 test_clipparts.py"""
import app

CLIP = {"track_id": 42, "file": "player_42.mp4", "toi": 10.0, "first_seen": 100.0}


def test_uncut_keeps_bare_key():
    # No cut -> the clip itself under its bare track_id, so decisions saved
    # before any cut still resolve.
    parts = app.clip_parts(CLIP, {})
    assert parts == [("42", CLIP)], parts


def test_one_cut_splits_in_two():
    parts = app.clip_parts(CLIP, {"42": [4.0]})
    assert [k for k, _ in parts] == ["42#0", "42#1"]
    a, b = (c for _, c in parts)
    assert (a["start"], a["end"], a["toi"], a["first_seen"]) == (0.0, 4.0, 4.0, 100.0)
    assert (b["start"], b["end"], b["toi"], b["first_seen"]) == (4.0, 10.0, 6.0, 104.0)
    assert a["toi"] + b["toi"] == CLIP["toi"]       # no time invented or lost


def test_two_cuts_sorted():
    parts = app.clip_parts(CLIP, {"42": [7.5, 2.0]})   # unsorted input
    assert [round(c["toi"], 1) for _, c in parts] == [2.0, 5.5, 2.5]


def test_out_of_range_cuts_ignored():
    # A cut at/past the clip edge would make an empty part.
    assert len(app.clip_parts(CLIP, {"42": [0, 10.0, 99]})) == 1


def test_part_with_own_file_takes_whole_file():
    # Once a cut is written out, the part IS that file: no frame range, and
    # finalize must copy all of it.
    files = {"42": ["player_42__p0.mp4", "player_42__p1.mp4"]}
    parts = app.clip_parts(CLIP, {"42": [4.0]}, files)
    assert [c["file"] for _, c in parts] == files["42"]
    assert all("start" not in c and "end" not in c for _, c in parts)
    # TOI/first_seen still come from the offsets — cutting moves no game clock
    assert [c["toi"] for _, c in parts] == [4.0, 6.0]
    assert [c["first_seen"] for _, c in parts] == [100.0, 104.0]


def test_missing_file_falls_back_to_range():
    # Only the first part was written out -> the second stays a range.
    parts = app.clip_parts(CLIP, {"42": [4.0]}, {"42": ["player_42__p0.mp4"]})
    (_, a), (_, b) = parts
    assert a["file"] == "player_42__p0.mp4" and "start" not in a
    assert b["file"] == CLIP["file"] and (b["start"], b["end"]) == (4.0, 10.0)


def test_other_track_untouched():
    assert app.clip_parts(CLIP, {"7": [5.0]}) == [("42", CLIP)]


if __name__ == "__main__":
    test_uncut_keeps_bare_key()
    test_one_cut_splits_in_two()
    test_two_cuts_sorted()
    test_out_of_range_cuts_ignored()
    test_part_with_own_file_takes_whole_file()
    test_missing_file_falls_back_to_range()
    test_other_track_untouched()
    print("clip_parts OK")
