"""Self-check for dedupe_by_number: the 'one player per number on ice' guard
(Fix 1 physical overlap) + weak-number gate (Fix 2). Run: python3 test_dedupe.py"""
import app


def _c(tid, t, toi, votes, share, fixed=False):
    return {"track_id": tid, "first_seen": t, "toi": toi,
            "num_votes": votes, "num_share": share, "fixed": fixed}


def test_overlap_rejected():
    # Two strong #9 clips overlapping in time -> only the stronger survives.
    clips = [_c(1, 0, 10, 50, 1.0), _c(2, 2, 8, 40, 0.9)]
    kept, rej = app.dedupe_by_number(clips)
    assert len(kept) == 1 and kept[0]["track_id"] == 1, kept
    assert len(rej) == 1 and rej[0]["track_id"] == 2


def test_sequential_kept():
    # Strong, non-overlapping clips are all real shifts -> all kept.
    clips = [_c(1, 0, 10, 50, 1.0), _c(2, 12, 8, 40, 0.9), _c(3, 30, 5, 20, 0.8)]
    kept, rej = app.dedupe_by_number(clips)
    assert len(kept) == 3 and not rej
    assert [c["first_seen"] for c in kept] == [0, 12, 30]   # sorted by time


def test_lone_weak_kept():
    # A single weak clip (few reads) is kept — likely the real player, hard to read.
    kept, rej = app.dedupe_by_number([_c(1, 0, 5, 2, 0.4)])
    assert len(kept) == 1 and not rej


def test_weak_rejected_after_strong():
    # Once a strong clip is kept, a later weak non-overlapping one is rejected.
    clips = [_c(1, 0, 10, 50, 1.0), _c(2, 20, 5, 2, 0.4)]
    kept, rej = app.dedupe_by_number(clips)
    assert [c["track_id"] for c in kept] == [1]
    assert [c["track_id"] for c in rej] == [2]


def test_hand_entered_identity_is_never_overruled():
    # The operator typed this one in during the live session. A longer, better-read
    # clip of the same number overlapping it does NOT get to throw it out — that
    # would turn the answer the operator gave back into a guess.
    clips = [_c(1, 0, 30, 90, 1.0), _c(2, 5, 4, 1, 0.2, fixed=True)]
    kept, rej = app.dedupe_by_number(clips)
    assert [c["track_id"] for c in kept] == [2], kept     # the typed-in one stays
    assert [c["track_id"] for c in rej] == [1], rej       # the system's guess yields
    # and a weak auto clip still loses to the fixed one that claimed the time
    clips = [_c(3, 0, 10, 40, 0.9, fixed=True), _c(4, 2, 8, 30, 0.9)]
    kept, rej = app.dedupe_by_number(clips)
    assert [c["track_id"] for c in kept] == [3] and [c["track_id"] for c in rej] == [4]


if __name__ == "__main__":
    test_overlap_rejected()
    test_sequential_kept()
    test_lone_weak_kept()
    test_weak_rejected_after_strong()
    test_hand_entered_identity_is_never_overruled()
    print("dedupe_by_number OK")
