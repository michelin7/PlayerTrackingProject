"""Self-check for the live mid-track cut: when the tracker jumps onto another
player, the operator cuts the track and each side must come out as its own clip,
with its own identity and its own share of the number reads.
Run: python3 test_livecut.py"""
import json, os, tempfile, types
from pathlib import Path
from threading import Lock
from live import LiveSession


def _sess(tmp, fps=30.0):
    s = LiveSession.__new__(LiveSession)
    s.lock = Lock()
    s.fps, s.frame_idx, s.status = fps, 0, "playing"
    s.out_dir = Path(tmp)
    s.videos_dir = s.out_dir / "player_videos"
    s.videos_dir.mkdir(parents=True, exist_ok=True)
    s.track_identity, s.track_votes, s.track_iv = {}, {}, {}
    s.track_cuts, s._written, s.overrides = {}, {}, {}
    s.ignored, s.agree = set(), {}
    s.pid_key, s.track_from, s.frozen = {}, {}, {}
    s.pending, s.track_crop, s._paused_to_ask = {}, {}, False
    s.events, s.log_path = 0, os.devnull
    s.active_tracks = []
    s.identifier = types.SimpleNamespace(
        Home={"10": "Ivanov"}, Away={"22": "Petrov"},
        _cache={}, _est_number={}, _team_votes={}, _num_switch_counts={},
        _est_team={}, _recent_team={})
    return s


def _track(s, pid, team, num, name, seconds, nums):
    s.track_identity[pid] = (team, num, name)
    s.track_iv[pid] = [[0.0, seconds]]
    s.track_votes[pid] = {"nums": nums, "confs": [0.9] * len(nums)}
    s._written[pid] = int(seconds * s.fps)


def test_cut_splits_the_ice_time_between_two_players():
    # The tracker jumped from one player to another: the time before the cut
    # stays with the player it was (banked), and the time after belongs to
    # whoever the operator names next — not to both.
    with tempfile.TemporaryDirectory() as tmp:
        s = _sess(tmp, fps=10.0)
        s.presence, s.names, s.on_ice, s.number_pids = {}, {}, {}, {}
        s.shift_gap, s.track_numbered = 1.5, set()
        s.identifier.Home = {"10": "Ivanov", "16": "Josephs"}
        _track(s, 4, "Home", 10, "Ivanov", 20.0, [10] * 60)   # 20s of footage
        s.frame_idx = 200                                     # t = 20.0s
        s.now_t = s.frame_idx / s.fps
        s.set_override(4, "10", "Home")                       # confirmed as #10
        assert abs(s.shifts()[0]["toi"] - 20.0) < 0.2, s.shifts()

        assert s.cut_track(4) is True
        s.track_iv[4] = [[0.0, 32.0]]                         # it keeps running
        s.frame_idx = 320
        s.now_t = s.frame_idx / s.fps
        s.set_override(4, "16", "Home")                       # the new player
        rows = {(r["team"], r["number"]): r for r in s.shifts()}
        assert abs(rows[("Home", "10")]["toi"] - 20.0) < 0.3, rows   # banked, unchanged
        assert abs(rows[("Home", "16")]["toi"] - 12.0) < 0.3, rows   # only after the cut
    print("test_cut_splits_the_ice_time_between_two_players OK")


def test_cut_needs_a_real_segment_on_each_side():
    with tempfile.TemporaryDirectory() as tmp:
        s = _sess(tmp)
        _track(s, 1, "Home", 10, "Ivanov", 0.5, [10] * 15)   # half a second so far
        assert s.cut_track(1) is False
        assert not s.track_cuts.get(1)


def test_cut_records_the_part_before_and_clears_the_track():
    with tempfile.TemporaryDirectory() as tmp:
        s = _sess(tmp)
        _track(s, 1, "Home", 10, "Ivanov", 8.0, [10] * 40)
        s.overrides[1] = {"team": "Home", "number": "10"}
        s.identifier._cache[1] = {"number": 10}
        s.identifier._team_votes[1] = [9.0, 1.0]
        s.frame_idx = 240
        s.now_t = s.frame_idx / s.fps
        assert s.cut_track(1) is True
        cut = s.track_cuts[1][0]
        assert cut["clip_t"] == 8.0 and cut["video_t"] == 8.0
        assert cut["identity"] == ("Home", "10", "Ivanov") and cut["votes"] == 40
        # the new part must start clean, or the old number is voted straight back
        assert 1 not in s.overrides and s.track_identity[1] == ("Unknown", None, "")
        assert s.track_votes[1] == {"nums": [], "confs": []}
        assert 1 not in s.identifier._cache and 1 not in s.identifier._team_votes


def test_segments_split_votes_presence_and_identity():
    with tempfile.TemporaryDirectory() as tmp:
        s = _sess(tmp)
        # 10s of #10, cut at 6s, then 4s of #22 on the other team
        _track(s, 1, "Home", 10, "Ivanov", 10.0, [10] * 30)
        s.frame_idx = int(6 * s.fps)
        s.now_t = s.frame_idx / s.fps
        s._written[1] = int(6 * s.fps)
        s.cut_track(1)
        s.track_votes[1] = {"nums": [10] * 30 + [22] * 12, "confs": [0.9] * 42}
        s.track_identity[1] = ("Away", 22, "Petrov")
        s.track_iv[1] = [[0.0, 10.0]]
        # a real file to cut, written by the shared helper
        import numpy as np, cv2
        src = s.videos_dir / "player_1.mp4"
        w = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"avc1"), s.fps, (64, 48))
        for _ in range(int(10 * s.fps)):
            w.write(np.zeros((48, 64, 3), np.uint8))
        w.release()

        segs = list(s._segments())
        assert len(segs) == 2, segs
        (p0, id0, a), (p1, id1, b) = segs
        assert id0 == ("Home", 10, "Ivanov") and id1 == ("Away", 22, "Petrov")
        assert a["track_id"] != b["track_id"] and a["file"] != b["file"]
        assert (s.videos_dir / a["file"]).exists() and (s.videos_dir / b["file"]).exists()
        assert a["nums"] == [10] * 30 and b["nums"] == [22] * 12   # votes split at the cut
        assert a["iv"] == [[0.0, 6.0]] and b["iv"] == [[6.0, 10.0]]  # presence too


def test_uncut_track_is_passed_straight_through():
    with tempfile.TemporaryDirectory() as tmp:
        s = _sess(tmp)
        _track(s, 7, "Home", 10, "Ivanov", 5.0, [10] * 20)
        (s.videos_dir / "player_7.mp4").write_bytes(b"x")
        (pid, ident, seg), = list(s._segments())
        assert pid == 7 and seg["track_id"] == 7 and seg["file"] == "player_7.mp4"
        assert seg["iv"] == [[0.0, 5.0]] and seg["nums"] == [10] * 20


if __name__ == "__main__":
    test_cut_splits_the_ice_time_between_two_players()
    test_cut_needs_a_real_segment_on_each_side()
    test_cut_records_the_part_before_and_clears_the_track()
    test_segments_split_votes_presence_and_identity()
    test_uncut_track_is_passed_straight_through()
    print("live track cut OK")
