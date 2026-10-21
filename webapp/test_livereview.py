"""Self-check for LiveSession._build_review: a live session must resolve clip
identity the same way Automate does — whole-track weighted voting, a confidence
per clip, and the one-player-per-number constraint.
Run: python3 test_livereview.py"""
import json, os, tempfile, types
from pathlib import Path
from threading import Lock
from live import LiveSession


def _sess(tmp):
    s = LiveSession.__new__(LiveSession)
    s.lock = Lock()
    s.fps, s.frame_idx, s.status = 30.0, 900, "done"
    s.out_dir = Path(tmp)
    s.videos_dir = s.out_dir / "player_videos"
    s.videos_dir.mkdir(parents=True, exist_ok=True)
    s.presence, s.on_ice, s.names = {}, {}, {}
    s.overrides, s.track_identity, s.track_votes, s.track_iv = {}, {}, {}, {}
    s.lines = {}
    s.track_cuts, s._written = {}, {}
    s.track_numbered, s.number_pids = set(), {}
    s.ignored, s.agree = set(), {}
    s.pid_key, s.track_from, s.frozen = {}, {}, {}
    s.video_path = ""
    s.team_hex = {"Home": "#29B6D8", "Away": "#FF6300"}
    s.boxes, s.frame_w, s.frame_h = [], 1280, 720
    s.pending, s.track_crop, s._paused_to_ask = {}, {}, False
    s.events, s.log_path = 0, os.devnull
    s.identifier = types.SimpleNamespace(Home={"10": "Ivanov"}, Away={"22": "Petrov"})
    s._finish_writers = lambda: None
    s.review_ready = False
    return s


def _track(s, pid, team, num, name, start, end, nums, confs):
    (s.videos_dir / f"player_{pid}.mp4").write_bytes(b"x")   # must exist and be non-empty
    s.track_identity[pid] = (team, num, name)
    s.track_iv[pid] = [[start, end]]
    s.track_votes[pid] = {"nums": nums, "confs": confs}


def _manifest(s):
    s._finalize = lambda m: None          # the finalize step has its own test
    s._build_review()
    return json.loads((s.out_dir / "review_manifest.json").read_text(encoding="utf-8"))


def test_number_votes_are_text_like_the_real_session():
    # A live session records numbers as TEXT — the model's read is formatted as
    # one and an override stores one. Feeding ints here (as this file's other
    # fixtures do) hid a crash in resolve_number that killed the whole save and
    # left the page waiting on a session that was already dead.
    with tempfile.TemporaryDirectory() as tmp:
        s = _sess(tmp)
        _track(s, 1, "Home", "10", "Ivanov", 0, 20, ["10"] * 30, [0.9] * 30)
        _track(s, 2, "Away", "22", "Petrov", 0, 20, ["4"] * 3 + ["44"] * 9, [0.8] * 12)
        m = _manifest(s)
        got = sorted((c["auto_number"], c["num_share"])
                     for g in m["groups"] for c in g["clips"])
        assert got == [("10", 1.0), ("44", 0.75)], got     # vote AND share work
    print("test_number_votes_are_text_like_the_real_session OK")


def test_vote_beats_last_frame_read():
    # The last frame said 22, but the track's own reads overwhelmingly say 10:
    # Automate would label this 10, and so must live.
    with tempfile.TemporaryDirectory() as tmp:
        s = _sess(tmp)
        _track(s, 1, "Home", 22, "", 0, 20, [10] * 30 + [22], [0.9] * 30 + [0.4])
        m = _manifest(s)
        clip = m["groups"][0]["clips"][0]
        assert clip["auto_number"] == "10", clip
        assert clip["auto_name"] == "Ivanov", clip      # name re-derived from the roster
        assert clip["num_conf"] >= 90, clip             # 31 reads, nearly unanimous


def test_ignored_track_produces_no_clip():
    # An ignored track (the goalie) must not reach review at all — otherwise the
    # operator throws it out live and meets it again on the review page.
    with tempfile.TemporaryDirectory() as tmp:
        s = _sess(tmp)
        _track(s, 1, "Home", 10, "Ivanov", 0, 20, [10] * 30, [0.9] * 30)
        _track(s, 2, "Away", 22, "Petrov", 0, 20, [22] * 30, [0.9] * 30)
        s.ignored.add(2)
        m = _manifest(s)
        nums = [c["auto_number"] for g in m["groups"] for c in g["clips"]]
        assert nums == ["10"], nums
        assert not m["unidentified"], m["unidentified"]
    print("test_ignored_track_produces_no_clip OK")


def test_hand_entered_identity_survives_the_whole_build():
    # End to end: the operator said track 2 is Home #10 during a stretch where
    # track 1 is also voting #10, loudly and for longer. The typed-in identity
    # must come out of the build untouched; it is the system's guess that yields.
    with tempfile.TemporaryDirectory() as tmp:
        s = _sess(tmp)
        _track(s, 1, "Home", 10, "Ivanov", 0, 30, [10] * 90, [0.95] * 90)
        _track(s, 2, "Home", 10, "Ivanov", 5, 12, [10] * 6, [0.5] * 6)
        s.overrides[2] = {"number": "10", "team": "Home"}
        m = _manifest(s)
        kept = [c for g in m["groups"] for c in g["clips"]]
        assert [c["track_id"] for c in kept] == [2], kept
        assert kept[0]["fixed"] and kept[0]["auto_number"] == "10", kept[0]
        ejected = [c for u in m["unidentified"] for c in u["clips"]]
        assert [c["track_id"] for c in ejected] == [1], ejected
    print("test_hand_entered_identity_survives_the_whole_build OK")


def test_confidence_reported_and_low_when_contested():
    with tempfile.TemporaryDirectory() as tmp:
        s = _sess(tmp)
        _track(s, 1, "Home", 10, "Ivanov", 0, 20, [10] * 8 + [22] * 7, [0.5] * 15)
        clip = _manifest(s)["groups"][0]["clips"][0]
        assert clip["num_conf"] is not None and clip["num_conf"] < 70, clip
        assert clip["num_votes"] == 15


def test_overlapping_same_number_is_ejected():
    # Two tracks claim Home #10 at the SAME time — physically impossible, so the
    # weaker one goes to review as unidentified instead of onto Ivanov's card.
    with tempfile.TemporaryDirectory() as tmp:
        s = _sess(tmp)
        _track(s, 1, "Home", 10, "Ivanov", 0, 30, [10] * 60, [0.9] * 60)
        _track(s, 2, "Home", 10, "Ivanov", 5, 25, [10] * 5, [0.4] * 5)
        m = _manifest(s)
        assert [c["track_id"] for c in m["groups"][0]["clips"]] == [1], m["groups"]
        ejected = [c for u in m["unidentified"] for c in u["clips"]]
        assert [c["track_id"] for c in ejected] == [2]
        assert ejected[0]["auto_number"] is None       # reviewer assigns it


def test_toi_is_union_not_sum():
    with tempfile.TemporaryDirectory() as tmp:
        s = _sess(tmp)
        _track(s, 1, "Home", 10, "Ivanov", 0, 20, [10] * 40, [0.9] * 40)
        _track(s, 2, "Home", 10, "Ivanov", 30, 40, [10] * 20, [0.9] * 20)
        g = _manifest(s)["groups"][0]
        assert g["total_toi"] == 30.0, g["total_toi"]   # 20 + 10, no double count


def test_user_override_wins_and_claims_no_confidence():
    # The reviewer answered during the session: their answer is ground truth, and
    # must NOT be reported as system confidence or the calibration would flatter.
    with tempfile.TemporaryDirectory() as tmp:
        s = _sess(tmp)
        _track(s, 1, "Home", 10, "Ivanov", 0, 20, [10] * 30, [0.9] * 30)
        s.overrides[1] = {"team": "Away", "number": "22"}
        clip = _manifest(s)["groups"][0]["clips"][0]
        assert (clip["auto_team"], str(clip["auto_number"])) == ("Away", "22")
        assert clip["auto_name"] == "Petrov"
        assert clip["num_conf"] is None, clip


def test_profile_present_for_diagnostics_bar():
    with tempfile.TemporaryDirectory() as tmp:
        s = _sess(tmp)
        _track(s, 1, "Home", 10, "Ivanov", 0, 20, [10] * 30, [0.9] * 30)
        p = _manifest(s)["profile"]
        assert p["processed_frames"] == 900 and p["clips_identified"] == 1
        assert p["clips_per_player"] == 1.0


def test_correction_relabels_the_panel_while_paused():
    # The on-screen snapshot is rebuilt only by the worker loop, which is parked
    # while the session is paused — exactly when the
    # operator uses the panel. A saved correction must still show up, or Save
    # looks dead.
    with tempfile.TemporaryDirectory() as tmp:
        s = _sess(tmp)
        s.status = "paused"                      # worker loop parked
        s.track_identity[4] = ("Unknown", None, "")
        s.active_tracks = [{"pid": 4, "team": None, "number": None, "name": None,
                            "label": "Track 4", "fixed": False}]
        s.set_override(4, "10", "Home")
        t = s.active_tracks[0]
        assert (t["label"], t["number"], t["team"], t["fixed"]) == ("#10 Ivanov", "10", "Home", True), t
        s.set_override(4, None, None, remove=True)   # "Clear fix" goes back
        t = s.active_tracks[0]
        assert (t["label"], t["fixed"]) == ("Track 4", False), t


def test_track_label_names_the_player_or_the_track():
    # The overlay must name the player when known, and otherwise give the track
    # id — that id is the handle the operator clicks to correct it.
    assert LiveSession.track_label(12, 25, "Olson") == "#25 Olson"
    assert LiveSession.track_label(12, 25, None) == "#25"
    assert LiveSession.track_label(12, 25, "Unknown") == "#25"
    assert LiveSession.track_label(12, None, "Olson") == "Track 12"
    assert LiveSession.track_label(12, None, None) == "Track 12"


def test_finalize_decisions_separate_fixes_from_unconfirmed():
    # Everything the operator had to fix must count as a CORRECTION, or a live
    # session would always report 100% auto-labeling accuracy.
    with tempfile.TemporaryDirectory() as tmp:
        s = _sess(tmp)
        calls = {}
        s.overrides = {2: {"team": "Home", "number": "10"}}
        # track 2 was set by hand; a CUT segment (id of its own, never in
        # self.overrides) must count as a fix too, not as a success
        manifest = {"groups": [{"canonical_pid": 1, "auto_team": "Home",
                                "auto_number": 10, "auto_name": "Ivanov",
                                "clips": [{"track_id": 1, "fixed": False},
                                          {"track_id": 2, "fixed": True},
                                          {"track_id": 90000201, "fixed": True}]}]}

        class _FakeApp:
            processing_jobs = {}
            @staticmethod
            def finalize_job(job_id):
                calls["job"] = job_id
                _FakeApp.processing_jobs[job_id]["status"] = "completed"

        import sys
        real, sys.modules["app"] = sys.modules.get("app"), _FakeApp
        try:
            s.job_id = "jobX"
            s._finalize(manifest)
        finally:
            if real is not None:
                sys.modules["app"] = real
            else:
                del sys.modules["app"]
        dec = json.loads((s.out_dir / "review_decisions.json").read_text(encoding="utf-8"))
        clips = dec["decisions"]["1"]["clips"]
        # Nothing is accepted on the system's word in live: a clip the operator
        # never confirmed is material for review, not a player.
        assert clips["1"]["action"] == "reject"
        assert clips["2"]["action"] == "reassign"         # operator had to fix it
        assert clips["2"]["target_number"] == 10          # onto its corrected identity
        assert clips["90000201"]["action"] == "reassign"  # a cut segment counts too
        assert calls["job"] == "jobX" and s.result_ready


def test_overlay_boxes_follow_the_cut_into_two_clips():
    """Review draws a box under the clip it belongs to. A cut track became two
    clips, so its frames must split at the cut — or the second player's boxes
    would be labelled with the first player's name."""
    with tempfile.TemporaryDirectory() as tmp:
        s = _sess(tmp)
        s.boxes = [[1.0, [[5, 10, 20, 30, 40]]], [9.0, [[5, 11, 21, 31, 41]]]]
        s.track_cuts[5] = [{"clip_t": 5.0, "video_t": 5.0, "identity": ("Home", "10", "Ivanov"),
                            "votes": 1, "fixed": True}]
        s._write_overlay()
        ov = json.loads((Path(tmp) / "overlay.json").read_text(encoding="utf-8"))
        ids = [f[1][0][0] for f in ov["frames"]]
        assert ids == [int("90000500"), int("90000501")], ids
        assert (ov["w"], ov["h"]) == (1280, 720), ov
    print("test_overlay_boxes_follow_the_cut_into_two_clips OK")


if __name__ == "__main__":
    test_number_votes_are_text_like_the_real_session()
    test_vote_beats_last_frame_read()
    test_ignored_track_produces_no_clip()
    test_hand_entered_identity_survives_the_whole_build()
    test_overlay_boxes_follow_the_cut_into_two_clips()
    test_confidence_reported_and_low_when_contested()
    test_overlapping_same_number_is_ejected()
    test_toi_is_union_not_sum()
    test_user_override_wins_and_claims_no_confidence()
    test_profile_present_for_diagnostics_bar()
    test_correction_relabels_the_panel_while_paused()
    test_track_label_names_the_player_or_the_track()
    test_finalize_decisions_separate_fixes_from_unconfirmed()
    print("live review build OK")
