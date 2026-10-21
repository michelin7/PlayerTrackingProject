"""Self-check for LiveSession shift accumulation (no models, no video)."""
import os
from live import LiveSession


def _sess(fps=10.0):
    import types
    s = LiveSession.__new__(LiveSession)
    from threading import Lock
    s.lock = Lock()
    s.fps = fps
    s.frame_idx = 0
    s.now_t = s.frame_idx / s.fps
    s.presence = {}
    s.on_ice = {}
    s.names = {}
    s.shift_gap = 1.5
    s.overrides = {}
    s.track_iv = {}
    s.track_numbered = set()
    s.number_pids = {}
    s.active_tracks = []
    s.track_identity = {}
    s.pid_key, s.track_from, s.frozen = {}, {}, {}
    s.ignored = set()
    s.agree = {}
    s.pending, s.asked, s.track_crop = {}, set(), {}
    s.track_cuts = {}
    s.track_last_seen = {}
    s._paused_to_ask = False
    s.status = 'playing'
    s.forget_frames = 25          # 2.5s at 10fps, as the tracker config says
    s._resume = __import__('threading').Event(); s._resume.set()
    s.events = 0
    s.log_path = os.devnull          # corrections are logged; nowhere to keep it here
    # roster: names are looked up from here by (team, number)
    s.identifier = types.SimpleNamespace(Home={"10": "Ivanov"}, Away={"22": "Petrov"})
    return s


def _step(s, frame, id_results):
    """One processed frame: the real lifetime bookkeeping, then the shift update."""
    s.frame_idx = frame
    s.now_t = s.frame_idx / s.fps
    s._track_lifetime([(pid, 0, 0, 10, 20) for pid in id_results], id_results)
    s._update_shifts(id_results)


def _confirm(s, pid, number, team):
    """What the operator does in the panel — the only thing that makes ice time."""
    s.set_override(pid, number, team)


def test_shifts():
    s = _sess(fps=10.0)  # 10 fps -> 1 frame = 0.1s
    # #10 Home present frames 1..16 (t=0.1..1.6) -> one shift ~1.5s. Ice time is
    # only counted for a track the operator confirmed, so confirm it first.
    _confirm(s, 1, "10", "Home")
    for f in range(1, 17):
        _step(s, f, {1: ("Ivanov", "10", "Home")})
    sh = {(x["team"], x["number"]): x for x in s.shifts()}
    row = sh[("Home", "10")]
    assert row["shifts"] == 1, row
    assert abs(row["toi"] - 1.5) < 0.11, row
    assert row["name"] == "Ivanov"
    assert row["on"] is True                      # present on last frame

    # off-screen gap of 5s (< 30s BRIDGE): same shift, but the gap is a BRIDGED
    # segment and still counts as ice time.
    for f in range(60, 71):                        # t=6.0..7.0, gap 4.4s from t=1.6
        _step(s, f, {1: ("Ivanov", "10", "Home")})
    row = sh_get(s, "Home", "10")
    assert row["shifts"] == 1, ("5s gap must NOT split", row)
    assert len(row["bridged"]) == 1, ("5s gap must be bridged", row)
    assert abs(row["toi"] - (1.5 + 1.0 + 4.4)) < 0.15, ("bridged gap counts as TOI", row)

    # off-screen gap of 33s (>= 30s): a new shift; the gap is NOT counted.
    for f in range(400, 411):                      # t=40.0..41.0, gap 33s from t=7.0
        _step(s, f, {1: ("Ivanov", "10", "Home")})
    row = sh_get(s, "Home", "10")
    assert row["shifts"] == 2, (">=30s gap must split", row)
    assert len(row["bridged"]) == 1, ("only the short gap is bridged", row)
    assert abs(row["toi"] - (1.5 + 1.0 + 4.4 + 1.0)) < 0.2, ("33s gap NOT counted", row)

    # What the SYSTEM reads is a guess and never becomes ice time on its own —
    # that is where the phantom players came from. Confirming the track brings
    # its whole life in at once, including the seconds before the operator got
    # to it.
    s2 = _sess(fps=10.0)
    for f in range(1, 31):                         # 3 seconds, unconfirmed
        _step(s2, f, {9: ("Olson", "25", "Away")})
    assert not s2.shifts(), s2.shifts()
    _confirm(s2, 9, "25", "Away")
    row = sh_get(s2, "Away", "25")
    assert abs(row["toi"] - 2.9) < 0.11, row       # the whole track, not the rest of it

    # a number-less / unknown track never creates a shift
    _step(s, 412, {2: ("Unknown", None, "Away"), 3: ("X", "7", "Unknown")})
    keys = {(x["team"], x["number"]) for x in s.shifts()}
    assert ("Away", None) not in keys and ("Unknown", "7") not in keys, keys

    # a confirmed player on screen is marked on-ice, one who left is not
    _confirm(s, 2, "22", "Away")
    for f in range(413, 430):
        _step(s, f, {2: ("Petrov", "22", "Away")})
    assert sh_get(s, "Home", "10")["on"] is False
    assert sh_get(s, "Away", "22")["on"] is True
    print("test_live OK")


def sh_get(s, team, num):
    for x in s.shifts():
        if x["team"] == team and x["number"] == num:
            return x
    raise KeyError((team, num))


def test_ignored_track_gives_its_time_back():
    # The goalie: the tracker holds him all period and the identifier keeps
    # handing him somebody's number. Ignoring the track must take the player he
    # invented out of the list — and leave alone a player other tracks also feed.
    s = _sess(fps=10.0)
    _confirm(s, 50, "10", "Home")                      # the goalie track, confirmed
    _confirm(s, 51, "22", "Away")                      # a real skater
    _confirm(s, 52, "22", "Away")                      # a second track for #22
    for f in range(1, 20):
        _step(s, f, {50: ("Ivanov", "10", "Home"), 51: ("Petrov", "22", "Away")})
    for f in range(20, 40):
        _step(s, f, {52: ("Petrov", "22", "Away")})
    assert sh_get(s, "Home", "10")["toi"] > 0

    s.ignore_track(50)
    assert 50 in s.ignored
    assert not [x for x in s.shifts() if x["number"] == "10"], s.shifts()
    # #22 is fed by another track too, so his time stands
    assert sh_get(s, "Away", "22")["toi"] > 0
    assert s.number_pids[("Away", "22")] == {51, 52}

    s.ignore_track(51, on=False)                       # taking one back is allowed
    assert 51 not in s.ignored
    print("test_ignored_track_gives_its_time_back OK")


def test_hand_entered_identity_scores_the_system():
    # What the operator types in is ground truth: it is never changed, and the
    # system's own read on that track becomes free evidence of how often the
    # identifier would have been right.
    s = _sess()
    ov = {"number": "10", "team": "Home"}
    s._score(1, "10", "Home", ov)        # system agrees
    s._score(1, "47", "Home", ov)        # misread the number
    s._score(1, "10", "Away", ov)        # right number, wrong team
    s._score(1, None, "Home", ov)        # no read at all -> not scored
    assert s.agree[1] == [1, 3], s.agree
    assert s.check_stats() == {"reads": 3, "agree": 1, "pct": 33, "tracks": 1}

    # a track with no hand-entered number scores nothing
    s._score(2, "22", "Away", {"number": None, "team": "Away"})
    assert 2 not in s.agree
    assert _sess().check_stats()["pct"] is None
    print("test_hand_entered_identity_scores_the_system OK")


def test_lost_track_is_asked_about():
    # A track nobody ruled on leaves the camera: it must become a question, or
    # its clip lands in the result wearing whatever the system last read.
    s = _sess(fps=10.0)
    s.track_identity[7] = ("Away", "86", "")
    s.track_iv[7] = [[0.0, 4.0]]                  # 4s on ice
    s.track_last_seen[7] = 40
    s.frame_idx = 40 + s.forget_frames + 11      # forgotten, plus a second of grace
    s.now_t = s.frame_idx / s.fps
    s._check_lost()
    assert 7 in s.pending and s.pending[7]["toi"] == 4.0, s.pending
    assert s.status == "paused" and s._paused_to_ask    # asked, not just listed

    # a blip is not a question — and is never asked about again
    s.track_identity[8] = ("Away", "99", "")
    s.track_iv[8] = [[0.0, 0.5]]
    s.track_last_seen[8] = 40
    s._check_lost()
    assert 8 not in s.pending and 8 in s.asked, (s.pending, s.asked)

    # answering the last question lets the video run on by itself
    s.set_override(7, "86", "Away")
    assert not s.pending and not s._paused_to_ask
    assert s.status == "playing", s.status

    # a track the operator already ruled on is never asked about
    s2 = _sess(fps=10.0)
    s2.track_identity[5] = ("Home", "10", "Ivanov")
    s2.track_iv[5] = [[0.0, 9.0]]
    s2.track_last_seen[5] = 10
    s2.frame_idx = 10 + s2.forget_frames + 11
    s2.now_t = s2.frame_idx / s2.fps
    s2.ignored.add(5)
    s2._check_lost()
    assert not s2.pending, s2.pending
    print("test_lost_track_is_asked_about OK")


def test_offscreen_tracks_are_listed_until_forgotten():
    # A track out of frame that the tracker still holds can come back under the
    # same id — the operator should see it coming, separately from who is really
    # in frame. Once the tracker has let go, it must leave the list.
    s = _sess(fps=10.0)
    s.active_tracks = [{"pid": 1}]
    s.track_identity = {1: ("Home", "10", "Ivanov"), 2: ("Away", "22", "Petrov"),
                        3: ("Away", "7", "")}
    s.frame_idx = 100
    s.now_t = s.frame_idx / s.fps
    s.track_last_seen = {1: 100, 2: 90, 3: 100 - s.forget_frames - 1}
    soon = s.offscreen_tracks()
    assert [t["pid"] for t in soon] == [2], soon      # 1 is in frame, 3 is forgotten
    assert soon[0]["label"] == "#22 Petrov" and soon[0]["gone"] == 1.0, soon[0]

    # an ignored track is not coming back to anything
    s.ignored.add(2)
    assert not s.offscreen_tracks()
    print("test_offscreen_tracks_are_listed_until_forgotten OK")


def test_edit_applied():
    s = _sess()
    s._apply_override(11, "28", "Away")
    s._apply_override(41, "4", "Away")
    assert set(s.overrides) == {11, 41}
    # every manual edit is logged as ground truth (that log IS the point of the mode)
    assert s.events == 2

    # edit an applied correction -> value changes
    s.set_override(11, "72", "Home")
    assert s.overrides[11] == {"number": "72", "team": "Home"}

    # remove an applied correction
    s.set_override(41, None, None, remove=True)
    assert 41 not in s.overrides
    print("test_edit_applied OK")


def test_unidentified_track_keeps_its_ice_time():
    # The operator names a track the model never read a number on (a chip that
    # says "Track 57"). Its presence was recorded against the TRACK, so naming
    # it must fold that ice time into the player — otherwise the correction
    # changes the label and silently throws the minutes away.
    s = _sess()
    s.identifier.Away = {"34": "Ilyenko"}
    s.track_iv[57] = [[0.0, 5.0], [8.0, 10.0]]      # 7s of ice time, unidentified
    assert not s.shifts()                            # nothing attributed yet

    s.set_override(57, "34", "Away")

    row = sh_get(s, "Away", "34")
    assert row["name"] == "Ilyenko"
    # 5s + 2s on screen + a 3s (<30s) gap bridged as ice time = 10s counted
    assert abs(row["toi"] - 10.0) < 0.2, ("lost track's ice time is counted", row)
    assert row["bridged"] == [[5.0, 8.0]], row
    # a second track confirmed as the same player adds ITS life too (one player,
    # several track ids over a period) — and the gap between them is bridged
    s.track_iv[58] = [[20.0, 30.0]]
    s.set_override(58, "34", "Away")
    row = sh_get(s, "Away", "34")
    assert abs(row["toi"] - 30.0) < 0.2, row
    assert s.number_pids[("Away", "34")] == {57, 58}

    # ...and taking that confirmation back removes exactly its share
    s.set_override(58, None, None, remove=True)
    assert abs(sh_get(s, "Away", "34")["toi"] - 10.0) < 0.2
    print("test_unidentified_track_keeps_its_ice_time OK")


def test_removed_player_stays_removed():
    # Dropping a player must hold: nothing the system reads can put him back,
    # only the operator confirming a track as him.
    s = _sess(fps=10.0)
    _confirm(s, 3, "6", "Away")
    for f in range(1, 20):
        _step(s, f, {3: ("Ryabov", "6", "Away")})
    assert sh_get(s, "Away", "6")["toi"] > 0

    s.edit_shift("Away", "6", "remove")
    assert not [x for x in s.shifts() if x["number"] == "6"], s.shifts()

    for f in range(30, 60):                       # the system keeps reading #6
        _step(s, f, {4: ("Ryabov", "6", "Away")})
    assert not [x for x in s.shifts() if x["number"] == "6"], "came back on a guess"

    # ...until the operator says a track really IS #6
    _confirm(s, 9, "6", "Away")
    for f in range(70, 90):
        _step(s, f, {9: ("Ryabov", "6", "Away")})
    assert sh_get(s, "Away", "6")["toi"] > 0
    print("test_removed_player_stays_removed OK")


def test_build_shifts_shared():
    # the shared builder (used by BOTH live and Automate-mode review)
    from live import build_shifts
    pres = {("Home", "10"): [[0.0, 5.0], [8.0, 10.0], [50.0, 52.0]]}
    rows = build_shifts(pres, {("Home", "10"): "Ivanov"})
    r = rows[0]
    assert r["name"] == "Ivanov"
    assert r["shifts"] == 2, r                     # 3s gap bridged, 40s gap splits
    assert r["bridged"] == [[5.0, 8.0]], r
    assert abs(r["toi"] - (5 + 3 + 2 + 2)) < 0.05, r   # 12s (bridged 3s counted)
    print("test_build_shifts_shared OK")


if __name__ == "__main__":
    test_shifts()
    test_build_shifts_shared()
    test_removed_player_stays_removed()
    test_ignored_track_gives_its_time_back()
    test_hand_entered_identity_scores_the_system()
    test_lost_track_is_asked_about()
    test_offscreen_tracks_are_listed_until_forgotten()
    test_edit_applied()
    test_unidentified_track_keeps_its_ice_time()
