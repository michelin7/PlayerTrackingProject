"""Live and Automate must run the SAME tracker and the same identification
thresholds. They drifted apart silently once already, so the agreement is
asserted here instead of being trusted. Run: python3 test_parity.py"""
import tempfile
from pathlib import Path
import app


def _yaml(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def test_both_modes_build_the_same_tracker():
    # Same base yaml + same builder -> byte-identical effective config.
    base = app.TRACKERS_DIR / "tracker_stable.yaml"
    assert base.exists(), base
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        pa = app.effective_tracker(str(base), Path(a), 30.0)
        pb = app.effective_tracker(str(base), Path(b), 30.0)
        assert Path(pa).read_text() == Path(pb).read_text()
        cfg = _yaml(pa)
        assert cfg["track_buffer"] == str(int(30 * app.TRACK_BUFFER_SEC))
        assert cfg["new_track_thresh"] == str(app.NEW_TRACK_THRESH)


def test_track_buffer_scales_with_fps_and_has_a_floor():
    with tempfile.TemporaryDirectory() as d:
        base = app.TRACKERS_DIR / "tracker_stable.yaml"
        assert _yaml(app.effective_tracker(str(base), Path(d), 60.0))["track_buffer"] == "150"
        # a low-fps source still keeps a usable buffer
        assert int(_yaml(app.effective_tracker(str(base), Path(d), 10.0))["track_buffer"]) >= 60


def test_live_does_not_keep_its_own_tracker_file():
    # A second base config is a second place to tune and forget.
    assert not (app.TRACKERS_DIR / "tracker_live.yaml").exists(), \
        "tracker_live.yaml is back — Live would silently diverge again"


def test_live_never_blocks_on_a_question():
    # The old trigger machinery parked the worker thread on an answer gate: the
    # tracker stopped, the panel froze, and a session could sit there forever.
    # Live asks about a lost track by queueing it and pausing playback, never by
    # blocking the loop — so no trigger set, no ask gate, no "asking" status.
    src = Path("live.py").read_text()
    for gone in ("_trigger", "self.triggers", '"asking"', "_answered.wait"):
        assert gone not in src, f"blocking question machinery is back: {gone}"
    assert "def _check_lost" in src and "self.pending" in src


def test_detection_call_parameters_match():
    # conf / max_det / imgsz are what decide which players exist at all.
    import re
    src = Path("app.py").read_text()
    calls = re.findall(r"model_det\.track\((.*?)\)", src, re.S)
    live_call = re.search(r"self\.model_det\.track\((.*?)\)",
                          Path("live.py").read_text(), re.S).group(1)
    assert calls, "no batch track() call found"
    for key in ("conf=0.6", "max_det=16", "persist=True"):
        assert key in calls[0].replace(" ", "").replace("\n", "") or key in calls[0], (key, calls[0])
        assert key in live_call.replace(" ", "").replace("\n", "") or key in live_call, (key, live_call)


if __name__ == "__main__":
    test_both_modes_build_the_same_tracker()
    test_track_buffer_scales_with_fps_and_has_a_floor()
    test_live_does_not_keep_its_own_tracker_file()
    test_live_never_blocks_on_a_question()
    test_detection_call_parameters_match()
    print("live/automate parity OK")
