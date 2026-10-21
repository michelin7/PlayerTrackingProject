"""Self-check for the review timeline: /api/job/<id>/shifts must rebuild the
on-ice chart from the CURRENT review decisions, with the same builder the live
session uses. Run: python3 test_reviewshifts.py"""
import json, tempfile
from pathlib import Path
import app


MANIFEST = {
    "fps": 30.0, "mode": "live",
    "roster": {"Home": {"10": "Ivanov"}, "Away": {"22": "Petrov"}},
    "lines": {},
    "groups": [
        {"canonical_pid": 1, "auto_team": "Home", "auto_number": "10", "auto_name": "Ivanov",
         "clips": [{"track_id": 1, "first_seen": 0.0, "toi": 20.0, "iv": [[0.0, 20.0]]},
                   {"track_id": 2, "first_seen": 40.0, "toi": 10.0, "iv": [[40.0, 50.0]]}]},
    ],
    "unidentified": [
        {"canonical_pid": "u", "auto_team": None, "auto_number": None, "auto_name": None,
         "clips": [{"track_id": 7, "first_seen": 60.0, "toi": 8.0, "iv": [[60.0, 68.0]],
                    "auto_team": None, "auto_number": None}]},
    ],
}


def _client(tmp):
    job = Path(tmp) / "live_test"
    job.mkdir(parents=True, exist_ok=True)
    (job / "review_manifest.json").write_text(json.dumps(MANIFEST), encoding="utf-8")
    app.RESULTS_FOLDER = Path(tmp)
    return app.app.test_client()


def _shifts(client, decisions):
    r = client.post("/api/job/live_test/shifts", json=decisions)
    assert r.status_code == 200, r.status_code
    return {(s["team"], s["number"]): s for s in r.get_json()["shifts"]}


def test_default_timeline_matches_the_session():
    with tempfile.TemporaryDirectory() as tmp:
        c = _client(tmp)
        rows = _shifts(c, {})
        row = rows[("Home", "10")]
        # two clips, 20s apart -> one bridged shift, the gap counted as ice time
        assert row["name"] == "Ivanov" and row["shifts"] == 1, row
        assert abs(row["toi"] - 50.0) < 0.1, row
        assert ("Away", "22") not in rows            # the unidentified clip counts as nobody
    print("test_default_timeline_matches_the_session OK")


def test_dropping_a_track_takes_its_time_off_the_chart():
    with tempfile.TemporaryDirectory() as tmp:
        c = _client(tmp)
        rows = _shifts(c, {"2": {"action": "reject"}})
        row = rows[("Home", "10")]
        assert abs(row["toi"] - 20.0) < 0.1, row     # only the first clip is left
        assert row["shifts"] == 1, row


def test_assigning_an_unidentified_track_puts_it_on_the_chart():
    with tempfile.TemporaryDirectory() as tmp:
        c = _client(tmp)
        rows = _shifts(c, {"7": {"action": "assign", "target_team": "Away",
                                 "target_number": "22", "target_name": "Petrov"}})
        row = rows[("Away", "22")]
        assert abs(row["toi"] - 8.0) < 0.1 and row["name"] == "Petrov", row


def test_reassigning_moves_the_time_between_players():
    with tempfile.TemporaryDirectory() as tmp:
        c = _client(tmp)
        rows = _shifts(c, {"1": {"action": "reassign", "target_team": "Away",
                                 "target_number": "22", "target_name": "Petrov"}})
        assert abs(rows[("Away", "22")]["toi"] - 20.0) < 0.1, rows
        assert abs(rows[("Home", "10")]["toi"] - 10.0) < 0.1, rows


def test_review_replays_the_video_the_run_was_made_from():
    """Both modes review the whole match, so the manifest must point at the
    source file — and the route must serve it without the job in memory."""
    with tempfile.TemporaryDirectory() as tmp:
        c = _client(tmp)
        video = Path(tmp) / "match.mp4"
        video.write_bytes(b"\x00fake mp4")
        job = Path(tmp) / "live_test" / "review_manifest.json"
        m = dict(MANIFEST, source_video=str(video))
        job.write_text(json.dumps(m), encoding="utf-8")
        app.processing_jobs.pop("live_test", None)
        r = c.get("/api/job/live_test/source")
        assert r.status_code == 200, r.status_code
        assert r.data == b"\x00fake mp4", r.data


def test_a_cut_splits_the_clip_between_two_players():
    """Review can cut a track the tracker jumped mid-way, exactly as live can.
    Each half must then carry its own player on the timeline."""
    with tempfile.TemporaryDirectory() as tmp:
        c = _client(tmp)
        r = c.post("/api/job/live_test/shifts", json={
            "decisions": {"1#1": {"action": "reassign", "target_team": "Away",
                                  "target_number": "22", "target_name": "Petrov"}},
            "splits": {"1": [10.0]}})
        rows = {(s["team"], s["number"]): s for s in r.get_json()["shifts"]}
        assert abs(rows[("Away", "22")]["toi"] - 10.0) < 0.1, rows
        assert abs(rows[("Home", "10")]["toi"] - 20.0) < 0.1, rows   # 1#0 + clip 2


def test_a_cut_is_placed_by_match_time_not_file_time():
    """A clip file holds only the frames its track was seen on. Given the cut's
    match time, each part must get its real share of the ice — the file offset
    alone would hand a track with gaps the wrong seconds (and lose short parts)."""
    with tempfile.TemporaryDirectory() as tmp:
        c = _client(tmp)
        gap = dict(MANIFEST)
        # one track, seen 0-5 and 35-40: 10s of ice spread over 40s of video
        gap["groups"] = [{"canonical_pid": 1, "auto_team": "Home", "auto_number": "10",
                          "auto_name": "Ivanov",
                          "clips": [{"track_id": 1, "first_seen": 0.0, "toi": 10.0,
                                     "iv": [[0.0, 5.0], [35.0, 40.0]]}]}]
        gap["unidentified"] = []
        (Path(tmp) / "live_test" / "review_manifest.json").write_text(
            json.dumps(gap), encoding="utf-8")
        r = c.post("/api/job/live_test/shifts", json={
            "decisions": {"1#1": {"action": "reassign", "target_team": "Away",
                                  "target_number": "22", "target_name": "Petrov"}},
            "splits": {"1": [5.0]}, "split_times": {"1": [35.0]}})
        rows = {(s["team"], s["number"]): s for s in r.get_json()["shifts"]}
        assert abs(rows[("Home", "10")]["toi"] - 5.0) < 0.1, rows
        assert abs(rows[("Away", "22")]["toi"] - 5.0) < 0.1, rows
        assert abs(rows[("Away", "22")]["intervals"][0][0] - 35.0) < 0.1, rows


if __name__ == "__main__":
    test_default_timeline_matches_the_session()
    test_dropping_a_track_takes_its_time_off_the_chart()
    test_assigning_an_unidentified_track_puts_it_on_the_chart()
    test_reassigning_moves_the_time_between_players()
    test_review_replays_the_video_the_run_was_made_from()
    test_a_cut_splits_the_clip_between_two_players()
    test_a_cut_is_placed_by_match_time_not_file_time()
    print("review timeline OK")
