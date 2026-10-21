"""Live review: play a video through the tracker and let the operator correct
any track while it runs, logging every correction as ground truth for tuning.

The heavy per-frame work (player detection + tracking + number/team ID) is the
SAME code path as the batch pipeline — this only wraps it in an interactive
loop with pause control. One session at a time (single-user local tool).

The system never stops to ask. It plays, shows every track it is holding, and
the operator fixes whatever looks wrong — a number/team, a cut where the
tracker jumped players, a phantom in the shift list. Auto-pause questions used
to live here; they interrupted more than they helped and are gone.
"""
import base64
import cv2
import json
import time
from threading import Thread, Lock, Event
from queue import Queue, Empty

MIN_SHIFT_TOI = 1.0                 # sec; hide sub-second blips (misreads) from the chart
SIGHTING_MIN = 1.0                  # sec; a presence segment shorter than this survived the
                                    # flicker merge, so the number really was read for only a
                                    # moment — a misread, not a player on the ice
FLICKER_GAP = 1.5                   # sec; merge tiny on-screen gaps into one solid segment
LOST_MIN_TOI = 2.0                  # don't ask about a track seen less than this
ASK_CROP_H = 220                    # px; stored crop height (asked about later)
BRIDGE_GAP = 30.0                   # sec; a shorter off-screen gap still counts as ice time
                                    # (player was on the ice, just off camera) — shown hatched
APP_ORANGE_RGB = (255, 99, 0)       # app brand (--primary) — marker fallback for the light team
APP_CYAN_RGB = (41, 182, 216)       # clean cyan — marker fallback for the dark team


def team_rgb(idf):
    """Per-team marker colour FROM THE CONFIG jersey colour. Neutral jerseys
    (white/black/grey read poorly as an overlay) fall back to the app palette:
    brand orange for the dark team, cyan for the light team. Always distinct.

    Shared by the live overlay and the batch run, so the same match is drawn in
    the same two colours wherever it is watched."""
    def cfg_rgb(team):
        for nm in idf.teams.get(team, []):
            c = idf.BASIC_COLORS.get(nm)
            if c and (max(c) - min(c)) >= 45:          # colourful enough to use
                return tuple(int(v) for v in c)
        return None

    fb = {idf._light_team: APP_ORANGE_RGB, idf._dark_team: APP_CYAN_RGB}
    rgb = {"Home": cfg_rgb("Home") or fb["Home"], "Away": cfg_rgb("Away") or fb["Away"]}
    if rgb["Home"] == rgb["Away"]:                     # ensure the teams differ
        rgb["Away"] = APP_ORANGE_RGB if rgb["Home"] != APP_ORANGE_RGB else APP_CYAN_RGB
    return rgb


def team_hex(idf):
    return {t: "#%02X%02X%02X" % c for t, c in team_rgb(idf).items()}


def _jpeg(bgr, quality=80):
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


def _numkey(n):
    """Sort jersey numbers numerically; anything unparsable goes last."""
    try:
        return (0, int(n))
    except (TypeError, ValueError):
        return (1, str(n))


def build_shifts(presence, names=None, on_ice=None):
    """On-ice shifts from presence intervals, shared by live + batch review.

    presence: {(team, number): [[start_s, end_s], ...]}  (already flicker-merged)
    Returns a list of rows: team, number, name, toi, shifts, on, intervals, bridged.
    A gap < BRIDGE_GAP still counts as ice time (shown hatched); a longer gap
    starts a new shift.

    Momentary sightings are dropped BEFORE bridging, which is the whole point: a
    bridge means "we lost sight of a player we had properly seen", so anchoring
    one on a misread invents ice time out of nothing. Left in, a number read for
    two frames at each end of a clip became a player with a full-length shift —
    one session showed 18.5s of ice time for someone never actually seen, and
    inflated a team to 6 skaters on ice where 4 were real.
    """
    names = names or {}
    on_ice = on_ice or {}

    out = []
    for (tm, num), iv in presence.items():
        solid = [[s, e] for s, e in sorted(iv) if e - s >= SIGHTING_MIN]
        if not solid:
            continue
        bridged = []
        shift_count = 1 if solid else 0
        for i in range(1, len(solid)):
            gap = solid[i][0] - solid[i - 1][1]
            if gap < BRIDGE_GAP:
                if gap > 0:
                    bridged.append([solid[i - 1][1], solid[i][0]])
            else:
                shift_count += 1
        seen = sum(e - s for s, e in solid)
        toi = seen + sum(e - s for s, e in bridged)
        on = on_ice.get((tm, num), False)
        if seen < MIN_SHIFT_TOI:      # judged on time actually SEEN, never on
            continue                  # time the bridge rule filled in
        out.append({"team": tm, "number": num, "name": names.get((tm, num), ""),
                    "toi": round(toi, 1), "shifts": shift_count, "on": on,
                    "intervals": [[round(s, 2), round(e, 2)] for s, e in solid],
                    "bridged": [[round(s, 2), round(e, 2)] for s, e in bridged]})
    out.sort(key=lambda r: (r["team"], _numkey(r["number"])))
    return out


class LiveSession:
    def __init__(self, cap, model_det, identifier, tracker_path, fps, total,
                 out_dir, det_imgsz=640, frame_stride=1,
                 device=None, half=False, lines=None, video_path=""):
        # Configured pairs/lines, passed through to the shift list so it reads
        # by unit (defence pairs, then forward lines, per team).
        self.lines = lines or {}
        self.cap = cap
        self.model_det = model_det
        self.device = device        # explicit CUDA/MPS/CPU so acceleration is deterministic
        self.half = half            # FP16 on CUDA
        self.identifier = identifier
        self._compute_team_colors()
        self.tracker_path = tracker_path
        self.video_path = str(video_path or "")   # replayed in review
        # How long the TRACKER itself keeps a track it cannot see. Past this it
        # is forgotten and a returning player comes back as a new id — so it is
        # also the honest horizon for "may come back into frame", and the moment
        # an unanswered track becomes a question. Read from the config in use,
        # never guessed, or the panel would promise a comeback that cannot happen.
        self.forget_frames = self._tracker_buffer(tracker_path)
        self.fps = fps or 25
        self.frame_stride = max(int(frame_stride), 1)   # process every Nth source frame (~30fps)
        self.total = total
        self.det_imgsz = det_imgsz

        self.out_dir = out_dir
        self.job_id = out_dir.name          # RESULTS_FOLDER/<job_id> -> /review/<job_id>
        self.log_path = out_dir / "live_log.jsonl"

        # Per-player clip recording — SAME format as the batch pipeline
        # (full frame + a box on the tracked player, one file per track id), so
        # a finished live session opens the identical review page as Automate mode.
        self.videos_dir = out_dir / "player_videos"
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self._fourcc = cv2.VideoWriter_fourcc(*"avc1")
        self._writers = {}                  # pid -> cv2.VideoWriter (lazy)
        self._write_q = Queue(maxsize=200)
        self._write_stop = False
        self._writer_thread = Thread(target=self._writer_loop, daemon=True)
        self.frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        self.frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        self.track_identity = {}            # pid -> (team, number, name) last resolved
        # Where every track was, frame by frame: review draws the same markers on
        # the original video, so the recording is watched the way it was worked.
        self.boxes = []                     # [[t, [[pid, x1, y1, x2, y2], ...]], ...]
        # Video time of the frame in hand, taken from the DECODER, not from a
        # frame count: a file can start at a non-zero timestamp or lose frames at
        # the head, and then counting frames drifts against the video a player
        # shows (measured: 1.04 s on a 25 fps clip).
        self.now_t = 0.0
        # Per-track number votes + the detector's confidence in each read — the
        # same evidence Automate accumulates, so a live session's clips can be
        # resolved by weighted voting (not by whatever the last frame happened to
        # say) and can report a confidence in review.
        self.track_votes = {}               # pid -> {"nums": [...], "confs": [...]}
        # Manual mid-track cuts. The tracker sometimes jumps onto another player;
        # an override only relabels the track, so without a cut the frames of BOTH
        # players stay in one clip and the whole thing is finalized as one person.
        # A cut records where the clip must be split and who the part before it
        # belonged to; the operator then names the new part as usual.
        self.track_cuts = {}                # pid -> [{clip_t, video_t, identity, votes}]
        self._written = {}                  # pid -> frames queued for its clip file
        self.review_ready = False           # manifest written
        self.result_ready = False           # finalized -> frontend opens /result/<job>
        self.active_tracks = []             # snapshot of who is on screen this frame

        self.lock = Lock()
        self._resume = Event(); self._resume.set()     # manual pause gate
        self._stop = False

        # shared, read under lock by HTTP handlers
        self.status = "playing"          # playing | paused | finalizing | done | error
        self.frame_idx = 0
        self.frame_jpeg = b""
        self.error = None                # message shown instead of the status
        self.events = 0                  # logged corrections so far

        # per-track memory
        self.overrides = {}   # pid -> {"number", "team"}  user-confirmed = truth
        # per-track lifetime, so a track that was never identified can still be
        # attributed to a player by the operator (its ice time is not lost).
        self.track_iv = {}         # pid -> [[start_s, end_s], ...] presence per track
        self.track_numbered = set()# pids that ever got a confident number
        self.number_pids = {}      # (team, number) -> set(pids) confirmed as him
        self.pid_key = {}          # pid -> the player it is currently counted as
        self.track_from = {}       # pid -> second its CURRENT identity starts at
                                   # (a cut makes the rest of a track someone else)
        self.frozen = {}           # (team, number) -> ice time banked before a cut
        # Tracks the operator told us to leave alone — the goalie, a linesman, a
        # bench detection. They keep their box on screen (so they can be taken
        # back) but feed nothing: no identity, no ice time, no clip.
        self.ignored = set()
        # Running check of the identifier against the operator's own answers: on
        # a track he has named, what would the system have read by itself?
        self.agree = {}            # pid -> [matched_reads, reads]
        # What the save is doing right now. Writing out a session takes minutes
        # (every clip is re-encoded per player), and without this the page just
        # said "saving clips" and looked hung.
        self.save = {"phase": "", "done": 0, "total": 0}
        # A track the operator never ruled on, now gone from the camera. Its clip
        # would otherwise go to the final list wearing whatever the system last
        # read on it — which is exactly how a phantom player gets in.
        self.pending = {}          # pid -> question
        self.asked = set()         # pids already surfaced (ask once)
        self.track_last_seen = {}  # pid -> frame_idx
        self.track_crop = {}       # pid -> (area, bgr) best look at the player
        self._paused_to_ask = False

        # On-ice shifts, keyed by identified player (team, number). A shift is a
        # contiguous stretch of screen presence; a gap under SHIFT_GAP keeps the
        # same shift (brief occlusion / off-camera), a longer gap starts a new one.
        self.presence = {}    # (team, number) -> list of [start_s, end_s]
        self.on_ice = {}      # (team, number) -> currently present this frame
        self.names = {}       # (team, number) -> roster name (if configured)
        self.shift_gap = FLICKER_GAP  # merge only tiny gaps into a solid segment
        self.duration = (total / self.fps) if total else 0.0

    # ---- controls (called from HTTP handlers) ----------------------------
    @staticmethod
    def _tracker_buffer(path, default=75):
        try:
            for line in open(path, encoding="utf-8"):
                k, _, v = line.partition(":")
                if k.strip() == "track_buffer":
                    return max(int(float(v.split("#")[0].strip())), 1)
        except Exception:
            pass
        return default

    def start(self):
        self._writer_thread.start()
        Thread(target=self._run, daemon=True).start()

    # ---- per-player clip writer (background I/O thread) -------------------
    def _writer_loop(self):
        while True:
            try:
                item = self._write_q.get(timeout=1.0)
            except Empty:
                if self._write_stop:
                    break
                continue
            if item is None:
                break
            pid, frame_data, box, team = item
            w = self._writers.get(pid)
            if w is None:
                vp = str(self.videos_dir / f"player_{pid}.mp4")
                w = cv2.VideoWriter(vp, self._fourcc, self.fps, (self.frame_w, self.frame_h))
                self._writers[pid] = w
            fc = frame_data.copy()
            # Broadcast-style highlight: ground ellipse at the feet, team colour,
            # NO text (just the marker) — same look as the live overlay.
            color = self.team_marker.get(team, (150, 150, 150))
            x1, y1, x2, y2 = box
            cx, bw = (x1 + x2) // 2, (x2 - x1)
            ew = max(int(bw * 0.5), 10)
            eh = max(int(ew * 0.35), 4)
            cv2.ellipse(fc, (cx, y2), (ew, eh), 0, 0, 360, color, 2, cv2.LINE_AA)
            w.write(fc)

    def _save_phase(self, phase, done=0, total=0):
        with self.lock:
            self.save = {"phase": phase, "done": done, "total": total}

    def _finish_writers(self):
        """Drain the clip-writing queue, reporting how much is left.

        The queue lags the loop by seconds, so at the end of a session there can
        be thousands of frames still to write — the longest silent stretch of the
        save and the one that looked most like a hang."""
        self._write_stop = True
        try:
            self._write_q.put_nowait(None)
        except Exception:
            pass
        left = self._write_q.qsize()
        self._save_phase("clips", 0, left)
        while self._writer_thread.is_alive():
            self._writer_thread.join(timeout=0.5)
            now = self._write_q.qsize()
            self._save_phase("clips", max(left - now, 0), left)
        for w in self._writers.values():
            try:
                w.release()
            except Exception:
                pass

    def pause(self):
        self._resume.clear()
        with self.lock:
            if self.status == "playing":
                self.status = "paused"

    def resume(self):
        with self.lock:
            if self.status == "paused":
                self.status = "playing"
        self._resume.set()

    def stop(self):
        self._stop = True
        self._resume.set()

    # ---- log --------------------------------------------------------------
    def _log(self, entry):
        entry["t"] = round(time.time(), 3)
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass
        self.events += 1

    # ---- main loop --------------------------------------------------------
    def _run(self):
        idf = self.identifier
        # Process EVERY frame (same as Automate mode) — no real-time frame dropping.
        # Playback advances as fast as inference allows; the interactive layer
        # (manual pause + inline track edits) sits on top. This keeps the
        # recorded per-player clips full-quality and identical to the batch path.
        try:
            while not self._stop:
                self._resume.wait()                      # manual pause gate
                if self._stop:
                    break

                # Downsample to ~30fps: skip (stride-1) frames cheaply, decode one.
                skipped_ok = True
                for _ in range(self.frame_stride - 1):
                    if not self.cap.grab():
                        skipped_ok = False
                        break
                ret, frame = self.cap.read() if skipped_ok else (False, None)
                if ret:
                    pos = self.cap.get(cv2.CAP_PROP_POS_MSEC) or 0
                    self.now_t = (pos / 1000.0) if pos > 0 else (self.frame_idx + 1) / self.fps
                    if self.frame_idx == 0 and self.now_t > 0:
                        # the file starts where the decoder says it does, not at 0
                        self.duration += self.now_t
                if not ret:
                    with self.lock:
                        self.status = "done"
                    break
                self.frame_idx += 1

                # Pin CUDA+FP16 explicitly so Windows/Linux NVIDIA is deterministic;
                # MPS/CoreML/CPU keep ultralytics' own auto-select (already correct).
                _dev_kw = {"device": self.device, "half": self.half} if self.device == "cuda" else {}
                results = self.model_det.track(
                    frame, conf=0.6, persist=True, tracker=self.tracker_path,
                    max_det=16, verbose=False, imgsz=self.det_imgsz, **_dev_kw)
                dets = []
                for r in results:
                    for b in r.boxes:
                        if int(b.cls) == 0 and b.id is not None:
                            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                            dets.append((int(b.id), x1, y1, x2, y2))

                ignored = self.ignored
                vis = [d for d in dets if d[0] not in ignored]
                id_results = {}
                if dets:
                    id_results, _, _ = idf.identify_batch(frame, dets, self.frame_idx)
                    # user corrections are ground truth — apply them over the model.
                    # Re-derive the NAME from the roster so it always matches the
                    # (corrected) number/team instead of carrying the old name.
                    for pid, ov in list(self.overrides.items()):
                        if pid in id_results:
                            nm, num, tm = id_results[pid]
                            self._score(pid, num, tm, ov)   # before it is overwritten
                            num = ov.get("number", num); tm = ov.get("team", tm)
                            id_results[pid] = (self._roster_name(tm, num) or nm, num, tm)
                    for pid in ignored:
                        id_results.pop(pid, None)

                # Record this frame's number read per track (identified frames
                # only, matching the batch path) for end-of-session voting.
                for _pid, (_nm, _num, _tm) in id_results.items():
                    if _nm in (None, "None", "Unknown"):
                        continue
                    _v = self.track_votes.setdefault(_pid, {"nums": [], "confs": []})
                    _v["nums"].append(_num)
                    _lg = idf._last_guess.get(_pid)
                    _v["confs"].append(_lg[1] if (_lg and _lg[0] == _num) else 0.5)

                self._track_lifetime(vis, id_results, frame)
                self._update_shifts(id_results)
                self._check_lost()
                # Record per-player clips (full frame + box) from the CLEAN frame,
                # and remember each track's resolved identity for the manifest.
                on_screen = []
                for pid, x1, y1, x2, y2 in dets:
                    if pid in ignored:                   # still shown, still clickable
                        on_screen.append({"pid": pid, "team": None, "number": None,
                                          "name": None, "label": f"Track {pid}",
                                          "fixed": False, "cuts": 0, "ignored": True})
                        continue
                    nm, num, tm = id_results.get(pid, ("Unknown", None, "Unknown"))
                    self.track_identity[pid] = (tm, num, nm)
                    # Built here, in the worker, so the HTTP side never walks a dict
                    # that this loop is mutating.
                    on_screen.append({
                        "pid": pid, "team": tm if tm != "Unknown" else None,
                        "number": num,
                        "name": nm if nm not in (None, "", "None", "Unknown") else None,
                        "label": self.track_label(pid, num, nm),
                        "fixed": pid in self.overrides,
                        "cuts": len(self.track_cuts.get(pid) or []),
                        "ignored": False})
                    try:
                        self._write_q.put_nowait((pid, frame, (x1, y1, x2, y2), tm))
                        # counted here, not in the writer thread: this is the exact
                        # position a cut must land on, and the queue lags by up to
                        # a couple of seconds.
                        self._written[pid] = self._written.get(pid, 0) + 1
                    except Exception:
                        pass                              # drop a frame rather than stall
                self.boxes.append([round(self.now_t, 2),
                                   [[p, a, b, c, d] for p, a, b, c, d in vis]])
                on_screen.sort(key=lambda t: (t["team"] or "~", _numkey(t["number"]), t["pid"]))
                # Draw overlays on a COPY so the recorded clips keep a clean frame.
                disp = frame.copy()
                self._draw(disp, vis, id_results)
                with self.lock:
                    self.frame_jpeg = _jpeg(disp)
                    self.active_tracks = on_screen
        except Exception as e:
            with self.lock:
                self.status = "error"
                self.error = str(e)
        finally:
            try:
                self._build_review()
            except Exception as e:                    # never die in silence here:
                import traceback                      # the page waits on this
                traceback.print_exc()
                with self.lock:
                    self.status = "error"
                    self.error = f"Saving the session: {e}"

    # ---- review build (unified with Automate mode) ------------------------
    def _build_review(self):
        """On finish/stop: flush clips and write review_manifest.json so the
        session opens the SAME review page as Automate mode (shifts + clips
        grouped by player). Distance/speed are 0 (not measured live).

        Identity per clip comes from weighted voting over the whole track and the
        physical one-player-per-number constraint, exactly as in Automate — the
        last frame's guess alone made live clips noisier than batch ones for the
        same footage."""
        # Imported here, not at module scope: app imports this module.
        from app import TrackMerger, clip_confidence, dedupe_by_number, union_toi
        self._finish_writers()
        groups_by_key = {}      # (team, number) -> group entry
        unidentified = []
        n_tracks = len(self.track_identity)
        self._save_phase("tracks", 0, n_tracks)
        seen_tracks = set()
        for pid, (tm, num, nm), seg in self._segments():
            seen_tracks.add(pid)
            self._save_phase("tracks", len(seen_tracks), n_tracks)
            vp = self.videos_dir / seg["file"]
            if not vp.exists() or vp.stat().st_size == 0:
                continue
            iv = seg["iv"]
            if not iv:
                continue
            # One type for a number inside a session: text. The reads come from
            # the model as ints and from an override as a string, and a mixed list
            # silently broke both the vote and the share.
            nums = [None if v is None else str(v) for v in seg["nums"]]
            confs = seg["confs"]
            num = None if num is None else str(num)   # the identity, same type
            ov = seg["override"]                         # user correction wins
            if ov:
                tm, num = ov.get("team", tm), ov.get("number", num)
                nm = self._roster_name(tm, str(num)) if num is not None else ""
                num_conf = None     # the reviewer's own answer, not a system claim:
                                    # counting it would flatter the calibration report
            else:
                voted = TrackMerger.resolve_number(nums, min_count=2, confs=confs)
                if voted is not None and str(voted) != str(num):
                    num = str(voted)   # whole-track vote beats the last frame's read
                    nm = self._roster_name(tm, num) or nm
                num_conf, _ = clip_confidence(nums, confs, num)
            cast = [v for v in nums if v is not None]
            share = (round(sum(1 for v in cast if v == str(num)) / len(cast), 2)
                     if cast and num is not None else 0.0)
            first_seen = round(min(s for s, _ in iv), 1)
            toi = round(sum(e - s for s, e in iv), 1)
            if toi < 1.0:                            # short clip -> noise, skip review
                continue
            pid = seg["track_id"]
            clip = {"track_id": pid, "file": seg["file"], "toi": toi,
                    "first_seen": first_seen, "iv": [[round(a, 1), round(b, 1)] for a, b in iv],
                    "auto_team": tm if tm != "Unknown" else None,
                    "auto_number": num, "auto_name": nm or None,
                    "num_votes": len(cast), "num_share": share, "num_conf": num_conf,
                    # Set by hand during the session, so finalize counts it as a
                    # correction. A cut gives the segment an id of its own, which
                    # is NOT the tracker id the override is stored under — reading
                    # self.overrides here reported every fixed segment as a
                    # success and sent the accuracy straight to 100%.
                    "fixed": bool(ov)}
            if num is not None and tm and tm != "Unknown":
                key = (tm, str(num))
                g = groups_by_key.get(key)
                if g is None:
                    g = {"canonical_pid": pid, "auto_team": tm, "auto_number": num,
                         "auto_name": nm or self._roster_name(tm, str(num)) or None,
                         "total_toi": 0.0, "clips": []}
                    groups_by_key[key] = g
                g["clips"].append(clip)
            else:
                unidentified.append({"canonical_pid": pid, "auto_team": None,
                                     "auto_number": None, "auto_name": None,
                                     "total_toi": toi, "clips": [clip]})
        # Only one player can wear a number on the ice at a time: clips of the
        # same (team, number) that OVERLAP cannot all be that player. Keep the
        # strongest non-overlapping timeline and send the rest to review as
        # unidentified instead of piling them onto one card.
        groups = list(groups_by_key.values())
        ejected = []
        for g in groups:
            kept, rejected = dedupe_by_number(g["clips"])
            for c in rejected:
                c["auto_number"] = None
                c["auto_name"] = None
            ejected.extend(rejected)
            g["clips"] = sorted(kept, key=lambda c: c.get("first_seen", 0))
            # TOI is the UNION of the clips, never their sum — one body, one clock
            g["total_toi"] = round(union_toi(kept), 1)
        groups = [g for g in groups if g["clips"]]
        if ejected:
            ejected.sort(key=lambda c: c.get("first_seen", 0))
            unidentified.append({"canonical_pid": "ejected", "auto_team": None,
                                 "auto_number": None, "auto_name": None,
                                 "total_toi": round(union_toi(ejected), 1),
                                 "clips": ejected})

        _n_id = sum(len(g["clips"]) for g in groups)
        _n_players = len([g for g in groups if g.get("auto_number") is not None])
        manifest = {
            "fps": self.fps,
            "mode": "live",
            "groups": groups,
            "unidentified": unidentified,
            # What the review page's diagnostics bar can honestly show for a live
            # session: no split/crossing counters here, those are batch-only.
            "profile": {
                "processed_frames": self.frame_idx,
                "clips_identified": _n_id,
                "clips_unidentified": sum(len(u["clips"]) for u in unidentified),
                "clips_per_player": round(_n_id / _n_players, 1) if _n_players else 0,
                # measured against the identities the operator typed in himself
                "check": self.check_stats(),
            },
            "shifts": build_shifts(self.presence, self.names, self.on_ice),
            "roster": {"Home": dict(self.identifier.Home), "Away": dict(self.identifier.Away)},
            "lines": self.lines,
            "source_video": self.video_path,
            "team_colors": self.team_hex,
        }
        self._write_overlay()
        try:
            (self.out_dir / "review_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            self.review_ready = True
        except Exception:
            pass
        if self.review_ready:
            with self.lock:
                if self.status != "error":
                    self.status = "finalizing"
            self._save_phase("render")
            self._finalize(manifest)
        self._save_phase("")
        with self.lock:
            if self.status != "error":
                self.status = "done"

    def _write_overlay(self):
        """Dump the per-frame boxes for review, keyed by CLIP id.

        A cut track became several clips, so the frames before a cut belong to a
        different clip than the frames after it — review looks a box up by clip
        id to know whose it is."""
        def clip_id(pid, t):
            cuts = self.track_cuts.get(pid)
            if not cuts:
                return pid
            i = sum(1 for c in cuts if t >= c["video_t"])
            return int(f"9{pid:05d}{i:02d}")

        try:
            frames = [[t, [[clip_id(p, t), a, b, c, d] for p, a, b, c, d in boxes]]
                      for t, boxes in self.boxes]
            (self.out_dir / "overlay.json").write_text(
                json.dumps({"w": self.frame_w, "h": self.frame_h, "frames": frames}),
                encoding="utf-8")
        except Exception:
            pass                       # an overlay is a nicety; the review is not

    def _segments(self):
        """Yield (pid, identity, segment) for every piece of every track.

        A track the operator never cut yields itself, so nothing downstream has to
        know about cuts. A cut track yields one piece per segment, each with its
        own clip FILE (written here from the source), its own share of the number
        votes and its own identity — the parts are different players, which is the
        whole point of having cut it.
        """
        from app import write_clip_range
        for pid, identity in list(self.track_identity.items()):
            if pid in self.ignored:
                continue
            src = self.videos_dir / f"player_{pid}.mp4"
            cuts = self.track_cuts.get(pid) or []
            votes = self.track_votes.get(pid, {})
            iv = self.track_iv.get(pid, [])
            if not cuts:
                yield pid, identity, {
                    "track_id": pid, "file": src.name, "iv": iv,
                    "nums": votes.get("nums", []), "confs": votes.get("confs", []),
                    "override": self.overrides.get(pid)}
                continue
            if not src.exists():
                continue
            bounds = [0.0] + [c["clip_t"] for c in cuts] + [None]
            v_bounds = [None] + [c["video_t"] for c in cuts] + [None]
            for i in range(len(bounds) - 1):
                last = i == len(bounds) - 2
                # Every piece gets an id and a file of its own, named the way the
                # rest of the app expects, so a live clip can still be cut again in
                # review. The uncut source stays on disk, unreferenced.
                seg_pid = int(f"9{pid:05d}{i:02d}")
                dst = self.videos_dir / f"player_{seg_pid}.mp4"
                if not write_clip_range(src, dst, self.fps,
                                        bounds[i] or 0.0, bounds[i + 1]):
                    continue
                # presence inside this segment's video window
                lo = v_bounds[i] if v_bounds[i] is not None else float("-inf")
                hi = v_bounds[i + 1] if v_bounds[i + 1] is not None else float("inf")
                seg_iv = [[max(a, lo), min(b, hi)] for a, b in iv
                          if min(b, hi) > max(a, lo)]
                v0 = 0 if i == 0 else cuts[i - 1]["votes"]
                v1 = cuts[i]["votes"] if not last else None
                ident = cuts[i]["identity"] if not last else identity
                yield pid, ident, {
                    "track_id": seg_pid, "file": dst.name, "iv": seg_iv,
                    "nums": (votes.get("nums") or [])[v0:v1],
                    "confs": (votes.get("confs") or [])[v0:v1],
                    "override": ({"team": ident[0], "number": ident[1]}
                                 if (cuts[i]["fixed"] if not last else
                                     self.overrides.get(pid)) else None)}

    def _finalize(self, manifest):
        """Go straight to the finalized result — there is nothing left to review.

        Corrections were made live, so every clip already carries the identity the
        operator signed off on. We write the decisions that say so and run the SAME
        finalize as Automate (per-player videos, Excel, shifts, accuracy), instead
        of parking the session on a review page that would only say "approve all".

        A track the operator had to fix is recorded as a CORRECTION, not as an
        approval: otherwise every live session would report 100% accuracy and the
        auto-labeling quality would become unmeasurable.
        """
        import app                      # late: app imports this module
        clips_by_group = {}
        for g in manifest.get("groups", []):
            clips_by_group[str(g["canonical_pid"])] = {"clips": {
                str(c["track_id"]): (
                    {"action": "reassign", "target_team": g["auto_team"],
                     "target_number": g["auto_number"], "target_name": g["auto_name"]}
                    if c.get("fixed") else {"action": "reject"})
                for c in g["clips"]}}
        decisions = {"decisions": clips_by_group, "unidentified_decisions": {},
                     "splits": {}, "split_files": {}}
        try:
            (self.out_dir / "review_decisions.json").write_text(
                json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8")
            app.processing_jobs.setdefault(self.job_id, {
                "status": "finalizing", "progress": 100, "message": "",
                "video": "", "result_path": str(self.out_dir), "stats": None,
                "total_frames": 0, "processed_frames": 0})
            app.finalize_job(self.job_id)
            self.result_ready = (
                app.processing_jobs[self.job_id].get("status") == "completed")
            # The approve-rate accuracy means nothing here (nothing is accepted
            # on the system's word) — report the running check against the
            # operator's own answers instead.
            acc = self.out_dir / "accuracy_report.json"
            if acc.exists():
                rep = json.loads(acc.read_text(encoding="utf-8"))
                rep["mode"] = "live"
                rep["check"] = self.check_stats()
                acc.write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                               encoding="utf-8")
        except Exception as e:
            with self.lock:
                self.status = "error"
                self.error = f"Finalization: {e}"

    # ---- team colours -----------------------------------------------------
    def _compute_team_colors(self):
        rgb = team_rgb(self.identifier)
        self.team_rgb = rgb
        self.team_marker = {t: (c[2], c[1], c[0]) for t, c in rgb.items()}   # BGR for cv2
        self.team_hex = {t: "#%02X%02X%02X" % c for t, c in rgb.items()}
        # black text on light markers, white on dark, by luminance
        self.team_text = {t: ((25, 25, 25) if (0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]) > 140
                              else (245, 245, 245)) for t, c in rgb.items()}

    # ---- draw -------------------------------------------------------------
    @staticmethod
    def _rounded(img, x, y, w, h, r, color):
        """Filled rounded rectangle (chip)."""
        r = max(0, min(r, w // 2, h // 2))
        cv2.rectangle(img, (x + r, y), (x + w - r, y + h), color, -1)
        cv2.rectangle(img, (x, y + r), (x + w, y + h - r), color, -1)
        for cxr, cyr in ((x + r, y + r), (x + w - r, y + r),
                         (x + r, y + h - r), (x + w - r, y + h - r)):
            cv2.circle(img, (cxr, cyr), r, color, -1, cv2.LINE_AA)

    def _draw(self, frame, dets, id_results):
        """Broadcast-style markers: a ground ellipse at the feet + a rounded
        name chip above the head, team-coloured (no rectangular boxes).

        The chip names the player — "#25 Olson" — so the operator can read the ice
        without cross-referencing a panel. A track the system has not identified
        shows its TRACK id instead, which is what a correction is addressed to.
        """
        H, W = frame.shape[:2]
        for pid, x1, y1, x2, y2 in dets:
            name, num, team = id_results.get(pid, ("Unknown", None, "Unknown"))
            color = self.team_marker.get(team, (150, 150, 150))
            text_color = self.team_text.get(team, (245, 245, 245))
            cx, bw = (x1 + x2) // 2, (x2 - x1)
            # ground ellipse at the feet
            ew = max(int(bw * 0.5), 10)
            eh = max(int(ew * 0.35), 4)
            cv2.ellipse(frame, (cx, y2), (ew, eh), 0, 0, 360, color, 2, cv2.LINE_AA)
            label = self.track_label(pid, num, name)
            fs = 0.55
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            cw, ch = tw + 12, th + 8
            lx = max(0, min(int(cx - cw / 2), W - cw))
            ly = max(y1 - ch - 6, 2)
            self._rounded(frame, lx, ly, cw, ch, 6, color)
            cv2.putText(frame, label, (lx + 6, ly + ch - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, text_color, 1, cv2.LINE_AA)

    @staticmethod
    def track_label(pid, num, name):
        """What to print over a player: the jersey identity when known, else the
        track id — the handle the operator uses to correct it."""
        if num is None:
            return f"Track {pid}"
        clean = name if name not in (None, "", "None", "Unknown") else ""
        return f"#{num} {clean}".strip()

    # ---- per-track lifetime (so a never-identified track keeps its ice time) --
    def _track_lifetime(self, dets, id_results, frame=None):
        now = self.now_t
        for pid, x1, y1, x2, y2 in dets:
            _nm, num, _tm = id_results.get(pid, ("Unknown", None, "Unknown"))
            self.track_last_seen[pid] = self.frame_idx
            if frame is not None and pid not in self.overrides:
                area = (x2 - x1) * (y2 - y1)         # keep the biggest look at it
                if area > self.track_crop.get(pid, (0, None))[0]:
                    crop = frame[max(y1, 0):y2, max(x1, 0):x2]
                    if crop.size:
                        if crop.shape[0] > ASK_CROP_H:   # stored, not encoded: a
                            k = ASK_CROP_H / crop.shape[0]   # session holds hundreds
                            crop = cv2.resize(crop, (max(int(crop.shape[1] * k), 1), ASK_CROP_H))
                        self.track_crop[pid] = (area, crop.copy())
            iv = self.track_iv.setdefault(pid, [])
            if iv and now - iv[-1][1] <= self.shift_gap:
                iv[-1][1] = now
            else:
                iv.append([now, now])
            if num is not None:
                self.track_numbered.add(pid)

    def _check_lost(self):
        """A track that left the camera without the operator ever ruling on it
        becomes a question. Playback pauses so the question is seen, and answering
        the last one lets it run on again."""
        stale = self.frame_idx - (self.forget_frames + self.fps)   # + a second of grace
        new = False
        for pid, last in list(self.track_last_seen.items()):
            if last >= stale or pid in self.asked:
                continue
            self.asked.add(pid)                  # ask about a track once, ever
            if pid in self.ignored or pid in self.overrides:
                continue                         # already ruled on
            toi = sum(e - s for s, e in self.track_iv.get(pid, []))
            if toi < LOST_MIN_TOI:
                continue                         # a blip, not worth a question
            tm, num, nm = self.track_identity.get(pid, ("Unknown", None, ""))
            with self.lock:
                self.pending[pid] = {
                    "pid": pid, "toi": round(toi, 1), "frame": self.frame_idx,
                    "team": tm if tm != "Unknown" else None, "number": num,
                    "name": nm if nm not in (None, "", "None", "Unknown") else None,
                    "label": self.track_label(pid, num, nm),
                    "crop": self._crop_b64(pid)}
            new = True
        if new and self.status == "playing":
            self._paused_to_ask = True
            self.pause()

    def offscreen_tracks(self):
        """Tracks out of frame that the tracker still holds — the ones that can
        reappear under the same id, with the identity they already carry.
        Caller holds the lock."""
        on = {t["pid"] for t in self.active_tracks}
        out = []
        for pid, last in self.track_last_seen.items():
            gone = self.frame_idx - last
            if pid in on or pid in self.ignored or not 0 < gone <= self.forget_frames:
                continue
            ov = self.overrides.get(pid)
            tm, num, nm = self.track_identity.get(pid, ("Unknown", None, ""))
            if ov:
                tm, num = ov.get("team", tm), ov.get("number", num)
                nm = self._roster_name(tm, str(num)) if num is not None else ""
            out.append({"pid": pid, "team": tm if tm != "Unknown" else None,
                        "number": num,
                        "name": nm if nm not in (None, "", "None", "Unknown") else None,
                        "label": self.track_label(pid, num, nm),
                        "fixed": pid in self.overrides,
                        "cuts": len(self.track_cuts.get(pid) or []),
                        "gone": round(gone / self.fps, 1)})
        out.sort(key=lambda t: (t["gone"], t["pid"]))
        return out

    def _crop_b64(self, pid):
        area_crop = self.track_crop.get(pid)
        if not area_crop or area_crop[1] is None:
            return ""
        jb = _jpeg(area_crop[1], 85)
        return ("data:image/jpeg;base64," + base64.b64encode(jb).decode()) if jb else ""

    def _answered(self, pid):
        """Drop a question. The CALLER HOLDS THE LOCK, so resuming (which takes
        it again) is left to _maybe_resume once the caller has let go."""
        self.pending.pop(pid, None)
        self.track_crop.pop(pid, None)

    def _maybe_resume(self):
        """Answering the last question lets playback run on by itself."""
        if not self.pending and self._paused_to_ask:
            self._paused_to_ask = False
            self.resume()

    def dismiss_pending(self, pid):
        """Skip: leave the track exactly as the system read it."""
        with self.lock:
            q = self.pending.get(pid) or {}
            self._log({"frame": self.frame_idx, "trigger": "lost", "pid": pid,
                       "proposed": {"number": q.get("number"), "team": q.get("team")},
                       "action": "skip", "applied": []})
            self._answered(pid)
        self._maybe_resume()
        return True

    # ---- shifts -------------------------------------------------------------
    # A player's ice time is the union of the lifetimes of the tracks the
    # OPERATOR confirmed as him — derived, never accumulated. The system's own
    # read is a guess, and guesses used to write themselves into the chart as
    # real ice time: that is where the phantom players came from.
    def _rebuild_presence(self, key):
        """Recompute one player's timeline from his tracks. Caller holds the lock."""
        iv = [list(p) for p in self.frozen.get(key, [])]
        for pid in self.number_pids.get(key) or ():
            start = self.track_from.get(pid, 0.0)
            iv += [[max(s, start), e] for s, e in self.track_iv.get(pid, []) if e > start]
        iv.sort()
        merged = []
        for s, e in iv:
            if merged and s - merged[-1][1] <= self.shift_gap:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        if merged:
            self.presence[key] = merged
            self.names[key] = self._roster_name(*key)
        else:                   # no ice time yet (a track just confirmed on its
            for d in (self.presence, self.names, self.on_ice):   # first frame)
                d.pop(key, None)  # — the attachment itself stays, number_pids is
                                  # what holds it

    def _detach(self, pid):
        """Stop counting a track for whoever it was counted as. Caller holds the lock."""
        old = self.pid_key.pop(pid, None)
        if old is None:
            return
        pids = self.number_pids.get(old) or set()
        pids.discard(pid)
        if not pids:
            self.number_pids.pop(old, None)
        self._rebuild_presence(old)

    def _attach(self, pid, key):
        """Count a track as this player, from its first frame. Caller holds the lock."""
        if self.pid_key.get(pid) == key:
            self._rebuild_presence(key)          # the track grew
            return
        self._detach(pid)
        self.pid_key[pid] = key
        self.number_pids.setdefault(key, set()).add(pid)
        self._rebuild_presence(key)

    def edit_shift(self, team, number, action, new_number=None, new_team=None):
        """Fix a player in the shift list: drop him, or move his tracks to
        another number/team."""
        key = (team, str(number))
        pids = set(self.number_pids.get(key, set()))
        nt, nk = (new_team or team), None
        if action == "reassign" and new_number:
            nk = (nt, str(new_number))
        for pid in pids:
            self._apply_override(pid, str(new_number) if nk else None, nt if nk else team)
        with self.lock:
            for pid in pids:
                if nk:
                    self._attach(pid, nk)
                else:
                    self._detach(pid)
                self._refresh_active(pid)
            self.frozen.pop(key, None)        # banked time goes with the player
            self._rebuild_presence(key)
        return True

    def _update_shifts(self, id_results):
        present = set()
        with self.lock:                     # state() iterates presence under the same lock
            for pid in id_results:
                ov = self.overrides.get(pid)
                if not ov or ov.get("number") is None:
                    continue                # unconfirmed: a guess, not ice time
                key = (ov["team"], str(ov["number"]))
                present.add(key)
                self._attach(pid, key)
            self.on_ice = {k: (k in present) for k in self.presence}

    def _score(self, pid, num, team, ov):
        """Count this frame's raw read against what the operator typed in.

        A hand-entered identity is ground truth and is never touched, so the
        system's own read for that track is free evidence: it says how often the
        identifier would have been right where we know the answer."""
        if num is None or ov.get("number") is None:
            return                      # no read to score against
        a = self.agree.setdefault(pid, [0, 0])
        a[1] += 1
        if str(num) == str(ov["number"]) and team == ov.get("team"):
            a[0] += 1

    def check_stats(self):
        """Aggregate of _score: {reads, agree, pct, tracks}."""
        reads = sum(v[1] for v in self.agree.values())
        hits = sum(v[0] for v in self.agree.values())
        return {"reads": reads, "agree": hits,
                "pct": round(hits / reads * 100) if reads else None,
                "tracks": len([1 for v in self.agree.values() if v[1]])}

    def _roster_name(self, team, num):
        squad = self.identifier.Home if team == "Home" else self.identifier.Away
        return squad.get(str(num), "")

    def shifts(self):
        return build_shifts(self.presence, self.names, self.on_ice)

    def _apply_override(self, pid, number, team):
        """Set a track's identity from the operator — ground truth, used forward.

        The single funnel for every manual identity change (track chip, applied-
        corrections list, shift editor), so the ground-truth log is written here
        rather than in each caller."""
        proposed = self.track_identity.get(pid, ("Unknown", None, ""))
        self.overrides[pid] = {"number": number, "team": team}
        action = "correct" if number is not None else "clear"
        self._log({"frame": self.frame_idx, "trigger": "manual", "pid": pid,
                   "proposed": {"number": proposed[1], "team": proposed[0]},
                   "action": action,
                   "applied": [{"pid": pid, "number": number, "team": team}]})

    def cut_track(self, pid):
        """Split a track at the CURRENT frame: everything up to now belonged to
        one player, everything after to another.

        Live's answer to the review-time cut. Here the operator is watching the
        jump happen, so the cut point is simply "now" — no scrubbing needed. The
        part before keeps the identity built so far; the part after starts clean,
        including the identifier's own memory of this track, so it re-reads the
        new jersey instead of carrying the old number forward.
        """
        with self.lock:
            written = self._written.get(pid, 0)
            if written < self.fps:          # under a second of footage on either
                return False                # side is a scrap, not a segment
            team, num, name = self.track_identity.get(pid, ("Unknown", None, ""))
            ov = self.overrides.get(pid)
            if ov:
                team, num = ov.get("team", team), ov.get("number", num)
                name = self._roster_name(team, str(num)) if num is not None else ""
            votes = self.track_votes.get(pid, {})
            self.track_cuts.setdefault(pid, []).append({
                "clip_t": round(written / self.fps, 2),
                "video_t": round(self.now_t, 2),
                "identity": (team, num, name),
                "fixed": ov is not None,
                # how many number reads belong to the part BEFORE the cut
                "votes": len(votes.get("nums", [])),
            })
            # A cut is the operator telling us the tracker jumped players — the
            # single most useful line in the log for tuning it.
            self._log({"frame": self.frame_idx, "trigger": "manual", "pid": pid,
                       "proposed": {"number": num, "team": team},
                       "action": "cut", "applied": []})
            # The part after the cut is a different player: forget everything this
            # track had decided, or the old number is voted back in immediately.
            # Everything up to the cut stays with the player it was: bank it,
            # then let the track go. What comes after is someone else's, and is
            # counted only from the cut once the operator names him.
            cut_t = self.now_t
            key = self.pid_key.get(pid)
            if key:
                start = self.track_from.get(pid, 0.0)
                self.frozen.setdefault(key, []).extend(
                    [max(s, start), min(e, cut_t)] for s, e in self.track_iv.get(pid, [])
                    if min(e, cut_t) > max(s, start))
                self._detach(pid)
            self.track_from[pid] = cut_t
            self.overrides.pop(pid, None)
            self.track_identity[pid] = ("Unknown", None, "")
            self.track_votes[pid] = {"nums": [], "confs": []}
            self.agree.pop(pid, None)
            for store in (self.identifier._cache, self.identifier._est_number,
                          self.identifier._team_votes, self.identifier._num_switch_counts,
                          self.identifier._est_team, self.identifier._recent_team):
                store.pop(pid, None)
            self._refresh_active(pid)
        return True

    def ignore_track(self, pid, on=True):
        """Leave this track out of everything: it is not a skater we are counting.

        The goalie is the standing case — the tracker holds him for the whole
        period and the identifier keeps handing him somebody's number. Clearing
        the correction only lets the next frame assign it again; ignoring stops
        the track at the source."""
        with self.lock:
            if on:
                self.ignored.add(pid)
                self.overrides.pop(pid, None)
                # Its ice time belongs to nobody: the player it was counted as
                # keeps whatever OTHER tracks give him, and loses this one.
                self._detach(pid)
            else:
                self.ignored.discard(pid)
            self._refresh_active(pid)
            self._answered(pid)
        self._maybe_resume()
        self._log({"frame": self.frame_idx, "trigger": "manual", "pid": pid,
                   "proposed": {"number": None, "team": None},
                   "action": "ignore" if on else "unignore", "applied": []})
        return True

    def _refresh_active(self, pid):
        """Re-label one track in the on-screen snapshot. Caller holds the lock.

        The snapshot is otherwise only rebuilt by the worker loop — which is
        parked while the session is paused or waiting for an answer, i.e. exactly
        when the operator is using the panel. Without this a saved correction
        produced no visible change at all and read as a dead Save button.
        """
        for t in self.active_tracks:
            if t["pid"] != pid:
                continue
            t["ignored"] = pid in self.ignored
            if t["ignored"]:
                t["team"] = t["number"] = t["name"] = None
                t["label"] = f"Track {pid}"
                t["fixed"] = False
                continue
            ov = self.overrides.get(pid)
            if ov:
                team, num = ov.get("team"), ov.get("number")
                name = self._roster_name(team, str(num)) if num is not None else None
            else:                      # correction cleared -> back to the model's read
                team, num, name = self.track_identity.get(pid, ("Unknown", None, ""))
            t["team"] = team if team and team != "Unknown" else None
            t["number"] = num
            t["name"] = name if name not in (None, "", "None", "Unknown") else None
            t["label"] = self.track_label(pid, num, name)
            t["fixed"] = pid in self.overrides
            t["cuts"] = len(self.track_cuts.get(pid) or [])

    def set_override(self, pid, number, team, remove=False):
        """Set / edit / remove a track's identity, live."""
        num = str(number) if number not in (None, "") else None
        with self.lock:
            if remove:
                self.overrides.pop(pid, None)
            else:
                self._apply_override(pid, num, team)
            self._refresh_active(pid)
            self._answered(pid)
            # The confirmation covers the WHOLE track, including the seconds
            # before the operator got to it — that is the ice time that would
            # otherwise be lost, and the reason presence is derived per track.
            if num is None or remove:
                self._detach(pid)
            else:
                self._attach(pid, (team, str(num)))
        self._maybe_resume()
        return True

    # ---- snapshot for HTTP ------------------------------------------------
    def save_progress(self):
        """Where the save is. The render phase is run by app.finalize_job, which
        reports into its own job record — read it rather than duplicate it."""
        with self.lock:
            sv = dict(self.save)
        if sv["phase"] == "render":
            import app
            j = app.processing_jobs.get(self.job_id) or {}
            sv["done"], sv["total"] = j.get("processed_frames", 0), j.get("total_frames", 0)
            sv["message"] = j.get("message", "")
        return sv

    def state(self):
        sv = self.save_progress()
        with self.lock:
            return {"status": self.status, "frame": self.frame_idx, "total": self.total,
                    "error": self.error, "events": self.events,
                    "overrides": len(self.overrides), "duration": round(self.duration, 2),
                    "now": round(self.now_t, 2),
                    "shifts": self.shifts(), "team_colors": self.team_hex,
                    "check": self.check_stats(),
                    "tracks": list(self.active_tracks),
                    "soon": self.offscreen_tracks(), "lines": self.lines,
                    # the squads: the page offers these numbers and no others
                    "roster": {"Home": dict(self.identifier.Home),
                               "Away": dict(self.identifier.Away)},
                    "save": sv, "pending": sorted(self.pending.values(),
                                                   key=lambda q: -q["toi"]),
                    "review_ready": self.review_ready,
                    "result_ready": self.result_ready, "job_id": self.job_id}
