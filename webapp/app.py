import os
import json
import math
import time
import hashlib
import random
import signal
import sys
import shutil
from threading import Lock, Thread
from collections import defaultdict, namedtuple, Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
RESULTS_FOLDER = BASE_DIR / "results"
MODELS_DIR = PROJECT_DIR / "models"
TRACKERS_DIR = PROJECT_DIR / "trackers"

UPLOAD_FOLDER.mkdir(exist_ok=True)
RESULTS_FOLDER.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2 GB
app.secret_key = "hockey-analytics-secret"

ALLOWED_VIDEO = {"mp4", "avi", "mov", "mkv"}
ALLOWED_EXCEL = {"xlsx", "xls"}

# ---------------------------------------------------------------------------
# State: running jobs
# ---------------------------------------------------------------------------
processing_jobs = {}  # job_id -> {status, progress, message, result_path, ...}

# ---------------------------------------------------------------------------
# Basic colors palette
# ---------------------------------------------------------------------------
BASIC_COLORS = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "gray": (128, 128, 128),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "brown": (165, 42, 42),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def allowed_file(filename, allowed):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed


def collect_existing_stats():
    """Scan project for all player_statistics / Analytics .xlsx files."""
    stats_files = []
    for root, _dirs, files in os.walk(str(PROJECT_DIR)):
        # skip webapp folder itself
        if "webapp" in root:
            continue
        for f in files:
            if f.endswith(".xlsx") and ("player_statistics" in f or "Analytics" in f or "play_by_play" in f):
                stats_files.append(os.path.join(root, f))
    return sorted(stats_files)


def load_stats_from_excel(path):
    """Load a stats Excel into a list of dicts."""
    try:
        df = pd.read_excel(path)
        # Drop duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]
        # Normalise column names
        col_map = {}
        for c in df.columns:
            cl = c.strip().lower()
            if "name" in cl:
                col_map[c] = "Name"
            elif "number" in cl and "unique" not in cl:
                col_map[c] = "Number"
            elif "team" in cl:
                col_map[c] = "Team"
            elif "time" in cl:
                col_map[c] = "Time (s)"
            elif "distance" in cl or "dist" in cl:
                col_map[c] = "Distance (m)"
            elif "unique" in cl or "player id" in cl:
                col_map[c] = "ID"
        df = df.rename(columns=col_map)
        df = df.loc[:, ~df.columns.duplicated()]
        records = df.to_dict("records")
        # clean NaN
        for r in records:
            for k, v in r.items():
                if isinstance(v, float) and math.isnan(v):
                    r[k] = None
        return records
    except Exception as e:
        return []


def load_all_datasets():
    """Return dict: dataset_name -> list of player records."""
    datasets = {}
    files = collect_existing_stats()
    for fp in files:
        rel = os.path.relpath(fp, str(PROJECT_DIR))
        records = load_stats_from_excel(fp)
        if records:
            datasets[rel] = records
    return datasets


def get_team_configs():
    """Find all JSON config files in the project."""
    configs = []
    for root, _dirs, files in os.walk(str(PROJECT_DIR)):
        if "webapp" in root:
            continue
        for f in files:
            if f.endswith("config.json"):
                fp = os.path.join(root, f)
                try:
                    with open(fp, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    configs.append({
                        "path": os.path.relpath(fp, str(PROJECT_DIR)),
                        "data": data,
                    })
                except Exception:
                    pass
    return configs


def aggregate_stats(records):
    """Aggregate player records: group by Name+Team, sum time/distance."""
    agg = {}
    for r in records:
        name = r.get("Name")
        if not name:
            continue
        team = r.get("Team", "Unknown")
        key = (name, team)
        if key not in agg:
            agg[key] = {"Name": name, "Number": r.get("Number"), "Team": team, "Time (s)": 0, "Distance (m)": 0, "appearances": 0}
        agg[key]["Time (s)"] += r.get("Time (s)", 0) or 0
        agg[key]["Distance (m)"] += r.get("Distance (m)", 0) or 0
        agg[key]["appearances"] += 1
        if r.get("Number") and not agg[key]["Number"]:
            agg[key]["Number"] = r.get("Number")
    return sorted(agg.values(), key=lambda x: x["Time (s)"], reverse=True)


# ---------------------------------------------------------------------------
# Player Identifier (adapted from notebook — CPU-friendly fallback)
# ---------------------------------------------------------------------------

class PlayerIdentifier:
    """Identifies player team and jersey number from a bounding-box crop.

    Team classification: hybrid pixel-score approach in HSV space.
    Each team declares 1-2 jersey colors (primary + optional secondary).
    The classifier:
      1. Crops the torso area and converts to HSV
      2. Filters out ice/rink pixels
      3. For each team, computes a weighted pixel-match score:
         each non-ice pixel is scored by proximity to the team's reference
         colors. The team score = mean of per-pixel scores.
      4. The team with higher score wins (with minimum margin).

    This avoids the median-color problem (where secondary elements shift
    the median) and also avoids the old binary pixel-counting problem
    (where any dark-green pixel counted as "black" match).
    Soft scoring means a dark-green pixel scores LOW for black (not 0/1),
    so it doesn't pollute the results nearly as much.
    """

    # How often (in frames) to re-run full identification per player
    ID_INTERVAL = 15

    # Reference HSV values for common jersey colors (OpenCV scale: H 0-179, S 0-255, V 0-255)
    _COLOR_HSV = {
        "red":     (0, 200, 200),     # hue=0 (wraps around 180)
        "blue":    (110, 180, 180),
        "green":   (60, 180, 150),
        "yellow":  (28, 200, 220),
        "orange":  (16, 220, 230),
        "cyan":    (90, 180, 200),
        "magenta": (150, 180, 200),
        "purple":  (135, 150, 150),
        "white":   (0, 0, 230),       # achromatic, high V
        "black":   (0, 0, 35),        # achromatic, low V
        "gray":    (0, 0, 120),       # achromatic, mid V
        "brown":   (15, 150, 100),
    }

    # Saturation threshold: below this, pixel is "achromatic" (white/black/gray)
    _ACHROMATIC_S = 45

    def __init__(self, teams_colors_info, home_squad, away_squad, basic_colors, number_model=None):
        self.teams = teams_colors_info
        self.Home = home_squad
        self.Away = away_squad
        self.BASIC_COLORS = basic_colors
        self.number_model = number_model

        # Cache: player_track_id -> {team, name, number, last_frame}
        self._cache = {}
        # Consecutive team-switch counter per track_id
        self._switch_counts = {}

        # Reference HSV list for each team (1-2 colors)
        home_colors = teams_colors_info.get("Home", ["blue"])
        away_colors = teams_colors_info.get("Away", ["white"])
        self._home_refs = [self._get_reference_hsv(c) for c in home_colors if c]
        self._away_refs = [self._get_reference_hsv(c) for c in away_colors if c]
        if not self._home_refs:
            self._home_refs = [self._get_reference_hsv("blue")]
        if not self._away_refs:
            self._away_refs = [self._get_reference_hsv("white")]

    def _get_reference_hsv(self, color_name):
        """Get reference HSV tuple for a color name."""
        key = color_name.lower().strip()
        if key in self._COLOR_HSV:
            return self._COLOR_HSV[key]
        rgb = self.BASIC_COLORS.get(key)
        if rgb:
            pixel = np.uint8([[list(rgb)]])
            hsv = cv2.cvtColor(pixel, cv2.COLOR_RGB2HSV)[0, 0]
            return (int(hsv[0]), int(hsv[1]), int(hsv[2]))
        return (0, 0, 128)

    @staticmethod
    def _is_achromatic(hsv_ref):
        """Check if a reference color is achromatic (white/black/gray)."""
        return hsv_ref[1] < 45

    @staticmethod
    def _pixel_affinity(h_arr, s_arr, v_arr, ref_h, ref_s, ref_v):
        """Compute per-pixel affinity (0..1) to a reference HSV color.

        Uses soft Gaussian-like scoring instead of hard thresholds.
        Higher affinity = closer to the reference color.
        """
        ref_achromatic = ref_s < 45
        s_f = s_arr.astype(np.float32)
        v_f = v_arr.astype(np.float32)

        if ref_achromatic:
            # For achromatic references (white/black/gray):
            # Score based on brightness distance + saturation must be LOW.
            # A real white/black jersey pixel has S < ~50.
            # Ice pixels also have low S, so the crop + ice filter must handle that.
            v_dist = np.abs(v_f - ref_v) / 255.0
            # Strong saturation penalty: S > 60 → score drops sharply
            s_penalty = np.clip(s_f / 100.0, 0, 1)
            score = np.exp(-4.0 * v_dist) * np.exp(-3.5 * s_penalty)
        else:
            # For chromatic references:
            # Score based on hue distance (circular), saturation match, value match
            dh = np.minimum(
                np.abs(h_arr.astype(np.float32) - ref_h),
                180.0 - np.abs(h_arr.astype(np.float32) - ref_h)
            ) / 90.0
            ds = np.abs(s_f - ref_s) / 255.0
            dv = np.abs(v_f - ref_v) / 255.0
            # Desaturated pixels should NOT match chromatic refs
            # S < 50 → weight near 0; S > 100 → weight = 1
            s_weight = np.clip((s_f - 40.0) / 60.0, 0, 1)
            score = np.exp(-4.0 * dh) * np.exp(-2.0 * ds) * np.exp(-1.5 * dv) * s_weight

        return score

    # ------------------------------------------------------------------
    # Hybrid team classification
    # ------------------------------------------------------------------
    def _classify_team_hsv(self, frame_bgr, x1, y1, x2, y2):
        """Classify team by soft pixel-affinity scoring.

        Each pixel in the jersey crop gets a score (0..1) for each team,
        based on its HSV distance to the team's reference colors.
        For teams with 2 colors, the per-pixel score = max of both colors.
        Team score = mean of all per-pixel scores (ignoring ice).
        Higher score wins. Minimum margin required.

        Returns ('Home'|'Away'|'Unknown', confidence).
        """
        h_box = y2 - y1
        w_box = x2 - x1
        if h_box < 4 or w_box < 4:
            return "Unknown", 0.0

        # Tight torso crop: upper chest / shoulders only.
        # Narrower to avoid ice on sides, shorter to avoid pants/skates.
        cy1 = y1 + int(h_box * 0.12)
        cy2 = y1 + int(h_box * 0.45)
        cx1 = x1 + int(w_box * 0.25)
        cx2 = x1 + int(w_box * 0.75)
        crop = frame_bgr[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return "Unknown", 0.0

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h_ch = hsv[:, :, 0].ravel()
        s_ch = hsv[:, :, 1].ravel()
        v_ch = hsv[:, :, 2].ravel()

        # Filter out ice/rink pixels.
        # If one team is white, use gentle filter (only extremely bright+desaturated).
        # Otherwise, aggressively remove ice-like pixels.
        any_white = any(r[1] < 45 and r[2] > 180 for r in self._home_refs + self._away_refs)
        if any_white:
            # Gentle: only filter the most obvious ice (very low S, very high V)
            is_ice = (s_ch < 25) & (v_ch > 180)
        else:
            # Aggressive: filter anything ice-like
            is_ice = ((s_ch < 50) & (v_ch > 140)) | ((s_ch < 30) & (v_ch > 120))
        not_ice = ~is_ice
        h_f = h_ch[not_ice]
        s_f = s_ch[not_ice]
        v_f = v_ch[not_ice]

        if len(h_f) < 10:
            return "Unknown", 0.0

        # Compute per-pixel affinity for each team (max across team's colors)
        home_score = np.zeros(len(h_f), dtype=np.float32)
        for ref in self._home_refs:
            aff = self._pixel_affinity(h_f, s_f, v_f, ref[0], ref[1], ref[2])
            np.maximum(home_score, aff, out=home_score)

        away_score = np.zeros(len(h_f), dtype=np.float32)
        for ref in self._away_refs:
            aff = self._pixel_affinity(h_f, s_f, v_f, ref[0], ref[1], ref[2])
            np.maximum(away_score, aff, out=away_score)

        home_mean = float(home_score.mean())
        away_mean = float(away_score.mean())

        # Minimum margin to declare a winner
        MIN_MARGIN = 0.03
        diff = abs(home_mean - away_mean)

        if diff < MIN_MARGIN:
            return "Unknown", 0.0

        if home_mean > away_mean:
            return "Home", home_mean
        else:
            return "Away", away_mean

    # ------------------------------------------------------------------
    # Number recognition helpers
    # ------------------------------------------------------------------
    def _parse_number_results(self, results, team, crop_width=None):
        """Parse YOLO number detection with low-conf sweep + roster validation.

        Strategy:
        1. Detect ALL digits at low conf (0.25) to avoid missing faint 2nd digits
        2. De-duplicate overlapping detections of the same digit
        3. Build candidate numbers (single-digit and two-digit)
        4. Validate every candidate against the team roster
        5. Score candidates; if ambiguous, return Unknown (let temporal voting decide)

        This reliably separates single-digit (e.g. 4) from two-digit numbers
        that contain that digit (e.g. 14, 47).
        """
        if not results[0].boxes or len(results[0].boxes) == 0:
            return "Unknown", None

        # --- Extract all detected digits ---
        raw_digits = []
        for box, cls, conf in zip(
            results[0].boxes.xyxy.tolist(),
            results[0].boxes.cls.tolist(),
            results[0].boxes.conf.tolist(),
        ):
            x1, y1, x2, y2 = box
            raw_digits.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "cx": (x1 + x2) / 2, "w": x2 - x1, "h": y2 - y1,
                "digit": int(results[0].names[int(cls)]),
                "conf": conf,
            })
        raw_digits.sort(key=lambda d: d["cx"])

        # --- De-duplicate: merge overlapping boxes of same digit ---
        digits = []
        for d in raw_digits:
            merged = False
            for f in digits:
                overlap_x = max(0, min(d["x2"], f["x2"]) - max(d["x1"], f["x1"]))
                min_w = min(d["w"], f["w"])
                if min_w > 0 and overlap_x / min_w > 0.5:
                    # Overlapping boxes — keep the higher-confidence one
                    if d["conf"] > f["conf"]:
                        f.update(d)
                    merged = True
                    break
            if not merged:
                digits.append(dict(d))
        digits.sort(key=lambda d: d["cx"])

        squad = self.Home if team == "Home" else self.Away
        STRONG_CONF = 0.50  # "confident" detection
        MIN_CONF = 0.30     # minimum to consider at all

        # Filter out very weak noise
        digits = [d for d in digits if d["conf"] >= MIN_CONF]
        if not digits:
            return "Unknown", None

        # --- Build candidates ---
        candidates = []  # list of (number, score, tag)

        # -- Single-digit candidates (from each detected digit) --
        for d in digits:
            num = d["digit"]
            score = d["conf"]
            if crop_width and crop_width > 0:
                rel_cx = d["cx"] / crop_width
                rel_w = d["w"] / crop_width
                # Well-centered digit = likely genuine single-digit number
                if 0.30 <= rel_cx <= 0.70:
                    # Extra boost if it also has reasonable width coverage
                    # A true single-digit jersey fills ~25-50% of crop width
                    if rel_w >= 0.18:
                        candidates.append((num, score * 1.0, "single_centered"))
                    else:
                        candidates.append((num, score * 0.7, "single_centered_narrow"))
                else:
                    # Off-center: almost certainly part of a 2-digit number
                    candidates.append((num, score * 0.15, "single_offcenter"))
            else:
                candidates.append((num, score * 0.6, "single_no_crop"))

        # -- Two-digit candidates (from pairs) --
        if len(digits) >= 2:
            # Try all pairs, prioritize by combined confidence
            for i in range(len(digits)):
                for j in range(i + 1, len(digits)):
                    d0, d1 = digits[i], digits[j]
                    # d0 should be left, d1 right (already sorted by cx)
                    gap = d1["x1"] - d0["x2"]
                    avg_w = (d0["w"] + d1["w"]) / 2

                    # Validate spatial relationship
                    if gap < -avg_w * 0.6:
                        continue  # too much overlap = same digit detected twice
                    if crop_width and gap > crop_width * 0.4:
                        continue  # too far apart = unrelated detections

                    two_digit = d0["digit"] * 10 + d1["digit"]
                    min_conf = min(d0["conf"], d1["conf"])
                    avg_conf = (d0["conf"] + d1["conf"]) / 2

                    # Score: reward when both digits are strong
                    if min_conf >= STRONG_CONF:
                        pair_score = avg_conf * 1.2  # both strong — high confidence
                    elif min_conf >= MIN_CONF:
                        pair_score = avg_conf * 0.8  # one weak but present
                    else:
                        pair_score = avg_conf * 0.4

                    # Spatial bonus: two digits should span more of the crop
                    if crop_width and crop_width > 0:
                        span = d1["x2"] - d0["x1"]
                        rel_span = span / crop_width
                        if rel_span >= 0.35:
                            pair_score *= 1.1  # good coverage for 2 digits
                        center = (d0["cx"] + d1["cx"]) / 2
                        rel_center = center / crop_width
                        if 0.30 <= rel_center <= 0.70:
                            pair_score *= 1.1  # pair is centered

                    candidates.append((two_digit, pair_score, "pair"))

        if not candidates:
            return "Unknown", None

        # --- Roster validation: boost roster hits, crush non-roster ---
        scored = []
        for num, score, tag in candidates:
            if str(num) in squad:
                scored.append((num, score * 2.0, tag))  # big boost
            else:
                scored.append((num, score * 0.05, tag))  # effectively eliminated

        scored.sort(key=lambda x: -x[1])
        best_num, best_score, best_tag = scored[0]

        # Minimum quality threshold
        if best_score < 0.25:
            return "Unknown", None

        # Ambiguity check: if runner-up is close, we're not confident
        if len(scored) > 1:
            runner_up_score = scored[1][1]
            if runner_up_score > best_score * 0.65:
                return "Unknown", None

        return squad.get(str(best_num), "Unknown"), best_num

    # ------------------------------------------------------------------
    # Batch identification for multiple players
    # ------------------------------------------------------------------
    def identify_batch(self, frame_bgr, detections, frame_num):
        """Identify players in batch.

        Returns (results, team_switches):
            results: dict track_id -> (name, number, team)
            team_switches: set of track_ids where a cross-team jump was detected

        detections: list of (track_id, x1, y1, x2, y2)
        Only runs full identification every ID_INTERVAL frames per player.
        Number recognition is batched into a single YOLO call.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        results = {}
        team_switches = set()
        needs_number = []  # (track_id, crop_rgb, team)

        for pid, x1, y1, x2, y2 in detections:
            # --- ALWAYS classify team (HSV is ~0.06ms, cheap enough) ---
            team, confidence = self._classify_team_hsv(frame_bgr, x1, y1, x2, y2)
            if team == "Unknown":
                old = self._cache.get(pid, {})
                team = old.get("team", "Unknown")

            cached = self._cache.get(pid)

            # Detect team switch — tracker jumped to a different player
            # Require 3 consecutive mismatches to confirm a real switch
            # (avoids splitting on single-frame classification flickers)
            team_switched = False
            if cached and cached.get("team") and cached["team"] != "Unknown":
                if team and team != "Unknown" and team != cached["team"]:
                    if confidence and confidence > 0.20:
                        cnt = self._switch_counts.get(pid, 0) + 1
                        self._switch_counts[pid] = cnt
                        if cnt >= 3:
                            team_switched = True
                            self._switch_counts.pop(pid, None)
                    else:
                        self._switch_counts.pop(pid, None)
                else:
                    # Teams match — reset counter
                    self._switch_counts.pop(pid, None)

            # Use cache if still fresh AND team didn't switch
            if cached and not team_switched and (frame_num - cached["last_frame"]) < self.ID_INTERVAL:
                results[pid] = (cached["name"], cached["number"], team)
                continue

            # If team switched, invalidate cache — force full re-identification
            if team_switched:
                team_switches.add(pid)
                self._cache.pop(pid, None)
                cached = None

            # Queue for batch number recognition
            crop = frame_rgb[y1:y2, x1:x2]
            if crop.size > 0 and self.number_model is not None:
                needs_number.append((pid, crop, team))
            else:
                old = self._cache.get(pid, {}) if cached else {}
                name = old.get("name", "Unknown")
                number = old.get("number")
                results[pid] = (name, number, team)
                self._cache[pid] = {"name": name, "number": number, "team": team, "last_frame": frame_num}

        # --- Batch number recognition ---
        if needs_number:
            pil_crops = [Image.fromarray(c) for c in [item[1] for item in needs_number]]
            batch_results = self.number_model(pil_crops, conf=0.25, max_det=4, imgsz=256, verbose=False)

            for i, (pid, crop, team) in enumerate(needs_number):
                name, number = self._parse_number_results([batch_results[i]], team, crop_width=crop.shape[1])
                results[pid] = (name, number, team)
                self._cache[pid] = {"name": name, "number": number, "team": team, "last_frame": frame_num}

        return results, team_switches

    # ------------------------------------------------------------------
    # Single-player fallback (kept for compatibility)
    # ------------------------------------------------------------------
    def player_identification(self, frame, x1, y1, x2, y2):
        team, _ = self._classify_team_hsv(frame, x1, y1, x2, y2)

        player, number = "Unknown", None
        if self.number_model is not None:
            crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            if crop.size > 0:
                pil_crop = Image.fromarray(crop)
                res = self.number_model(pil_crop, conf=0.25, max_det=4, imgsz=256, verbose=False)
                player, number = self._parse_number_results(res, team, crop_width=crop.shape[1])
        return player, number, team


# ---------------------------------------------------------------------------
# Track Merger — resolves ID switches by merging fragmented tracks
# ---------------------------------------------------------------------------

class TrackMerger:
    """Merges fragmented tracker IDs into consistent player identities.

    Two-pass strategy:
    Pass 1 (safe): merge tracks sharing the same team + jersey number.
    Pass 2 (spatial): merge tracks with same team, close in space/time,
           no temporal overlap — but verify group integrity at each step.
    """

    MAX_GAP_SEC = 1.5        # tighter: was 3.0
    MAX_SPATIAL_DIST = 80.0   # tighter: was 150.0
    MIN_TRACK_SEC = 0.3       # ignore micro-tracks in spatial merge

    def __init__(self, fps):
        self.fps = fps

    def merge(self, players_data):
        if not players_data:
            return players_data, {}

        # Build track summaries with temporal spans
        tracks = {}
        for pid, data in players_data.items():
            toi = data.get("time_on_ice", 0)
            dist = data.get("total_distance", 0)
            # Flag stationary tracks (likely goalie): low speed + significant TOI
            avg_speed = (dist / toi) * 3.6 if toi > 0 else 0  # km/h
            tracks[pid] = {
                "team": self._majority(data.get("player_team", [])),
                "number": self.resolve_number(data.get("player_number", [])),
                "first_seen": data.get("first_seen"),
                "last_seen": data.get("last_seen"),
                "first_position": data.get("first_position"),
                "last_position": data.get("last_position"),
                "time_on_ice": toi,
                "total_distance": dist,
                "is_stationary": toi > 5 and avg_speed < 2.0,
            }

        # Groups: list of sets of pids
        # Start with each pid in its own group
        groups = {pid: {pid} for pid in tracks}  # pid -> group_set

        def group_of(pid):
            return groups[pid]

        def can_add_to_group(group, candidate_pid):
            """Check that candidate doesn't temporally overlap any member."""
            ct = tracks[candidate_pid]
            for member in group:
                if self._overlaps(tracks[member], ct):
                    return False
            return True

        def merge_groups(pid_a, pid_b):
            ga, gb = groups[pid_a], groups[pid_b]
            if ga is gb:
                return True
            # Verify no overlaps between all members
            for pa in ga:
                for pb in gb:
                    if self._overlaps(tracks[pa], tracks[pb]):
                        return False
            # Safe to merge
            merged = ga | gb
            for p in merged:
                groups[p] = merged
            return True

        pids = sorted(tracks.keys())

        # --- Pass 1: merge by same team + same number (strongest signal) ---
        by_team_number = defaultdict(list)
        for pid in pids:
            t = tracks[pid]
            if t["team"] and t["number"] is not None and not t["is_stationary"]:
                by_team_number[(t["team"], t["number"])].append(pid)

        for (team, number), members in by_team_number.items():
            members.sort(key=lambda p: tracks[p].get("first_seen") or 0)
            for i in range(len(members) - 1):
                merge_groups(members[i], members[i + 1])

        # --- Pass 2: spatial-temporal proximity (same team, no conflicting numbers) ---
        # Only merge when a track ends and another begins very close by.
        # STRICT: check that the candidate's detected numbers don't conflict
        # with the existing group's consensus number.

        def group_number(pid):
            """Get consensus number for pid's current group."""
            g = groups[pid]
            nums = []
            for p in g:
                if tracks[p]["number"] is not None:
                    nums.append(tracks[p]["number"])
            if not nums:
                return None
            c = Counter(nums)
            best, cnt = c.most_common(1)[0]
            return best if cnt >= len(nums) * 0.5 else None

        pids_by_time = sorted(pids, key=lambda p: tracks[p].get("first_seen") or 0)

        for i, pid_a in enumerate(pids_by_time):
            ta = tracks[pid_a]
            if ta["last_seen"] is None:
                continue
            # Skip micro-tracks as source (they cause chain bridges)
            if ta["time_on_ice"] < self.MIN_TRACK_SEC:
                continue
            # Don't merge stationary (goalie) tracks with others
            if ta["is_stationary"]:
                continue

            for pid_b in pids_by_time[i + 1:]:
                tb = tracks[pid_b]
                if tb["first_seen"] is None:
                    continue
                # Skip micro-tracks as targets too
                if tb["time_on_ice"] < self.MIN_TRACK_SEC:
                    continue
                # Don't merge stationary (goalie) tracks with others
                if tb["is_stationary"]:
                    continue

                gap = tb["first_seen"] - ta["last_seen"]
                if gap > self.MAX_GAP_SEC:
                    break
                if gap < 0:
                    continue

                # Same team required
                if not (ta["team"] and tb["team"] and ta["team"] == tb["team"]):
                    continue
                if ta["team"] == "Unknown":
                    continue

                # Spatial proximity
                ep = ta.get("last_position")
                fp = tb.get("first_position")
                if not ep or not fp:
                    continue
                dist = math.sqrt((ep[0] - fp[0]) ** 2 + (ep[1] - fp[1]) ** 2)
                if dist > self.MAX_SPATIAL_DIST:
                    continue

                # STRICT number conflict check:
                # If either track has a detected number, it must not conflict
                # with the other track's number OR the other group's consensus.
                if ta["number"] is not None and tb["number"] is not None:
                    if ta["number"] != tb["number"]:
                        continue
                # Check against group consensus
                gn_a = group_number(pid_a)
                gn_b = group_number(pid_b)
                if gn_a is not None and tb["number"] is not None and gn_a != tb["number"]:
                    continue
                if gn_b is not None and ta["number"] is not None and gn_b != ta["number"]:
                    continue
                if gn_a is not None and gn_b is not None and gn_a != gn_b:
                    continue

                merge_groups(pid_a, pid_b)

        # --- Pass 3: deduplicate same team+number across remaining groups ---
        # Use RAW number votes from players_data (not strict majority) so that
        # ambiguous tracks (e.g. 50/50 #4/#99) can still be assigned to the
        # correct group when a dominant identity exists.
        group_identity = {}  # (team, number) -> representative pid

        for pid in pids:
            g = groups[pid]
            # Collect ALL raw number/team votes from original players_data
            team_votes = []
            num_votes = []
            for p in g:
                src = players_data.get(p, {})
                team_votes.extend([v for v in src.get("player_team", []) if v and v != "Unknown"])
                num_votes.extend([v for v in src.get("player_number", []) if v is not None])

            if not team_votes:
                continue
            team = Counter(team_votes).most_common(1)[0][0]
            if team == "Unknown":
                continue

            if not num_votes:
                continue
            # Use confusion-aware resolution (handles 4 vs 14/47 etc.)
            number = self.resolve_number(num_votes, min_count=2)
            if number is None:
                continue

            key = (team, number)
            if key in group_identity:
                other_group = groups[group_identity[key]]
                if g is not other_group:
                    merged_g = g | other_group
                    for p in merged_g:
                        groups[p] = merged_g
            else:
                group_identity[key] = pid

        # Collect unique groups
        seen_groups = []
        seen_ids = set()
        for pid in pids:
            gid = id(groups[pid])
            if gid not in seen_ids:
                seen_ids.add(gid)
                seen_groups.append(groups[pid])

        # Build merged data
        merged_data = {}
        pid_mapping = {}
        for group in seen_groups:
            # Canonical = member with most time_on_ice
            canonical = max(group, key=lambda p: tracks[p]["time_on_ice"])
            merged = {
                "time_on_ice": 0, "last_seen": None, "last_position": None,
                "first_seen": None, "first_position": None, "total_distance": 0,
                "player_name": [], "player_number": [], "player_team": [], "frames": [],
            }
            for pid in sorted(group, key=lambda p: tracks[p].get("first_seen") or float("inf")):
                pid_mapping[pid] = canonical
                src = players_data[pid]
                merged["player_name"].extend(src.get("player_name", []))
                merged["player_number"].extend(src.get("player_number", []))
                merged["player_team"].extend(src.get("player_team", []))
                merged["frames"].extend(src.get("frames", []))
                merged["time_on_ice"] += src.get("time_on_ice", 0)
                merged["total_distance"] += src.get("total_distance", 0)
                fs = src.get("first_seen")
                ls = src.get("last_seen")
                if fs is not None and (merged["first_seen"] is None or fs < merged["first_seen"]):
                    merged["first_seen"] = fs
                    merged["first_position"] = src.get("first_position")
                if ls is not None and (merged["last_seen"] is None or ls > merged["last_seen"]):
                    merged["last_seen"] = ls
                    merged["last_position"] = src.get("last_position")
            merged_data[canonical] = merged

        return merged_data, pid_mapping

    @staticmethod
    def resolve_number(num_votes, min_count=2):
        """Confusion-aware number resolution.

        Key insight: missing a digit (seeing "4" when jersey is #14) is
        MUCH more common than hallucinating an extra digit (seeing "14"
        when jersey is #4).  Therefore, even a minority of two-digit votes
        is strong evidence that the real number is two-digit.

        Algorithm:
        1. For each single-digit N in the vote set, find two-digit numbers
           in the same vote set whose tens or units digit equals N.
        2. If such two-digit votes exist (>= 2 votes AND >= 10% of the
           single-digit count), redistribute the single-digit votes to the
           best matching two-digit candidate.
        3. Pick the overall winner from the adjusted counts.
        """
        filtered = [v for v in num_votes if v is not None]
        if not filtered:
            return None
        counts = Counter(filtered)

        # Single number, no confusion possible
        if len(counts) == 1:
            num, cnt = counts.most_common(1)[0]
            return num if cnt >= min_count else None

        adjusted = Counter(counts)

        for num in list(counts.keys()):
            if not (0 <= num <= 9):
                continue
            # Find two-digit numbers in the votes that contain this digit
            related = {n: c for n, c in counts.items()
                       if 10 <= n <= 99 and (n // 10 == num or n % 10 == num)}
            if not related:
                continue

            total_related = sum(related.values())
            single_count = counts[num]

            # Even a modest number of two-digit votes is significant:
            # >=2 occurrences AND >=10 % of single-digit votes.
            if total_related >= 2 and total_related >= single_count * 0.10:
                best_related = max(related, key=related.get)
                # Absorb single-digit votes into the best two-digit match
                adjusted[best_related] += adjusted[num]
                del adjusted[num]

        if not adjusted:
            return None
        best, best_count = adjusted.most_common(1)[0]
        if best_count < min_count:
            return None
        # Ambiguity check: winner must be clearly ahead
        if len(adjusted) >= 2:
            _, second_count = adjusted.most_common(2)[1]
            if best_count < second_count * 1.5:
                return None
        return best

    def _majority(self, votes, min_ratio=0.5, min_count=2):
        """Return majority element with strict thresholds (for team votes)."""
        filtered = [v for v in votes if v is not None and v != "Unknown"]
        if not filtered:
            return None
        counts = Counter(filtered)
        best, best_count = counts.most_common(1)[0]
        if best_count < min_count:
            return None
        if best_count / len(filtered) < min_ratio:
            return None
        if len(counts) >= 2:
            _, second_count = counts.most_common(2)[1]
            if best_count < second_count * 2:
                return None
        return best

    def _overlaps(self, ta, tb):
        a_start = ta.get("first_seen")
        a_end = ta.get("last_seen")
        b_start = tb.get("first_seen")
        b_end = tb.get("last_seen")
        if None in (a_start, a_end, b_start, b_end):
            return False
        # Small tolerance (0.5s) for near-simultaneous appearance
        return a_start < (b_end - 0.5) and b_start < (a_end - 0.5)


# ---------------------------------------------------------------------------
# Video Processing Job (background thread)
# ---------------------------------------------------------------------------

def process_video_job(job_id, video_path, config_data):
    """Run full detection + tracking pipeline in background."""
    job = processing_jobs[job_id]
    job["status"] = "running"
    job["message"] = "Загрузка моделей..."

    try:
        from ultralytics import YOLO
    except ImportError:
        job["status"] = "error"
        job["message"] = "ultralytics не установлен. Установите: pip install ultralytics"
        return

    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    except ImportError:
        device = "cpu"

    try:
        # Model paths — prefer CoreML (.mlpackage) for Apple Silicon, fallback to PyTorch
        detection_coreml = str(MODELS_DIR / "Player_Detection.mlpackage")
        detection_pt = str(MODELS_DIR / "Player_Detection.pt")
        detection_path = detection_coreml if os.path.exists(detection_coreml) and device == "mps" else detection_pt
        segmentation_path = str(MODELS_DIR / "LinesDetection.pt")
        number_path = str(MODELS_DIR / "NumbersRec.pt")
        # Prefer fast tracker (ByteTrack, no ReID) if available
        tracker_fast = str(TRACKERS_DIR / "tracker_fast.yaml")
        tracker_orig = str(TRACKERS_DIR / "tracker.yaml")
        tracker_path = tracker_fast if os.path.exists(tracker_fast) else tracker_orig

        for p, label in [(detection_path, os.path.basename(detection_path)), (segmentation_path, "LinesDetection.pt"),
                         (number_path, "NumbersRec.pt"), (tracker_path, os.path.basename(tracker_path))]:
            if not os.path.exists(p):
                job["status"] = "error"
                job["message"] = f"Модель не найдена: {label}"
                return

        use_coreml = detection_path.endswith(".mlpackage")
        if use_coreml:
            model_det = YOLO(detection_path, task="detect")
        else:
            model_det = YOLO(detection_path).to(device)
        model_seg = YOLO(segmentation_path).to(device)
        model_num = YOLO(number_path).to(device)

        # Enable FP16 inference on CUDA for speedup
        use_half = device == "cuda"
        if use_half:
            model_det.model.half()
            model_num.model.half()

        print(f"[Job {job_id}] Device: {device}, CoreML det: {use_coreml}, FP16: {use_half}", flush=True)

        # Parse config
        teams_colors_info = config_data.get("teams_colors_info", {"Home": ["blue"], "Away": ["white", "red"]})
        home_squad = config_data.get("home_team", {})
        away_squad = config_data.get("away_team", {})
        # Goalie numbers to exclude from field player results
        goalies_home = set(str(n) for n in config_data.get("goalies_home", []))
        goalies_away = set(str(n) for n in config_data.get("goalies_away", []))
        basic_colors = {}
        for name, rgb in config_data.get("basic_colors", BASIC_COLORS).items():
            if isinstance(rgb, list):
                basic_colors[name] = tuple(rgb)
            else:
                basic_colors[name] = rgb
        if not basic_colors:
            basic_colors = BASIC_COLORS

        identifier = PlayerIdentifier(teams_colors_info, home_squad, away_squad, basic_colors, model_num)

        # Build line-mate lookup from config: number -> set of line-mate numbers
        # Used for contextual disambiguation (e.g., 7 vs 47)
        line_mates = {}  # int number -> set of int numbers (line-mates)
        for key in ("home_defense_pairs", "away_defense_pairs",
                     "home_forward_lines", "away_forward_lines"):
            for group in config_data.get(key, []):
                nums = [int(n) for n in group if n]
                for n in nums:
                    if n not in line_mates:
                        line_mates[n] = set()
                    line_mates[n].update(nums)
                    line_mates[n].discard(n)  # exclude self

        # Result folder for this job
        result_dir = RESULTS_FOLDER / job_id
        result_dir.mkdir(exist_ok=True)
        videos_dir = result_dir / "player_videos"
        videos_dir.mkdir(exist_ok=True)

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            job["status"] = "error"
            job["message"] = "Не удалось открыть видеофайл"
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        job["message"] = "Калибровка по линиям разметки..."
        job["total_frames"] = total_frames

        # Calibration: try multiple frames to find >= 2 line markings
        # KHL rink: 60m x 30m. Blue lines are ~17.2m apart.
        pixels_to_meters = None
        calibration_frames = [0, 30, 60, 150, 300, 600, 900, 1200, 1800]
        for cal_pos in calibration_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, cal_pos)
            ret, frame = cap.read()
            if not ret:
                continue
            results = model_seg(frame, verbose=False)
            for result in results:
                if result.masks and len(result.masks) >= 2:
                    line_centers = [mask.xy[0].mean(axis=0) for mask in result.masks]
                    # Sort by x-coordinate
                    line_centers.sort(key=lambda c: c[0])
                    # Use the two most separated lines
                    x1, y1 = line_centers[0]
                    x2, y2 = line_centers[-1]
                    distance_pixels = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    if distance_pixels > 50:
                        # Known distance between blue lines in KHL = 17.2m
                        pixels_to_meters = 17.2 / distance_pixels
                        break
            if pixels_to_meters is not None:
                break

        if pixels_to_meters is None:
            # Fallback: typical broadcast framing covers ~40m of rink width
            pixels_to_meters = 40.0 / width

        # Perspective correction: broadcast cameras are elevated ~10-15m at center ice.
        # Objects at the bottom of the frame (near side) are closer and appear larger,
        # so fewer real meters per pixel. Objects at top (far side) are farther away.
        # Linear model: scale(y) = base * (1 + perspective_k * (cy_norm - 0.5))
        # where cy_norm = y / height, ranging 0 (top) to 1 (bottom).
        # Positive k means top of frame = more meters/px, bottom = fewer meters/px.
        PERSPECTIVE_K = 0.6  # empirical for typical broadcast angle (~15-20 deg)
        MAX_PLAYER_SPEED_MS = 13.0  # 46.8 km/h — absolute physical cap for hockey
        SPEED_WINDOW_FRAMES = max(int(fps * 0.5), 5)  # 0.5 sec sliding window

        def perspective_scale(cy):
            """Return pixels_to_meters adjusted for vertical position in frame."""
            cy_norm = cy / height  # 0=top (far), 1=bottom (near)
            return pixels_to_meters * (1.0 + PERSPECTIVE_K * (0.5 - cy_norm))

        # Player data
        players_data = defaultdict(lambda: {
            "time_on_ice": 0, "last_seen": None, "last_position": None,
            "first_seen": None, "first_position": None,
            "total_distance": 0, "max_speed_ms": 0,
            "speed_buffer": [],  # sliding window of (dist_m, dt) pairs
            "player_name": [], "player_number": [],
            "player_team": [], "frames": [],
        })
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writers = {}

        # Track splitting: when a cross-team jump is detected, we create a
        # new virtual PID so the old track data stays clean.
        pid_remap = {}        # tracker_pid -> current virtual_pid
        next_virtual_pid = 100000  # virtual PIDs start high to avoid collisions

        # --- Speed optimization ---
        # DET_STRIDE: full identification every Nth frame (saves ~10ms on non-ID frames).
        # All frames are processed through YOLO tracker for tracking continuity.
        # CoreML on Apple Neural Engine provides ~2x speedup over MPS PyTorch.
        FRAME_SKIP = 1    # process every frame (full quality clips)
        DET_STRIDE = 3    # identify every 3rd frame
        DET_IMGSZ = 640   # full resolution
        clip_fps = fps     # output clips at original frame rate

        # --- Async video writing: offload I/O to a background thread ---
        import queue as _queue
        write_queue = _queue.Queue(maxsize=200)
        _write_stop = False

        def _writer_thread():
            while True:
                try:
                    item = write_queue.get(timeout=1.0)
                except _queue.Empty:
                    if _write_stop:
                        break
                    continue
                if item is None:
                    break
                vpid, frame_data, box = item
                if vpid not in writers:
                    vp = str(videos_dir / f"player_{vpid}.mp4")
                    writers[vpid] = cv2.VideoWriter(vp, fourcc, clip_fps, (width, height))
                fc = frame_data.copy()
                cv2.rectangle(fc, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                writers[vpid].write(fc)

        wt = Thread(target=_writer_thread, daemon=True)
        wt.start()

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_count = 0
        job["message"] = "Обработка видео..."
        t_start = time.time()

        last_detections = []    # cached detections from last YOLO run
        last_id_results = {}    # cached identification results
        _prof = {"track": 0, "ident": 0, "data": 0, "n": 0}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            current_time = frame_count / fps

            if frame_count % 30 == 0:
                job["progress"] = round(frame_count / max(total_frames, 1) * 100, 1)
                job["processed_frames"] = frame_count

            is_det_frame = (frame_count % DET_STRIDE == 1) or DET_STRIDE == 1

            _t0 = time.time()
            results = model_det.track(
                frame, conf=0.6, persist=True, tracker=tracker_path,
                max_det=16, verbose=False, imgsz=DET_IMGSZ,
            )
            _t1 = time.time()

            # Collect detections
            detections = []
            for result in results:
                for box in result.boxes:
                    if int(box.cls) == 0 and box.id is not None:
                        pid = int(box.id)
                        bx1, by1, bx2, by2 = [int(v) for v in box.xyxy[0].tolist()]
                        detections.append((pid, bx1, by1, bx2, by2))

            if detections:
                # Full identification only on detection frames; otherwise reuse cache
                if is_det_frame:
                    id_results, team_switches = identifier.identify_batch(frame, detections, frame_count)
                    last_id_results = id_results

                    for raw_pid in team_switches:
                        old_vpid = pid_remap.get(raw_pid, raw_pid)
                        if old_vpid in writers:
                            writers[old_vpid].release()
                            del writers[old_vpid]
                        pid_remap[raw_pid] = next_virtual_pid
                        next_virtual_pid += 1
                else:
                    id_results = {}
                    for pid, bx1, by1, bx2, by2 in detections:
                        cached = identifier._cache.get(pid)
                        if cached:
                            id_results[pid] = (cached["name"], cached["number"], cached["team"])
                        elif pid in last_id_results:
                            id_results[pid] = last_id_results[pid]
                        else:
                            id_results[pid] = ("Unknown", None, "Unknown")
                _t2 = time.time()

                # Camera motion compensation
                cam_dx, cam_dy = 0.0, 0.0
                if len(detections) >= 3:
                    deltas = []
                    for pid, bx1, by1, bx2, by2 in detections:
                        vpid = pid_remap.get(pid, pid)
                        pd_entry = players_data[vpid]
                        if pd_entry["last_position"] is not None:
                            cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
                            lx, ly = pd_entry["last_position"]
                            deltas.append((cx - lx, cy - ly))
                    if len(deltas) >= 3:
                        cam_dx = float(np.median([d[0] for d in deltas]))
                        cam_dy = float(np.median([d[1] for d in deltas]))

                for pid, bx1, by1, bx2, by2 in detections:
                    name, number, team = id_results.get(pid, ("Unknown", None, "Unknown"))

                    vpid = pid_remap.get(pid, pid)
                    pd_entry = players_data[vpid]

                    if is_det_frame:
                        if name not in (None, "None", "Unknown"):
                            pd_entry["player_name"].append(name)
                            pd_entry["player_number"].append(number)
                        if team:
                            pd_entry["player_team"].append(team)

                    cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
                    if pd_entry["first_seen"] is None:
                        pd_entry["first_seen"] = current_time
                        pd_entry["first_position"] = (cx, cy)
                    if pd_entry["last_seen"] is not None:
                        td = current_time - pd_entry["last_seen"]
                        pd_entry["time_on_ice"] += td
                        if pd_entry["last_position"] and td > 0:
                            lx, ly = pd_entry["last_position"]
                            real_dx = (cx - lx) - cam_dx
                            real_dy = (cy - ly) - cam_dy
                            dist_px = math.sqrt(real_dx ** 2 + real_dy ** 2)
                            # Perspective-corrected scale at midpoint Y
                            mid_y = (cy + ly) / 2
                            dist_m = dist_px * perspective_scale(mid_y)
                            # Reject physically impossible jumps (detection glitch)
                            inst_speed = dist_m / td
                            if inst_speed <= MAX_PLAYER_SPEED_MS:
                                pd_entry["total_distance"] += dist_m
                                # Sliding window max speed (smoothed over 0.5s)
                                buf = pd_entry["speed_buffer"]
                                buf.append((dist_m, td))
                                if len(buf) > SPEED_WINDOW_FRAMES:
                                    buf.pop(0)
                                if len(buf) >= 3:
                                    win_dist = sum(b[0] for b in buf)
                                    win_dt = sum(b[1] for b in buf)
                                    if win_dt > 0:
                                        win_speed = win_dist / win_dt
                                        if win_speed > pd_entry["max_speed_ms"]:
                                            pd_entry["max_speed_ms"] = win_speed

                    pd_entry["last_seen"] = current_time
                    pd_entry["last_position"] = (cx, cy)

                    # Async video write
                    try:
                        write_queue.put_nowait((vpid, frame, (bx1, by1, bx2, by2)))
                    except _queue.Full:
                        pass
                _t3 = time.time()

                _prof["track"] += (_t1 - _t0)
                _prof["ident"] += (_t2 - _t1)
                _prof["data"] += (_t3 - _t2)
                _prof["n"] += 1

        cap.release()

        # Signal writer thread to finish and wait
        _write_stop = True
        write_queue.put(None)
        wt.join(timeout=30)
        for w in writers.values():
            w.release()

        t_elapsed = time.time() - t_start
        # FPS relative to original video frames (not just processed frames)
        processing_fps = frame_count / max(t_elapsed, 0.1)
        job["processing_fps"] = round(processing_fps, 1)
        job["processing_time"] = round(t_elapsed, 1)
        profile_info = None
        if _prof["n"] > 0:
            n = _prof["n"]
            profile_info = {
                "yolo_track_ms": round(_prof["track"] / n * 1000, 1),
                "identify_ms": round(_prof["ident"] / n * 1000, 1),
                "data_write_ms": round(_prof["data"] / n * 1000, 1),
                "overhead_s": round(t_elapsed - _prof["track"] - _prof["ident"] - _prof["data"], 1),
                "processed_frames": frame_count,
            }
        job["profile"] = profile_info
        if profile_info:
            print(f"[Job {job_id}] {frame_count} frames in {t_elapsed:.1f}s = {processing_fps:.1f} FPS | "
                  f"track={profile_info['yolo_track_ms']}ms ident={profile_info['identify_ms']}ms "
                  f"data={profile_info['data_write_ms']}ms", flush=True)

        # Merge fragmented tracks
        job["message"] = "Слияние фрагментированных треков..."
        merger = TrackMerger(fps)
        merged_data, pid_mapping = merger.merge(dict(players_data))

        n_before = len(players_data)
        n_after = len(merged_data)
        job["message"] = f"Слияние: {n_before} треков -> {n_after} игроков"

        # Most likely value helper (for team/name — simple majority)
        def most_likely(values, threshold=0.4):
            filtered = [v for v in values if v is not None]
            if not filtered:
                return None
            counts = Counter(filtered)
            best, best_count = counts.most_common(1)[0]
            return best if best_count / len(filtered) >= threshold else None

        # Build reverse lookup: name -> (team, number)
        name_to_identity = {}
        for num, nm in home_squad.items():
            name_to_identity[nm] = ("Home", int(num))
        for num, nm in away_squad.items():
            name_to_identity[nm] = ("Away", int(num))

        # Resolve identity for each merged player
        player_info = {}  # canonical_pid -> {name, number, team}
        for pid, stats in merged_data.items():
            team = most_likely(stats["player_team"])
            # Use confusion-aware resolution for numbers (handles 4 vs 14/47 etc.)
            number = TrackMerger.resolve_number(stats["player_number"])
            name = None
            if number is not None and team:
                squad = home_squad if team == "Home" else away_squad
                resolved = squad.get(str(number))
                if resolved:
                    name = resolved
            if not name:
                name = most_likely(stats["player_name"])

            # --- Cross-validate against roster ---
            # If name is known, enforce correct team and number from roster
            if name and name in name_to_identity:
                correct_team, correct_number = name_to_identity[name]
                team = correct_team
                number = correct_number
            # If name unknown but number+team are known, look up name
            elif number is not None and team:
                squad = home_squad if team == "Home" else away_squad
                name = squad.get(str(number))
                # Check other roster too (in case team vote is wrong)
                if not name:
                    other = away_squad if team == "Home" else home_squad
                    other_name = other.get(str(number))
                    if other_name:
                        name = other_name
                        team = "Away" if team == "Home" else "Home"

            player_info[pid] = {"name": name, "number": number, "team": team}

        # --- Second pass: line-context disambiguation ---
        # For players whose number is ambiguous (confusable digits, e.g. 7 vs 47),
        # check which line-mates were on the ice at the same time.
        # If confident players share a line with one candidate but not another,
        # re-assign to the line-mate candidate.
        if line_mates:
            # Build temporal intervals for each pid
            intervals = {}
            for pid, stats in merged_data.items():
                fs = stats.get("first_seen")
                ls = stats.get("last_seen")
                if fs is not None and ls is not None:
                    intervals[pid] = (fs, ls)

            # Find confident players (have name + team + number)
            confident_pids = {
                pid for pid, info in player_info.items()
                if info["name"] and info["number"] is not None and info["team"]
            }

            def overlaps(pid_a, pid_b, min_overlap=1.0):
                """Check if two players' time on ice overlaps by at least min_overlap seconds."""
                iv_a = intervals.get(pid_a)
                iv_b = intervals.get(pid_b)
                if not iv_a or not iv_b:
                    return False
                start = max(iv_a[0], iv_b[0])
                end = min(iv_a[1], iv_b[1])
                return (end - start) >= min_overlap

            for pid, stats in merged_data.items():
                info = player_info[pid]
                if info["name"] and info["number"] is not None:
                    continue  # already resolved confidently
                if not info["team"]:
                    continue

                # Get the raw number votes for this track
                raw_votes = stats.get("player_number", [])
                filtered_votes = [v for v in raw_votes if v is not None]
                if not filtered_votes:
                    continue

                vote_counts = Counter(filtered_votes)
                candidates = list(vote_counts.keys())

                # Build all plausible candidates (including confusion expansions)
                # e.g., votes for 7 could mean 7, 17, 27, 47, 70, 71, ...
                squad = home_squad if info["team"] == "Home" else away_squad
                plausible = set()
                for c in candidates:
                    plausible.add(c)
                    # If single-digit, also consider two-digit roster numbers containing it
                    if 0 <= c <= 9:
                        for roster_num_str in squad:
                            rn = int(roster_num_str)
                            if 10 <= rn <= 99 and (rn // 10 == c or rn % 10 == c):
                                plausible.add(rn)

                # Only proceed if there are multiple plausible candidates
                if len(plausible) < 2:
                    continue

                # Find confident co-occurring players from the same team
                co_occurring_numbers = set()
                for cpid in confident_pids:
                    ci = player_info[cpid]
                    if ci["team"] == info["team"] and overlaps(pid, cpid):
                        co_occurring_numbers.add(ci["number"])

                if not co_occurring_numbers:
                    continue

                # Score each candidate by how many of its line-mates are on the ice
                best_candidate = None
                best_score = -1
                for cand in plausible:
                    if str(cand) not in squad:
                        continue  # not in roster
                    mates = line_mates.get(cand, set())
                    mate_score = len(mates & co_occurring_numbers)
                    # Also add a small bonus for having direct vote evidence
                    vote_bonus = vote_counts.get(cand, 0) / max(len(filtered_votes), 1)
                    score = mate_score + vote_bonus * 0.5
                    if score > best_score:
                        best_score = score
                        best_candidate = cand

                # Only override if we found a candidate with actual line-mate evidence
                if best_candidate is not None and best_score >= 1.0:
                    cand_name = squad.get(str(best_candidate))
                    if cand_name:
                        player_info[pid] = {
                            "name": cand_name,
                            "number": best_candidate,
                            "team": info["team"],
                        }

        # --- Goalie detection and exclusion ---
        goalie_pids = set()
        for pid, stats in merged_data.items():
            info = player_info[pid]
            toi = stats["time_on_ice"]
            dist = stats["total_distance"]
            num_str = str(info["number"]) if info["number"] is not None else None

            # Method 1: explicit goalie numbers from config
            if num_str:
                if info["team"] == "Home" and num_str in goalies_home:
                    goalie_pids.add(pid)
                    continue
                if info["team"] == "Away" and num_str in goalies_away:
                    goalie_pids.add(pid)
                    continue

            # Method 2: movement heuristic — goalies barely move
            # Average speed < 1.5 km/h with TOI > 30s is a goalie signature
            if toi > 30 and dist > 0:
                avg_speed_kmh = (dist / toi) * 3.6  # m/s -> km/h
                if avg_speed_kmh < 1.5:
                    goalie_pids.add(pid)

        # --- Build review manifest (no auto-merge) ---
        job["message"] = "Формирование манифеста для ревизии..."

        groups = []
        unidentified = []

        for canonical_pid, stats in merged_data.items():
            if canonical_pid in goalie_pids:
                continue
            info = player_info[canonical_pid]
            is_identified = info["name"] is not None
            toi = stats["time_on_ice"]

            # Filter: identified OR TOI > 3s
            if not is_identified and toi < 3:
                continue

            # Collect clips (raw tracks in this group)
            member_pids = [raw for raw, canon in pid_mapping.items() if canon == canonical_pid]
            member_pids.sort(key=lambda p: players_data[p].get("first_seen") or 0)

            clips = []
            for raw_pid in member_pids:
                vp = videos_dir / f"player_{raw_pid}.mp4"
                if not vp.exists() or vp.stat().st_size == 0:
                    continue
                pd_entry = players_data[raw_pid]
                # Per-clip auto-identity from its own votes
                clip_team = most_likely(pd_entry.get("player_team", []))
                clip_number = TrackMerger.resolve_number(pd_entry.get("player_number", []), min_count=1)
                clip_name = None
                if clip_number is not None and clip_team:
                    sq = home_squad if clip_team == "Home" else away_squad
                    clip_name = sq.get(str(clip_number))
                    # Check other roster if not found
                    if not clip_name:
                        other = away_squad if clip_team == "Home" else home_squad
                        clip_name = other.get(str(clip_number))
                        if clip_name:
                            clip_team = "Away" if clip_team == "Home" else "Home"
                # If name known, enforce correct team+number from roster
                if not clip_name:
                    clip_name_vote = most_likely(pd_entry.get("player_name", []))
                    if clip_name_vote and clip_name_vote in name_to_identity:
                        clip_team, clip_number = name_to_identity[clip_name_vote]
                        clip_name = clip_name_vote
                elif clip_name in name_to_identity:
                    clip_team, clip_number = name_to_identity[clip_name]
                # Skip very short clips (< 0.5s) — they are noise
                clip_toi = pd_entry.get("time_on_ice", 0)
                if clip_toi < 0.5:
                    continue

                clips.append({
                    "track_id": raw_pid,
                    "file": f"player_{raw_pid}.mp4",
                    "toi": round(clip_toi, 1),
                    "distance": round(pd_entry.get("total_distance", 0)),
                    "max_speed_ms": round(pd_entry.get("max_speed_ms", 0), 2),
                    "first_seen": round(pd_entry.get("first_seen", 0), 1),
                    "auto_team": clip_team,
                    "auto_number": clip_number,
                    "auto_name": clip_name,
                })

            if not clips:
                continue

            entry = {
                "canonical_pid": canonical_pid,
                "auto_team": info["team"],
                "auto_number": info["number"],
                "auto_name": info["name"],
                "total_toi": round(toi, 1),
                "total_distance": round(stats["total_distance"]),
                "clips": clips,
            }
            if is_identified:
                groups.append(entry)
            else:
                unidentified.append(entry)

        # --- Post-merge: consolidate groups with same identity ---
        # After track splitting + TrackMerger, multiple canonical_pids may
        # resolve to the same player (e.g., two "#7 Kiryanov Away" groups).
        # Merge them into a single group so the reviewer sees one player.
        consolidated = {}  # key (team, number) -> entry
        leftover = []      # groups without team+number go as-is
        for g in groups:
            key = (g["auto_team"], g["auto_number"])
            if key[0] and key[1] is not None:
                if key in consolidated:
                    # Merge clips and accumulate stats
                    consolidated[key]["clips"].extend(g["clips"])
                    consolidated[key]["total_toi"] = round(
                        consolidated[key]["total_toi"] + g["total_toi"], 1)
                    consolidated[key]["total_distance"] += g["total_distance"]
                else:
                    consolidated[key] = g
            else:
                leftover.append(g)
        # Sort clips within each consolidated group by first_seen
        for g in consolidated.values():
            g["clips"].sort(key=lambda c: c.get("first_seen", 0))
        groups = list(consolidated.values()) + leftover

        manifest = {
            "fps": fps,
            "groups": groups,
            "unidentified": unidentified,
            "roster": {
                "Home": home_squad,
                "Away": away_squad,
            },
        }
        manifest_path = result_dir / "review_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        job["status"] = "review"
        job["message"] = f"Обработано {frame_count} кадров. Готово к ревизии."
        job["result_path"] = str(result_dir)
        job["progress"] = 100

    except Exception as e:
        job["status"] = "error"
        job["message"] = f"Ошибка: {str(e)}"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    datasets = load_all_datasets()
    configs = get_team_configs()
    return render_template("dashboard.html", datasets=datasets, configs=configs)


@app.route("/api/datasets")
def api_datasets():
    datasets = load_all_datasets()
    return jsonify(datasets)


@app.route("/api/dataset/<path:name>")
def api_dataset(name):
    datasets = load_all_datasets()
    records = datasets.get(name, [])
    aggregated = aggregate_stats(records)
    return jsonify({"raw": records, "aggregated": aggregated})


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "GET":
        return render_template("upload.html")

    # POST: start processing
    video_file = request.files.get("video")
    if not video_file or not allowed_file(video_file.filename, ALLOWED_VIDEO):
        return jsonify({"error": "Загрузите видеофайл (mp4, avi, mov, mkv)"}), 400

    filename = secure_filename(video_file.filename)
    video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    video_file.save(video_path)

    # Config — sent as JSON string in config_json form field
    config_text = request.form.get("config_json", "")
    config_data = {}
    if config_text.strip():
        try:
            config_data = json.loads(config_text)
        except json.JSONDecodeError:
            return jsonify({"error": "Невалидная конфигурация (JSON)"}), 400

    job_id = hashlib.md5(f"{filename}_{time.time()}".encode()).hexdigest()[:12]
    processing_jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "message": "В очереди...",
        "video": filename,
        "result_path": None,
        "stats": None,
        "total_frames": 0,
        "processed_frames": 0,
    }

    thread = Thread(target=process_video_job, args=(job_id, video_path, config_data), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id, "redirect": url_for("job_status_page", job_id=job_id)})


@app.route("/job/<job_id>")
def job_status_page(job_id):
    job = processing_jobs.get(job_id)
    if not job:
        # Try to restore from manifest on disk
        result_dir = RESULTS_FOLDER / job_id
        manifest = result_dir / "review_manifest.json"
        if manifest.exists():
            processing_jobs[job_id] = {
                "status": "review", "progress": 100,
                "message": "Готово к ревизии.", "result_path": str(result_dir),
                "stats": None, "total_frames": 0, "processed_frames": 0, "video": "",
            }
            return redirect(url_for("review_page", job_id=job_id))
        return redirect(url_for("index"))
    if job["status"] == "review":
        return redirect(url_for("review_page", job_id=job_id))
    return render_template("job.html", job_id=job_id, job=job)


@app.route("/api/job/<job_id>")
def api_job_status(job_id):
    job = processing_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/review/<job_id>")
def review_page(job_id):
    job = processing_jobs.get(job_id)
    if not job:
        # Restore from disk
        result_dir = RESULTS_FOLDER / job_id
        manifest = result_dir / "review_manifest.json"
        if manifest.exists():
            processing_jobs[job_id] = {
                "status": "review", "progress": 100,
                "message": "Готово к ревизии.", "result_path": str(result_dir),
                "stats": None, "total_frames": 0, "processed_frames": 0, "video": "",
            }
            job = processing_jobs[job_id]
        else:
            return redirect(url_for("index"))
    return render_template("review.html", job_id=job_id)


@app.route("/api/job/<job_id>/review", methods=["GET"])
def api_review_get(job_id):
    result_dir = RESULTS_FOLDER / job_id
    manifest_path = result_dir / "review_manifest.json"
    if not manifest_path.exists():
        return jsonify({"error": "Manifest not found"}), 404
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    # Merge saved decisions if they exist
    decisions_path = result_dir / "review_decisions.json"
    if decisions_path.exists():
        with open(decisions_path, "r", encoding="utf-8") as f:
            manifest["saved_decisions"] = json.load(f)
    return jsonify(manifest)


@app.route("/api/job/<job_id>/review", methods=["POST"])
def api_review_save(job_id):
    result_dir = RESULTS_FOLDER / job_id
    if not result_dir.exists():
        return jsonify({"error": "Job not found"}), 404
    decisions = request.get_json()
    decisions_path = result_dir / "review_decisions.json"
    with open(decisions_path, "w", encoding="utf-8") as f:
        json.dump(decisions, f, ensure_ascii=False, indent=2)
    return jsonify({"status": "ok"})


@app.route("/api/job/<job_id>/finalize", methods=["POST"])
def api_finalize(job_id):
    job = processing_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    result_dir = RESULTS_FOLDER / job_id
    manifest_path = result_dir / "review_manifest.json"
    decisions_path = result_dir / "review_decisions.json"
    if not manifest_path.exists() or not decisions_path.exists():
        return jsonify({"error": "Missing manifest or decisions"}), 400

    job["status"] = "finalizing"
    job["message"] = "Финализация..."
    thread = Thread(target=finalize_job, args=(job_id,), daemon=True)
    thread.start()
    return jsonify({"status": "ok"})


def finalize_job(job_id):
    """Merge approved/reassigned clips, generate Excel, calculate accuracy."""
    job = processing_jobs[job_id]
    try:
        result_dir = RESULTS_FOLDER / job_id
        videos_dir = result_dir / "player_videos"

        with open(result_dir / "review_manifest.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)
        with open(result_dir / "review_decisions.json", "r", encoding="utf-8") as f:
            decisions = json.load(f)

        fps = manifest.get("fps", 25)
        roster = manifest.get("roster", {})
        group_decisions = decisions.get("decisions", {})
        unid_decisions = decisions.get("unidentified_decisions", {})

        # Build final player groups: player_key -> list of clips
        # player_key = "Team_Number_Name"
        final_groups = {}  # key -> {"team", "number", "name", "clips": [...]}

        def player_key(team, number, name):
            return f"{team}_{number}_{name}"

        def add_clip(key, team, number, name, clip):
            if key not in final_groups:
                final_groups[key] = {"team": team, "number": number, "name": name, "clips": []}
            final_groups[key]["clips"].append(clip)

        # Accuracy counters
        correct = 0
        corrected = 0
        rejected = 0
        newly_assigned = 0
        total_reviewed = 0

        # Process identified groups
        all_groups = manifest.get("groups", []) + manifest.get("unidentified", [])
        for group in manifest.get("groups", []):
            cpid = str(group["canonical_pid"])
            grp_team = group["auto_team"]
            grp_number = group["auto_number"]
            grp_name = group["auto_name"]
            grp_key = player_key(grp_team, grp_number, grp_name)
            grp_dec = group_decisions.get(cpid, {}).get("clips", {})

            for clip in group["clips"]:
                tid = str(clip["track_id"])
                action = grp_dec.get(tid, {}).get("action", "approve")
                total_reviewed += 1

                if action == "approve":
                    add_clip(grp_key, grp_team, grp_number, grp_name, clip)
                    correct += 1
                elif action == "reject":
                    rejected += 1
                elif action == "reassign":
                    dec = grp_dec[tid]
                    t_team = dec.get("target_team", grp_team)
                    t_number = dec.get("target_number")
                    t_name = dec.get("target_name", "")
                    t_key = player_key(t_team, t_number, t_name)
                    add_clip(t_key, t_team, t_number, t_name, clip)
                    corrected += 1

        # Process unidentified
        for group in manifest.get("unidentified", []):
            cpid = str(group["canonical_pid"])
            # Unidentified decisions are stored per track_id
            for clip in group["clips"]:
                tid = str(clip["track_id"])
                dec = unid_decisions.get(tid, {})
                action = dec.get("action", "ignore")

                if action == "assign":
                    t_team = dec.get("target_team", "Unknown")
                    t_number = dec.get("target_number")
                    t_name = dec.get("target_name", "")
                    t_key = player_key(t_team, t_number, t_name)
                    add_clip(t_key, t_team, t_number, t_name, clip)
                    newly_assigned += 1
                    total_reviewed += 1
                # ignore = skip

        # Merge videos per final group
        job["message"] = "Склейка видео..."
        final_videos_dir = result_dir / "player_videos_final"
        final_videos_dir.mkdir(exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        stats_list = []

        for key, group in final_groups.items():
            clips = sorted(group["clips"], key=lambda c: c["track_id"])
            if not clips:
                continue

            team = group["team"] or "Unknown"
            number = group["number"]
            name = group["name"]

            if name and number is not None:
                fname = f"{team}_{number}_{name}.mp4"
            elif number is not None:
                fname = f"{team}_{number}.mp4"
            else:
                fname = f"{team}_unknown.mp4"
            fname = fname.replace(" ", "_").replace("/", "_")
            out_path = str(final_videos_dir / fname)

            out_writer = None
            total_toi = 0
            total_dist = 0
            max_speed_ms = 0  # max instantaneous speed in m/s across all clips
            for clip in clips:
                src = str(videos_dir / clip["file"])
                if not os.path.exists(src):
                    continue
                src_cap = cv2.VideoCapture(src)
                while True:
                    ret, fr = src_cap.read()
                    if not ret:
                        break
                    if out_writer is None:
                        h, w = fr.shape[:2]
                        out_writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
                    out_writer.write(fr)
                src_cap.release()
                total_toi += clip.get("toi", 0)
                total_dist += clip.get("distance", 0)
                clip_max = clip.get("max_speed_ms", 0)
                if clip_max > max_speed_ms:
                    max_speed_ms = clip_max

            if out_writer is not None:
                out_writer.release()

            # Speeds in km/h (convert from m/s: * 3.6)
            avg_speed_kmh = round((total_dist / total_toi * 3.6) if total_toi > 0 else 0, 1)
            max_speed_kmh = round(max_speed_ms * 3.6, 1)

            stats_list.append({
                "Name": name,
                "Number": number,
                "Team": team,
                "Time (s)": round(total_toi),
                "Distance (m)": round(total_dist),
                "Avg Speed (km/h)": avg_speed_kmh,
                "Max Speed (km/h)": max_speed_kmh,
            })

        # Generate Excel
        stats_path = str(result_dir / "player_statistics.xlsx")
        pd.DataFrame(stats_list).to_excel(stats_path, index=False)

        # Accuracy report
        identified_total = correct + corrected + rejected
        accuracy_pct = round(correct / identified_total * 100, 1) if identified_total > 0 else 0
        accuracy = {
            "total_reviewed": total_reviewed,
            "correct": correct,
            "corrected": corrected,
            "rejected": rejected,
            "newly_assigned": newly_assigned,
            "accuracy_pct": accuracy_pct,
        }
        with open(result_dir / "accuracy_report.json", "w", encoding="utf-8") as f:
            json.dump(accuracy, f, ensure_ascii=False, indent=2)

        job["status"] = "completed"
        job["message"] = f"Готово! Точность авторазметки: {accuracy_pct}%"
        job["stats"] = stats_list
        job["accuracy"] = accuracy

    except Exception as e:
        job["status"] = "error"
        job["message"] = f"Ошибка финализации: {str(e)}"


@app.route("/stats")
def stats_page():
    datasets = load_all_datasets()
    all_records = []
    for records in datasets.values():
        all_records.extend(records)
    aggregated = aggregate_stats(all_records)

    # Per-dataset aggregation
    per_dataset = {}
    for name, records in datasets.items():
        per_dataset[name] = aggregate_stats(records)

    return render_template("stats.html", aggregated=aggregated, per_dataset=per_dataset, datasets=datasets)


CONFIGS_FOLDER = BASE_DIR / "configs"
CONFIGS_FOLDER.mkdir(exist_ok=True)


@app.route("/config")
def config_page():
    return render_template("config.html")


@app.route("/api/configs", methods=["GET"])
def api_configs_list():
    """Return list of all saved configurations."""
    configs = []
    if CONFIGS_FOLDER.exists():
        for f in sorted(CONFIGS_FOLDER.iterdir()):
            if f.suffix == ".json":
                try:
                    with open(f, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    configs.append({
                        "id": f.stem,
                        "name": data.get("config_name", f.stem),
                    })
                except Exception:
                    pass
    # Also include legacy configs from project dir
    for cfg in get_team_configs():
        configs.append({
            "id": "legacy__" + cfg["path"].replace("/", "__").replace("\\", "__"),
            "name": "[legacy] " + cfg["path"],
        })
    return jsonify(configs)


@app.route("/api/configs/<config_id>", methods=["GET"])
def api_config_get(config_id):
    """Return a single configuration by id."""
    if config_id.startswith("legacy__"):
        # Legacy config from project dir
        rel_path = config_id[len("legacy__"):].replace("__", "/")
        cfg_path = PROJECT_DIR / rel_path
        if not cfg_path.exists():
            return jsonify({"error": "Config not found"}), 404
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("config_name", rel_path)
        return jsonify(data)
    else:
        cfg_path = CONFIGS_FOLDER / f"{config_id}.json"
        if not cfg_path.exists():
            return jsonify({"error": "Config not found"}), 404
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)


@app.route("/api/configs", methods=["POST"])
def api_config_save():
    """Save a configuration."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data"}), 400
    name = data.get("config_name", "unnamed")
    # Create safe filename from name
    safe_name = "".join(c if c.isalnum() or c in " -_" else "_" for c in name).strip()
    if not safe_name:
        safe_name = "config"
    cfg_path = CONFIGS_FOLDER / f"{safe_name}.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return jsonify({"status": "ok", "id": safe_name})


@app.route("/api/configs/<config_id>", methods=["PUT"])
def api_config_update(config_id):
    """Update an existing configuration."""
    if config_id.startswith("legacy__"):
        return jsonify({"error": "Cannot edit legacy configs"}), 400
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data"}), 400
    cfg_path = CONFIGS_FOLDER / f"{config_id}.json"
    if not cfg_path.exists():
        return jsonify({"error": "Config not found"}), 404
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return jsonify({"status": "ok", "id": config_id})


@app.route("/api/configs/<config_id>", methods=["DELETE"])
def api_config_delete(config_id):
    """Delete a configuration."""
    if config_id.startswith("legacy__"):
        return jsonify({"error": "Cannot delete legacy configs"}), 400
    cfg_path = CONFIGS_FOLDER / f"{config_id}.json"
    if cfg_path.exists():
        cfg_path.unlink()
    return jsonify({"status": "ok"})


@app.route("/results/<path:filename>")
def serve_result(filename):
    return send_from_directory(str(RESULTS_FOLDER), filename)


@app.route("/api/models")
def api_models():
    """Check which YOLO models are available."""
    models = {}
    for name in ["Player_Detection.pt", "LinesDetection.pt", "NumbersRec.pt"]:
        p = MODELS_DIR / name
        models[name] = {"exists": p.exists(), "size_mb": round(p.stat().st_size / 1024 / 1024, 1) if p.exists() else 0}
    tracker = TRACKERS_DIR / "tracker.yaml"
    models["tracker.yaml"] = {"exists": tracker.exists()}
    return jsonify(models)


@app.route("/results-list")
def results_list_page():
    """Show all locally saved job results."""
    return render_template("results_list.html")


@app.route("/api/results-list")
def api_results_list():
    """Return list of all completed/review jobs on disk."""
    jobs = []
    if RESULTS_FOLDER.exists():
        for d in sorted(RESULTS_FOLDER.iterdir(), reverse=True):
            if not d.is_dir():
                continue
            job_id = d.name
            manifest = d / "review_manifest.json"
            accuracy = d / "accuracy_report.json"
            excel = d / "player_statistics.xlsx"
            status = "unknown"
            info = {"job_id": job_id}

            if accuracy.exists():
                status = "completed"
                with open(accuracy, "r", encoding="utf-8") as f:
                    info["accuracy"] = json.load(f)
            elif manifest.exists():
                status = "review"

            if excel.exists():
                info["has_excel"] = True
                info["excel_path"] = f"{job_id}/player_statistics.xlsx"

            final_dir = d / "player_videos_final"
            raw_dir = d / "player_videos"
            if final_dir.exists():
                info["has_final_videos"] = True
                info["video_count"] = len(list(final_dir.glob("*.mp4")))
            elif raw_dir.exists():
                info["video_count"] = len(list(raw_dir.glob("*.mp4")))

            if manifest.exists():
                try:
                    with open(manifest, "r", encoding="utf-8") as f:
                        m = json.load(f)
                    info["groups_count"] = len(m.get("groups", []))
                    info["unidentified_count"] = len(m.get("unidentified", []))
                except Exception:
                    pass

            info["status"] = status
            info["created"] = d.stat().st_mtime
            jobs.append(info)

    return jsonify(jobs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    app.run(debug=args.debug, host="0.0.0.0", port=args.port)
