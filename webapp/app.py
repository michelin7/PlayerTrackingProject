import os
import json
import math
import time
import hashlib
import random
import signal
import sys
import shutil
from threading import Lock, Thread, Event
from collections import defaultdict, namedtuple, Counter, deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for, send_file
from werkzeug.utils import secure_filename

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Packaged (PyInstaller) the code sits in a read-only bundle, so everything we
# write goes to the per-user data directory. Run from source, nothing moves.
FROZEN = getattr(sys, "frozen", False)
BASE_DIR = Path(__file__).resolve().parent


def user_data_dir():
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "PlayerTracking"
    if sys.platform == "win32":
        local = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "PlayerTracking"
        # OpenCV on Windows cannot open a file whose path is not ASCII, and it
        # fails quietly: a user called "Михаил" would get an empty results
        # folder and no error. C:/Users/Public is ASCII whoever is logged in.
        if str(local).isascii():
            return local
        return Path(os.environ.get("PUBLIC") or "C:/Users/Public") / "PlayerTracking"
    return Path(os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share")) / "PlayerTracking"


DATA_DIR = user_data_dir() if FROZEN else BASE_DIR
PROJECT_DIR = DATA_DIR if FROZEN else BASE_DIR.parent
MODELS_DIR = (BASE_DIR if FROZEN else PROJECT_DIR) / "models"
TRACKERS_DIR = (BASE_DIR if FROZEN else PROJECT_DIR) / "trackers"
UPLOAD_FOLDER = DATA_DIR / "uploads"
RESULTS_FOLDER = DATA_DIR / "results"
# Accumulated training-data staging (NumbersRec fine-tuning), Roboflow-ready.
TRAINING_FOLDER = DATA_DIR / "training_dataset"
CONFIGS_FOLDER = DATA_DIR / "configs"

for _d in (UPLOAD_FOLDER, RESULTS_FOLDER, CONFIGS_FOLDER,
           TRAINING_FOLDER / "images", TRAINING_FOLDER / "labels"):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__,
            template_folder=str(BASE_DIR / "templates"),
            static_folder=str(BASE_DIR / "static"))
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

    # How often (in frames) to re-run full identification per player. 1 = read the
    # number EVERY processed frame: at 30fps that is the most number votes per
    # second (best identification accuracy) and catches swaps fastest.
    ID_INTERVAL = 1
    # Consecutive differing number reads before we treat it as a swap and split.
    # Calibrated for reading the number EVERY frame (ID_INTERVAL=1): a real swap
    # shows the new number for many frames, while OCR jitter is a brief burst, so
    # we require sustained disagreement (~0.3s @30fps) instead of just 3 frames —
    # otherwise number-guard false-fires and shreds a track into many clips.
    NUM_SWITCH_CONFIRM = 10
    NUM_SWITCH_CONFIRM_CROSS = 8   # steeper views overlap boxes often; don't split on 4 noisy reads
    CROSS_IOU = 0.50          # box overlap that counts as a crossing (raised: incidental overlaps don't count)
    CROSS_WINDOW_SEC = 1.0    # post-crossing scrutiny window

    # --- Cross-TEAM swap detection ---
    # A player cannot change jersey, so sustained opposite-team evidence is proof
    # the tracker moved onto a DIFFERENT player (body check, post-crossing handoff)
    # — cases the kinematic guard misses because the new player is right next to
    # the old one. Judged on a RECENT window (the lifetime tally is deliberately
    # sticky and would never flip), and deliberately NOT gated to crossings: the
    # jersey changed, so a swap happened whether or not we flagged an overlap.
    # Wide window + high margin so classifier noise (ice/glare) cannot shred a track.
    # OFF by default: the splits it produced cost more review time than the
    # cross-team swaps they fixed. Flip to True to re-enable (detection below
    # keeps running either way, so this is the only switch).
    TEAM_SPLIT = False
    TEAM_WINDOW = 12            # recent raw team reads kept per track
    TEAM_SWITCH_CONFIRM = 8     # ...need at least this many before judging
    TEAM_SWITCH_MARGIN = 0.75   # ...and this share of their weight must back the other team

    # --- Number recognition thresholds (tunable) ---
    NUM_STRONG_CONF = 0.50   # a "confident" digit detection
    NUM_MIN_CONF = 0.30      # min conf to be a single-digit number on its own
    # A faint SECOND digit (e.g. the trailing "4" of #44) often sits below
    # NUM_MIN_CONF. We let it into two-digit PAIRING down to NUM_WEAK_CONF, but
    # only when the pair forms a roster-valid number — so faint noise can never
    # invent a number, yet a real dropped digit is recovered.
    NUM_WEAK_CONF = 0.18
    # Training-data mining: a crop is saved ONLY if it has a digit at least this
    # confident, and only such digits go into the pre-annotation. Higher = less
    # garbage in the dataset.
    NUM_COLLECT_CONF = 0.50
    # How decisively the winning digit-count hypothesis (1-digit vs 2-digit)
    # must beat the other before we commit; otherwise -> Unknown.
    NUM_COUNT_MARGIN = 1.4

    # --- Team classifier (light vs dark cluster tracking) ---
    TEAM_EMA = 0.03          # how fast each brightness cluster center adapts
    TEAM_MIN_GAP = 15.0      # never let the two centers collapse closer than this
    # Ice exclusion (proposal B): ice is low-saturation + bright, and it touches
    # the crop BORDER, whereas the jersey is a central island. We flood ice in
    # from the edges and measure brightness only on what's left.
    ICE_S_MAX = 55           # ice/white: saturation below this ...
    ICE_V_MIN = 150          # ... and value above this
    ICE_JERSEY_MIN_FRAC = 0.15  # if <15% survives, the whole torso is bright -> white jersey
    # Vote weighting (proposal D): a per-frame team vote counts more from a big
    # (close) box and less from a tiny (far) one, on top of the margin.
    TEAM_SIZE_REF = 150.0    # box height (px) at which the size weight saturates
    TEAM_SIZE_MIN = 0.15     # floor so far boxes still contribute a little
    TEAM_SIZE_MAX = 1.5      # cap so one huge close-up can't dominate outright
    # Number-anchored team vote (proposal A): the digit bbox is guaranteed ON the
    # jersey, so its polarity (digit darker/lighter than the jersey around it)
    # gives an ice-proof team read. Cast as a strong extra vote when a number is
    # read and the digit/jersey contrast is clear.
    NUM_TEAM_WEIGHT = 3.0    # max strength of a number-anchored vote
    NUM_TEAM_MIN_CONTRAST = 30  # min |V_jersey - V_digit| to trust the polarity

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
        # Team-first: confidence-weighted light/dark vote tally per track. The
        # track's team is argmax of these accumulated votes, so it is decided by
        # the whole track's history and a few flickering crops cannot change it.
        self._team_votes = {}  # pid -> [light_weight, dark_weight]
        # Cross-team swap detection: the team this segment is established as, plus
        # a RECENT window of raw team reads (the tally above is too sticky to flip).
        self._est_team = {}      # pid -> established team for the current segment
        self._recent_team = {}   # pid -> deque[(team, weight)]
        # Number-swap detection: established number + consecutive-mismatch count
        self._est_number = {}
        self._num_switch_counts = {}
        # Last number the digits SUGGESTED per track, even if rejected, with the
        # read's score — the per-frame confidence weighted voting runs on.
        self._last_guess = {}  # pid -> (number:int|None, score:float)
        # Crossing awareness: frame until which a track is under post-crossing
        # scrutiny (a swap is most likely right after two boxes overlap).
        self._crossed_until = {}
        # Per-track brightness history — smooth the team signal so an unstable
        # crop (catching ice/boards for a frame) can't flip the team.
        self._bright_hist = {}
        # Diagnostics
        self._last_bright = None
        self._last_mid = None
        # Number-decision debug log: one record per number read with the RAW
        # YOLO digit detections + the reason, so a wrong number can be traced to
        # the model or to the parse logic. On by default; disable with
        # PSF_NUM_DEBUG=0. Flushed to <result_dir>/number_debug.json.
        self._num_debug = []
        self._num_debug_on = os.environ.get("PSF_NUM_DEBUG", "1") != "0"
        self._num_debug_cap = 40000
        # Training-data mining: when _collect_dir is set, every player crop in
        # which YOLO found a digit is saved as an image + YOLO-format label
        # (the model's own boxes as a pre-annotation to refine in Roboflow).
        self._collect_dir = None
        self._collect_cap = 800
        self._collected = 0
        self._collect_per_track = {}   # pid -> saved count (diversity cap)
        self._split_log = []

        # Reference HSV list for each team (1-2 colors)
        home_colors = teams_colors_info.get("Home", ["blue"])
        away_colors = teams_colors_info.get("Away", ["white"])
        self._home_refs = [self._get_reference_hsv(c) for c in home_colors if c]
        self._away_refs = [self._get_reference_hsv(c) for c in away_colors if c]
        if not self._home_refs:
            self._home_refs = [self._get_reference_hsv("blue")]
        if not self._away_refs:
            self._away_refs = [self._get_reference_hsv("white")]

        # Team classification is by jersey brightness (light vs dark), not by
        # matching specific hues. We only need to know which declared team
        # wears the lighter jersey; the actual split point adapts online.
        self._home_V = float(np.mean([r[2] for r in self._home_refs]))
        self._away_V = float(np.mean([r[2] for r in self._away_refs]))
        self._light_team = "Home" if self._home_V >= self._away_V else "Away"
        self._dark_team = "Away" if self._light_team == "Home" else "Home"
        # Two online cluster centers (light-jersey / dark-jersey brightness).
        # The split point is their midpoint, which — unlike a single running
        # mean of all crops — does NOT drift when one team is on screen more
        # than the other. Seeded from the config reference brightnesses and
        # then tracked per side with a slow EMA to follow exposure changes.
        self._light_center = max(self._home_V, self._away_V)
        self._dark_center = min(self._home_V, self._away_V)

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

    # ------------------------------------------------------------------
    # Team classification (light vs dark jersey)
    # ------------------------------------------------------------------
    def _update_team_centers(self, brightness):
        """Assign a brightness reading to the nearer cluster, nudge that center
        with a slow EMA, and return the current split midpoint. Tracking the two
        clusters separately keeps the boundary stable under on-screen imbalance;
        a floor stops the centers from collapsing together.
        """
        if abs(brightness - self._light_center) <= abs(brightness - self._dark_center):
            new_light = self._light_center + self.TEAM_EMA * (brightness - self._light_center)
            new_dark = self._dark_center
        else:
            new_light = self._light_center
            new_dark = self._dark_center + self.TEAM_EMA * (brightness - self._dark_center)
        if new_light - new_dark >= self.TEAM_MIN_GAP:
            self._light_center, self._dark_center = new_light, new_dark
        return (self._light_center + self._dark_center) / 2

    def _jersey_brightness(self, crop_bgr):
        """Median jersey brightness (HSV V) with ICE excluded (proposal B).

        Ice is low-saturation + bright AND touches the crop border; the jersey
        is a central island. We label ice-like pixels that are connected to the
        border and measure V only on the rest. If almost nothing survives, the
        whole torso is bright -> it is genuinely a WHITE jersey (light team), so
        we measure those bright pixels (they ARE the jersey), not throw them out.
        A plain median over the crop would flip a dark jersey to 'light' whenever
        ice fills more than half the box.
        """
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        V = hsv[:, :, 2]
        ice_like = ((hsv[:, :, 1] < self.ICE_S_MAX) & (V > self.ICE_V_MIN)).astype(np.uint8)
        border_ice = np.zeros(ice_like.shape, dtype=bool)
        if ice_like.any():
            n, labels = cv2.connectedComponents(ice_like)
            border_labels = (set(labels[0, :]) | set(labels[-1, :]) |
                             set(labels[:, 0]) | set(labels[:, -1]))
            border_labels.discard(0)                 # 0 = non-ice background
            if border_labels:
                border_ice = np.isin(labels, list(border_labels))
        jersey = ~border_ice
        if jersey.mean() < self.ICE_JERSEY_MIN_FRAC:  # all bright & border-linked -> white jersey
            sample = V[border_ice] if border_ice.any() else V.reshape(-1)
        else:
            sample = V[jersey]
        return float(np.median(sample))

    def _number_team(self, crop_rgb, boxes):
        """Team from the number's polarity (proposal A): compare the digit
        pixels' brightness to the jersey ring just around them. Both are on the
        player, never on ice, so this is ice-proof. Returns (team, strength) or
        (None, 0.0) when no clear digit/jersey contrast is available.
        """
        if boxes is None or len(boxes) == 0:
            return None, 0.0
        H, W = crop_rgb.shape[:2]
        dboxes = [(bx, cf) for bx, cf in zip(boxes.xyxy.tolist(), boxes.conf.tolist())
                  if cf >= self.NUM_MIN_CONF]
        if not dboxes:
            return None, 0.0
        digit_mask = np.zeros((H, W), dtype=bool)
        ux1, uy1, ux2, uy2 = W, H, 0, 0
        for (bx1, by1, bx2, by2), _cf in dboxes:
            ix1, iy1 = max(int(bx1), 0), max(int(by1), 0)
            ix2, iy2 = min(int(bx2), W), min(int(by2), H)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            digit_mask[iy1:iy2, ix1:ix2] = True
            ux1, uy1, ux2, uy2 = min(ux1, ix1), min(uy1, iy1), max(ux2, ix2), max(uy2, iy2)
        if not digit_mask.any():
            return None, 0.0
        # jersey ring = a padded box around the digits, minus the digits themselves
        padx, pady = int((ux2 - ux1) * 0.6) + 2, int((uy2 - uy1) * 0.35) + 2
        rx1, ry1 = max(ux1 - padx, 0), max(uy1 - pady, 0)
        rx2, ry2 = min(ux2 + padx, W), min(uy2 + pady, H)
        ring = np.zeros((H, W), dtype=bool)
        ring[ry1:ry2, rx1:rx2] = True
        ring &= ~digit_mask
        if ring.sum() < 8:
            return None, 0.0
        V = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)[:, :, 2]
        v_digit = float(np.median(V[digit_mask]))
        v_jersey = float(np.median(V[ring]))
        contrast = abs(v_jersey - v_digit)
        if contrast < self.NUM_TEAM_MIN_CONTRAST:     # digit ~ jersey -> unreliable
            return None, 0.0
        team = self._light_team if v_jersey >= v_digit else self._dark_team
        strength = min(contrast / 80.0, 1.0) * self.NUM_TEAM_WEIGHT
        return team, strength

    # ------------------------------------------------------------------
    # Cross-team swap detection
    # ------------------------------------------------------------------
    def _push_team_evidence(self, pid, team, weight):
        """Record one RAW team read (brightness or number-polarity) in the track's
        recent window. Kept separate from the lifetime tally, which is sticky by
        design and therefore cannot reveal that the track changed player."""
        if not team or team == "Unknown" or weight <= 0:
            return
        dq = self._recent_team.get(pid)
        if dq is None:
            dq = self._recent_team[pid] = deque(maxlen=self.TEAM_WINDOW)
        dq.append((team, float(weight)))

    def _team_swap_target(self, pid):
        """Return the other team when the recent window sustainedly contradicts
        this segment's established team (i.e. the tracker is now on a player of
        the OTHER team), else None."""
        est = self._est_team.get(pid)
        dq = self._recent_team.get(pid)
        if est is None or not dq or len(dq) < self.TEAM_SWITCH_CONFIRM:
            return None
        by_team = {}
        for t, w in dq:
            by_team[t] = by_team.get(t, 0.0) + w
        total = sum(by_team.values())
        if total <= 0:
            return None
        others = [t for t in by_team if t != est]
        if not others:
            return None
        other = max(others, key=lambda t: by_team[t])
        return other if by_team[other] / total >= self.TEAM_SWITCH_MARGIN else None

    def _apply_team_swap(self, pid, new_team):
        """Start a fresh identity segment for this track after a confirmed swap:
        drop the previous player's accumulated team/number evidence (otherwise the
        sticky tally keeps reporting the OLD team) and seed the new team."""
        self._est_team[pid] = new_team
        self._team_votes[pid] = ([1.0, 0.0] if new_team == self._light_team
                                 else [0.0, 1.0])
        self._recent_team.pop(pid, None)
        self._est_number.pop(pid, None)
        self._num_switch_counts.pop(pid, None)
        self._cache.pop(pid, None)

    def _classify_team_hsv(self, frame_bgr, x1, y1, x2, y2, pid=None):
        """Classify team by jersey brightness (light vs dark).

        Hockey rules require the two teams to wear contrasting jerseys — one
        light (usually white), one dark. Brightness is therefore a far more
        robust and cheaper signal than matching specific hues, which drift
        with lighting and broadcast color grading.

        We take the MEDIAN brightness (HSV V) of a tight torso crop and split
        it at the midpoint between two online cluster centers (light-jersey and
        dark-jersey brightness), each tracked with a slow EMA. Median (not mean)
        resists the few ice/skin pixels leaking into the crop; tracking the two
        clusters separately keeps the split point steady even when one team is
        on screen far more than the other (a single running mean would drift).

        # ponytail: assumes one light + one dark team (true in regulated
        # hockey) and roughly balanced on-ice presence over the video; if both
        # jerseys are similarly bright the signal is weak and the human review
        # step corrects the misses.

        Returns ('Home'|'Away'|'Unknown', confidence 0..1).
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

        brightness = self._jersey_brightness(crop)
        midpoint = self._update_team_centers(brightness)

        # Classify on a per-track SMOOTHED brightness (median of recent frames)
        # so a single bad crop can't flip the team.
        if pid is not None:
            h = self._bright_hist.setdefault(pid, [])
            h.append(brightness)
            if len(h) > 5:
                h.pop(0)
            bright_used = float(np.median(h))
        else:
            bright_used = brightness

        # Margin scaled to half the CURRENT cluster separation (with a floor),
        # so confidence reflects the real light/dark gap in this broadcast.
        half_gap = (self._light_center - self._dark_center) / 2
        margin = abs(bright_used - midpoint) / max(half_gap, 20.0)
        self._last_bright = bright_used
        self._last_mid = midpoint
        if margin < 0.25:               # borderline -> Unknown
            return "Unknown", 0.0

        team = self._light_team if bright_used >= midpoint else self._dark_team
        return team, min(margin, 1.0)

    # ------------------------------------------------------------------
    # Number recognition helpers
    # ------------------------------------------------------------------
    def _parse_number_results(self, results, team, crop_width=None, pid=None, frame=None):
        """Parse YOLO number detection with low-conf sweep + roster validation.

        Strategy:
        1. Detect ALL digits at low conf (0.25) to avoid missing faint 2nd digits
        2. De-duplicate overlapping detections of the same digit
        3. Build candidate numbers (single-digit and two-digit)
        4. Validate every candidate against the team roster
        5. Score candidates; if ambiguous, return Unknown (let temporal voting decide)

        This reliably separates single-digit (e.g. 4) from two-digit numbers
        that contain that digit (e.g. 14, 47).

        When number debugging is on, every decision with at least one detected
        digit is logged with the RAW YOLO detections and the reason — so a wrong
        number can be blamed on the model (bad/missing digit boxes) or on this
        logic (right digits, wrong pick).
        """
        def _log(num, reason, dets):
            if not self._num_debug_on or dets is None:
                return
            if len(self._num_debug) < self._num_debug_cap:
                self._num_debug.append({
                    "pid": pid, "frame": frame, "team": team,
                    "dets": dets, "num": num, "reason": reason,
                })

        if not results[0].boxes or len(results[0].boxes) == 0:
            _log(None, "no_digit_boxes", [])
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

        # What YOLO actually saw (digit, confidence, horizontal position), for
        # the debug log: this is the "model" side of the logic-vs-model question.
        dbg = [[d["digit"], round(d["conf"], 2),
                round(d["cx"] / crop_width, 2) if crop_width else -1.0]
               for d in digits]

        squad = self.Home if team == "Home" else self.Away
        STRONG_CONF = self.NUM_STRONG_CONF
        MIN_CONF = self.NUM_MIN_CONF
        WEAK_CONF = self.NUM_WEAK_CONF
        roster_two = {n for n in (int(k) for k in squad if k.isdigit()) if 10 <= n <= 99}

        # Two digit pools: strong ones stand alone as single-digit numbers;
        # faint ones may only complete a two-digit number (see NUM_WEAK_CONF).
        strong_digits = [d for d in digits if d["conf"] >= MIN_CONF]
        weak_digits = [d for d in digits if WEAK_CONF <= d["conf"] < MIN_CONF]
        if not strong_digits:
            _log(None, "all_digits_below_min_conf", dbg)
            return "Unknown", None

        # Candidates are grouped by digit-count hypothesis so we can weigh
        # "is this one digit or two?" explicitly before naming the number.
        single_cands = []  # (number, score, tag)
        two_cands = []

        # -- Single-digit hypotheses (strong digits only) --
        for d in strong_digits:
            num = d["digit"]
            score = d["conf"]
            if crop_width and crop_width > 0:
                rel_cx = d["cx"] / crop_width
                rel_w = d["w"] / crop_width
                if 0.30 <= rel_cx <= 0.70:
                    # A true single-digit jersey is centered, ~25-50% of width
                    if rel_w >= 0.18:
                        single_cands.append((num, score * 1.0, "single_centered", score))
                    else:
                        single_cands.append((num, score * 0.7, "single_narrow", score))
                else:
                    # Off-center: almost certainly one half of a 2-digit number
                    single_cands.append((num, score * 0.15, "single_offcenter", score))
            else:
                single_cands.append((num, score * 0.6, "single_no_crop", score))

        # -- Two-digit hypotheses (pairs). A pair may use one faint digit, but
        #    then the number MUST be on the roster (guards against faint noise). --
        pool = strong_digits + weak_digits
        pool.sort(key=lambda d: d["cx"])
        for i in range(len(pool)):
            for j in range(i + 1, len(pool)):
                d0, d1 = pool[i], pool[j]
                gap = d1["x1"] - d0["x2"]
                avg_w = (d0["w"] + d1["w"]) / 2
                if gap < -avg_w * 0.6:
                    continue  # too much overlap = same digit twice
                if crop_width and gap > crop_width * 0.4:
                    continue  # too far apart = unrelated

                two_digit = d0["digit"] * 10 + d1["digit"]
                min_conf = min(d0["conf"], d1["conf"])
                avg_conf = (d0["conf"] + d1["conf"]) / 2
                uses_faint = min_conf < MIN_CONF
                # A faint second digit may only complete a DOUBLED roster number
                # (44, 22, 99 …). Faint completion of a two-different-digit number
                # fabricated magnets (#28, #34) for players whose number was not
                # actually readable, so it is not trusted here.
                if uses_faint and (two_digit not in roster_two
                                   or d0["digit"] != d1["digit"]):
                    continue

                if min_conf >= STRONG_CONF:
                    pair_score = avg_conf * 1.2
                elif min_conf >= MIN_CONF:
                    pair_score = avg_conf * 0.8
                else:
                    pair_score = avg_conf * 0.6  # faint but roster-backed

                if crop_width and crop_width > 0:
                    span = d1["x2"] - d0["x1"]
                    if span / crop_width >= 0.35:
                        pair_score *= 1.1
                    rel_center = ((d0["cx"] + d1["cx"]) / 2) / crop_width
                    if 0.30 <= rel_center <= 0.70:
                        pair_score *= 1.1

                two_cands.append((two_digit, pair_score, "pair", avg_conf))

        # (Roster inference from a single lone digit was removed: it fabricated a
        #  confident number whenever the roster made one unique, which created the
        #  #28/#34 magnets for players whose number was never actually visible.)

        # --- Roster validation: boost roster hits, crush non-roster ---
        def _roster_weight(cands):
            out = []
            for num, score, tag, conf in cands:
                mult = 2.0 if str(num) in squad else 0.05
                out.append((num, score * mult, tag, conf))
            return sorted(out, key=lambda x: -x[1])

        single_cands = _roster_weight(single_cands)
        two_cands = _roster_weight(two_cands)

        best_single = single_cands[0] if single_cands else (None, 0.0, None, 0.0)
        best_two = two_cands[0] if two_cands else (None, 0.0, None, 0.0)

        # --- Decide digit-count first, then the number ---
        if best_single[1] >= best_two[1]:
            winner, loser_score = best_single, best_two[1]
        else:
            winner, loser_score = best_two, best_single[1]
        best_num, best_score, best_tag, best_conf = winner

        # Record the raw suggestion (even if it will be rejected below) so live
        # mode can surface weak reads for an otherwise-unidentified track. Store
        # the raw digit CONFIDENCE (0..1), not best_score (a roster-weighted rank
        # that can exceed 1 -> "242%").
        if pid is not None and best_num is not None:
            self._last_guess[pid] = (best_num, round(min(float(best_conf), 1.0), 2))

        if best_num is None or best_score < 0.25:
            _log(None, f"low_score({round(best_score, 2)}<0.25)", dbg)
            return "Unknown", None
        # The winning hypothesis must clearly beat the other digit-count option
        # (unless the loser is empty) — this is the 1-vs-2 confidence gate.
        if loser_score > 0 and best_score < loser_score * self.NUM_COUNT_MARGIN:
            _log(None, f"1v2_ambiguous({round(best_score, 2)}vs{round(loser_score, 2)})", dbg)
            return "Unknown", None
        # Ambiguity within the winning hypothesis (two plausible numbers).
        same_class = single_cands if winner is best_single else two_cands
        if len(same_class) > 1 and same_class[1][0] != best_num:
            if same_class[1][1] > best_score * 0.65:
                _log(None, f"tie({best_num}vs{same_class[1][0]})", dbg)
                return "Unknown", None

        _log(best_num, f"chosen:{best_tag}", dbg)
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
        number_switches = set()
        needs_number = []  # (track_id, crop_rgb, team)

        for pid, x1, y1, x2, y2 in detections:
            # --- Team-first: classify this frame, add a confidence-weighted vote,
            # then take the track's team as the argmax of ALL its votes so far.
            # The team is a stable property of the whole track — a few flickering
            # crops can no longer flip it, and we no longer SPLIT on a team flip
            # (the kinematic and number-swap guards catch real cross-player jumps).
            raw_team, confidence = self._classify_team_hsv(frame_bgr, x1, y1, x2, y2, pid=pid)
            if raw_team != "Unknown" and confidence > 0:
                # Weight the vote by box size (proposal D): a close/large box is a
                # reliable read; a far/tiny one barely counts.
                size_w = min(max((y2 - y1) / self.TEAM_SIZE_REF, self.TEAM_SIZE_MIN),
                             self.TEAM_SIZE_MAX)
                w = confidence * size_w
                self._push_team_evidence(pid, raw_team, w)
                v = self._team_votes.setdefault(pid, [0.0, 0.0])
                if raw_team == self._light_team:
                    v[0] += w
                else:
                    v[1] += w
            v = self._team_votes.get(pid)
            if v and (v[0] or v[1]):
                team = self._light_team if v[0] >= v[1] else self._dark_team
            else:
                team = self._cache.get(pid, {}).get("team", "Unknown")

            cached = self._cache.get(pid)

            # Use cache if still fresh
            if cached and (frame_num - cached["last_frame"]) < self.ID_INTERVAL:
                results[pid] = (cached["name"], cached["number"], team)
                continue

            # Queue for batch number recognition (in the resolved team's roster)
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
                name, number = self._parse_number_results(
                    [batch_results[i]], team, crop_width=crop.shape[1],
                    pid=pid, frame=frame_num)
                _bx = getattr(batch_results[i], "boxes", None)
                # Number-anchored team vote (proposal A): ice-proof polarity read.
                # It votes strongly, then we re-resolve this track's team from the
                # updated tally and re-derive the name for the number in that team.
                nteam, nstrength = self._number_team(crop, _bx)
                if nteam is not None:
                    self._push_team_evidence(pid, nteam, nstrength)
                    v = self._team_votes.setdefault(pid, [0.0, 0.0])
                    if nteam == self._light_team:
                        v[0] += nstrength
                    else:
                        v[1] += nstrength
                    v2 = self._team_votes[pid]
                    team = self._light_team if v2[0] >= v2[1] else self._dark_team
                    if number is not None:
                        sq = self.Home if team == "Home" else self.Away
                        name = sq.get(str(number), name)
                self._collect_sample(pid, crop, batch_results[i], frame_num)
                results[pid] = (name, number, team)
                self._cache[pid] = {"name": name, "number": number, "team": team, "last_frame": frame_num}

                # --- Number-swap detection ---
                # A track whose freshly-read number decisively changes is the
                # tracker sitting on a DIFFERENT player. But a real player-swap
                # needs an OPPORTUNITY: two boxes must have crossed. So only split
                # when the track is inside a post-crossing window; a number change
                # with no crossing is OCR flicker — it must NOT fragment a clean,
                # long track (confidence-weighted voting + the per-number on-ice
                # constraint resolve the identity instead). This is the main lever
                # for LONGER clips. Compare against an established number (not the
                # cache, which we overwrite every read).
                if number is not None:
                    in_cross = frame_num <= self._crossed_until.get(pid, -1)
                    est = self._est_number.get(pid)
                    if est is None:
                        self._est_number[pid] = number
                    elif number != est and in_cross:
                        cnt = self._num_switch_counts.get(pid, 0) + 1
                        self._num_switch_counts[pid] = cnt
                        if cnt >= self.NUM_SWITCH_CONFIRM_CROSS:
                            number_switches.add(pid)
                            self._est_number[pid] = number
                            self._num_switch_counts.pop(pid, None)
                    else:
                        # same number, or a change with no crossing -> not a split
                        self._num_switch_counts.pop(pid, None)

        # --- Cross-TEAM swap detection ---
        # Establish each track's team once, then split it the moment the recent
        # window sustainedly shows the OTHER team: a jersey cannot change, so this
        # is the tracker sitting on a different player (body check, post-crossing
        # handoff) — the case the kinematic guard misses because the new player is
        # adjacent, not teleported.
        for pid in list(results.keys()):
            cur_team = results[pid][2]
            if self._est_team.get(pid) is None:
                if cur_team and cur_team != "Unknown":
                    self._est_team[pid] = cur_team
                continue
            new_team = self._team_swap_target(pid) if self.TEAM_SPLIT else None
            if new_team:
                team_switches.add(pid)
                self._apply_team_swap(pid, new_team)
                # The number/name still belong to the PREVIOUS player — drop them
                # so the fresh segment starts clean and re-reads the new jersey.
                results[pid] = ("Unknown", None, new_team)

        return results, team_switches, number_switches

    def _collect_sample(self, pid, crop_rgb, yolo_result, frame_num):
        """Save a training candidate: the crop image + a YOLO-format label built
        from the model's OWN digit detections (a pre-annotation to refine in
        Roboflow). Only crops where at least one digit was found are useful, and
        we cap per-track + overall so a run yields a diverse, bounded set.
        """
        if self._collect_dir is None or self._collected >= self._collect_cap:
            return
        boxes = getattr(yolo_result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return
        if self._collect_per_track.get(pid, 0) >= 6:  # diversity: few per track
            return
        h, w = crop_rgb.shape[:2]
        if h < 12 or w < 12:
            return
        lines = []
        for box, cls, conf in zip(boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist()):
            if conf < self.NUM_COLLECT_CONF:
                continue  # skip weak/garbage detections
            x1, y1, x2, y2 = box
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            digit = int(yolo_result.names[int(cls)])
            lines.append(f"{digit} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        if not lines:
            return
        stem = f"p{pid}_f{frame_num}"
        try:
            img_dir = self._collect_dir / "images"
            lbl_dir = self._collect_dir / "labels"
            cv2.imwrite(str(img_dir / f"{stem}.jpg"),
                        cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        except Exception:
            return
        self._collected += 1
        self._collect_per_track[pid] = self._collect_per_track.get(pid, 0) + 1

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
            tracks[pid] = {
                "team": self._majority(data.get("player_team", [])),
                "number": self.resolve_number(data.get("player_number", []),
                                              confs=data.get("player_number_conf")),
                "first_seen": data.get("first_seen"),
                "last_seen": data.get("last_seen"),
                "first_position": data.get("first_position"),
                "last_position": data.get("last_position"),
                "time_on_ice": toi,
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
            if t["team"] and t["number"] is not None:
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

            for pid_b in pids_by_time[i + 1:]:
                tb = tracks[pid_b]
                if tb["first_seen"] is None:
                    continue
                # Skip micro-tracks as targets too
                if tb["time_on_ice"] < self.MIN_TRACK_SEC:
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

                # A track that never earned a confident number of its own must
                # NOT inherit one by spatial association — it stays Unknown. So
                # only stitch when the two sides are on the same footing: both
                # carry a positive number, or both are number-less (Unknown +
                # Unknown is fine and remains Unknown). This stops number-less
                # strangers from being absorbed into a numbered identity (the
                # bloated "#4 Keane" card was 7 such number-less tracks).
                num_a = ta["number"] if ta["number"] is not None else gn_a
                num_b = tb["number"] if tb["number"] is not None else gn_b
                if (num_a is None) != (num_b is None):
                    continue

                merge_groups(pid_a, pid_b)

        # --- Pass 3: deduplicate same team+number across remaining groups ---
        # Use RAW per-track number votes so that ambiguous tracks (e.g. 50/50
        # #4/#99) can still be assigned to the correct identity.
        # CRITICAL: merge via merge_groups() so the temporal-overlap invariant
        # is preserved — two concurrent tracks that both read "#4" are two
        # different players, NOT one, and must never be fused.
        by_key = defaultdict(list)
        for pid in pids:
            src = players_data.get(pid, {})
            team_votes = [v for v in src.get("player_team", []) if v and v != "Unknown"]
            num_votes = [v for v in src.get("player_number", []) if v is not None]
            if not team_votes or not num_votes:
                continue
            team = Counter(team_votes).most_common(1)[0][0]
            if team == "Unknown":
                continue
            number = self.resolve_number(src.get("player_number", []), min_count=2,
                                         confs=src.get("player_number_conf"))
            if number is None:
                continue
            by_key[(team, number)].append(pid)

        for members in by_key.values():
            members.sort(key=lambda p: tracks[p].get("first_seen") or 0)
            for i in range(len(members) - 1):
                merge_groups(members[i], members[i + 1])  # no-op on overlap

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
                "first_seen": None, "first_position": None,
                "player_name": [], "player_number": [], "player_number_conf": [],
                "player_team": [], "frames": [],
            }
            for pid in sorted(group, key=lambda p: tracks[p].get("first_seen") or float("inf")):
                pid_mapping[pid] = canonical
                src = players_data[pid]
                merged["player_name"].extend(src.get("player_name", []))
                merged["player_number"].extend(src.get("player_number", []))
                merged["player_number_conf"].extend(src.get("player_number_conf", []))
                merged["player_team"].extend(src.get("player_team", []))
                merged["frames"].extend(src.get("frames", []))
                merged["time_on_ice"] += src.get("time_on_ice", 0)
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
    def resolve_number(num_votes, min_count=2, confs=None):
        """Confusion-aware, CONFIDENCE-WEIGHTED number resolution.

        Each read is weighted by its OCR confidence (default 1.0 when none is
        given, so behaviour matches plain frequency voting). This de-ranks a
        frequent-but-shaky misread: a noisy "9" seen at conf 0.45 loses to a
        confident "4" at 0.68, which pure frequency voting got wrong. Absolute
        floors (min_count, SINGLE_MIN) still use the RAW observation count so a
        single lucky high-confidence read can never win on its own.

        Missing a digit (seeing "4" for #14) is far more common than
        hallucinating one, so a minority of two-digit votes redistributes the
        matching single digit into the best two-digit number.
        """
        # Votes arrive as ints from the batch pipeline and as strings from live
        # (an operator types "10", an override stores it as text). The arithmetic
        # below — digit floors, 4 -> 44 redistribution — is integer arithmetic, so
        # normalize once here instead of trusting every caller to agree on a type.
        def _num(v):
            try:
                return int(str(v).strip())
            except (TypeError, ValueError):
                return None
        pairs = [(iv, (confs[i] if confs and i < len(confs) and confs[i] else 1.0))
                 for i, v in enumerate(num_votes) if (iv := _num(v)) is not None]
        if not pairs:
            return None
        wcounts, rcounts = Counter(), Counter()   # weighted score / raw obs count
        for v, c in pairs:
            wcounts[v] += c
            rcounts[v] += 1

        # Single-digit numbers are the noise / missing-digit class: a bare "4"
        # is far more often a stray read (or #44/#84 with a dropped digit) than
        # a real #4. Demand stronger evidence for them than for two-digit reads.
        SINGLE_MIN = 3        # absolute observations for a single-digit winner
        SINGLE_SHARE = 0.60   # and it must dominate the weighted score

        if len(wcounts) == 1:
            num = next(iter(wcounts))
            floor = max(min_count, SINGLE_MIN) if 0 <= num <= 9 else min_count
            return num if rcounts[num] >= floor else None

        adjusted = Counter(wcounts)   # weighted scores we rank on
        adj_obs = Counter(rcounts)    # observation counts, redistributed in step
        for num in list(wcounts.keys()):
            if not (0 <= num <= 9):
                continue
            related = {n: w for n, w in wcounts.items()
                       if 10 <= n <= 99 and (n // 10 == num or n % 10 == num)}
            if not related:
                continue
            related_obs = sum(rcounts[n] for n in related)
            # A modest amount of two-digit evidence absorbs the single digit.
            if related_obs >= 2 and sum(related.values()) >= wcounts[num] * 0.10:
                best_related = max(related, key=related.get)
                adjusted[best_related] += adjusted[num]; del adjusted[num]
                adj_obs[best_related] += adj_obs[num]; del adj_obs[num]

        if not adjusted:
            return None
        best, best_w = adjusted.most_common(1)[0]
        if adj_obs.get(best, 0) < min_count:
            return None
        # Ambiguity check: winner must be clearly ahead by weighted score.
        if len(adjusted) >= 2:
            _, second_w = adjusted.most_common(2)[1]
            if best_w < second_w * 1.5:
                return None
        # Single-digit winner must clear the stricter noise gate.
        if 0 <= best <= 9:
            total_w = sum(adjusted.values())
            if adj_obs[best] < SINGLE_MIN or best_w < total_w * SINGLE_SHARE:
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
# TOI helper
# ---------------------------------------------------------------------------

def union_toi(clips):
    """Real time-on-ice = union of clip intervals, not the sum.

    A player is one body: overlapping clips (a merge that fused concurrent
    tracks, or a fragment that re-detects mid-track) must count once, never
    twice. This bounds TOI by the covered timespan, so a bad merge can never
    produce a TOI longer than the video.
    """
    ivals = []
    for c in clips:
        s = c.get("first_seen")
        toi = c.get("toi", 0)
        if s is None or toi <= 0:
            continue
        ivals.append((s, s + toi))
    if not ivals:
        return 0.0
    ivals.sort()
    total = 0.0
    cur_s, cur_e = ivals[0]
    for s, e in ivals[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    total += cur_e - cur_s
    return total


# Per-run tracker tuning, shared by Automate and Live so the two cannot drift.
TRACK_BUFFER_SEC = 2.5   # survive a normal occlusion, but drop a lost track
                         # before it can re-link onto a look-alike
NEW_TRACK_THRESH = 0.70  # a confident new detection spawns its OWN id


def squad_lines(config_data):
    """Team structure as configured: defence pairs, then forward lines, each in
    the order the config lists them.

    Numbers come back as strings so they match the roster keys and the shift rows
    without any caller having to remember which side is an int. A config with no
    lines yields empty lists, and every consumer falls back to a flat list.
    """
    def groups(key):
        return [[str(n) for n in grp if n not in (None, "")]
                for grp in (config_data.get(key) or []) if grp]

    return {"Home": {"defense": groups("home_defense_pairs"),
                     "forward": groups("home_forward_lines")},
            "Away": {"defense": groups("away_defense_pairs"),
                     "forward": groups("away_forward_lines")}}


def effective_tracker(base_path, out_dir, fps):
    """Write the tracker config a run actually uses: the base yaml with
    track_buffer scaled to this video's fps.

    Both modes call this. They used to carry their own copy of the rewrite with
    slightly different floors, which is exactly the kind of difference nobody
    notices until the two modes disagree on the same footage. Returns the path to
    use (the base path unchanged if anything goes wrong — never fail a run over
    tuning).
    """
    import re as _re

    def _set(txt, key, val):
        if _re.search(rf'(?m)^\s*{key}\s*:', txt):
            return _re.sub(rf'(?m)^(\s*{key}\s*:).*$', rf'\1 {val}', txt)
        return txt + f'\n{key}: {val}\n'

    try:
        buf = max(int(fps * TRACK_BUFFER_SEC), 60)
        txt = open(base_path).read()
        txt = _set(txt, "track_buffer", buf)
        txt = _set(txt, "new_track_thresh", NEW_TRACK_THRESH)
        eff = out_dir / "tracker_effective.yaml"
        eff.write_text(txt)
        print(f"[tracker] {os.path.basename(base_path)} -> track_buffer={buf} "
              f"({TRACK_BUFFER_SEC}s @ {fps:.0f}fps), new_track_thresh={NEW_TRACK_THRESH}",
              flush=True)
        return str(eff)
    except Exception as e:
        print(f"[tracker] rewrite skipped: {e}", flush=True)
        return str(base_path)


def write_clip_range(src, dst, fps, start=0.0, end=None):
    """Write [start, end) seconds of a clip to dst, re-encoding frame by frame.

    Decodes from the top and skips rather than seeking: these clips are seconds
    long, and frame seeking on avc1 lands on keyframes — which would move the cut
    away from the point the reviewer picked, the whole purpose of the operation.
    Returns the number of frames written.
    """
    f_from = int(round((start or 0.0) * fps))
    f_to = int(round(end * fps)) if end is not None else None
    cap = cv2.VideoCapture(str(src))
    writer, idx, written = None, 0, 0
    while f_to is None or idx < f_to:
        ok, frame = cap.read()
        if not ok:
            break
        if idx >= f_from:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"avc1"),
                                         fps, (w, h))
            writer.write(frame)
            written += 1
        idx += 1
    cap.release()
    if writer is not None:
        writer.release()
    return written


def clip_confidence(numbers, confs, chosen, prior_reads=2.0):
    """How strongly a track's own number reads back the number we label it with.

    Each read is weighted by the detector's confidence in it (the same ranking
    resolve_number uses to pick `chosen`), and the result is shrunk toward 50% by
    `prior_reads` pseudo-reads, so three lucky frames report as ~80% rather than
    as certainty. Returns (percent, number of reads) — the read count is reported
    next to it because a share alone hides its own sample size.
    """
    confs = confs or []
    reads = [(n, float(confs[i]) if i < len(confs) and confs[i] else 1.0)
             for i, n in enumerate(numbers) if n is not None]
    if chosen is None or not reads:
        return None, len(reads)
    total = sum(w for _, w in reads)
    if total <= 0:
        return None, len(reads)
    backing = sum(w for n, w in reads if n == chosen)
    prior = prior_reads * (total / len(reads))     # pseudo-reads of average weight
    return round((backing + prior * 0.5) / (total + prior) * 100), len(reads)


CONF_BANDS = [(90, "90–100%"), (75, "75–90%"), (60, "60–75%"), (0, "<60%")]


def conf_band(pct):
    for lo, label in CONF_BANDS:
        if pct >= lo:
            return label
    return CONF_BANDS[-1][1]


def clip_parts(clip, splits, split_files=None, split_times=None):
    """Expand a review clip into the parts the reviewer cut it into.

    splits[track_id] = [cut offsets, seconds inside the clip file] — set by hand in
    review when the tracker jumped onto another player mid-clip. An uncut clip
    yields itself under its bare track_id, so decisions saved before any cut keep
    resolving. Mirrors the clipParts() keys the old clip-card review wrote.

    split_files[track_id] names the real files the cuts were written to; a part
    that has one IS that whole file. Parts without one fall back to a frame range
    of the original, which is what older saved decisions carry.

    split_times[track_id] are the same cuts as MATCH time. A clip file holds only
    the frames its track was seen on, so a second inside the file is not a second
    of the match; given the match times, each part gets the share of the track's
    real presence that is actually its own (and a part no longer loses the gaps).
    """
    tid = str(clip["track_id"])
    toi = float(clip.get("toi") or 0)
    cuts = sorted(c for c in (splits.get(tid) or []) if 0 < c < toi)
    if not cuts:
        return [(tid, clip)]
    files = (split_files or {}).get(tid) or []
    marks = sorted((split_times or {}).get(tid) or [])
    iv = clip.get("iv") or []
    bounds = [0.0] + cuts + [toi]
    parts = []
    for i in range(len(bounds) - 1):
        start, end = bounds[i], bounds[i + 1]
        part = dict(clip)
        if i < len(files):
            part["file"] = files[i]          # a file of its own: take all of it
        else:
            part["start"], part["end"] = start, end
        part.pop("iv", None)                 # the parent's presence covers both parts
        part["toi"] = round(end - start, 1)
        part["first_seen"] = round(float(clip.get("first_seen") or 0) + start, 1)
        if len(marks) == len(cuts) and iv:
            lo = marks[i - 1] if i else float("-inf")
            hi = marks[i] if i < len(marks) else float("inf")
            piv = [[max(a, lo), min(b, hi)] for a, b in iv if min(b, hi) > max(a, lo)]
            if piv:
                part["iv"] = [[round(a, 2), round(b, 2)] for a, b in piv]
                part["toi"] = round(sum(b - a for a, b in piv), 1)
                part["first_seen"] = round(piv[0][0], 1)
        parts.append((f"{tid}#{i}", part))
    return parts


def dedupe_by_number(clips, weak_votes=4, weak_share=0.5):
    """Enforce 'at most one player per number on ice at a time'.

    Clips of the same (team, number) that overlap in time cannot all be that
    player, so keep the strongest-evidence (num_share x votes, then TOI),
    non-overlapping timeline and reject the rest. A weak-number clip is also
    rejected once a stronger one is already kept (a lone weak clip is kept — it
    is likely the real player, just hard to read). Returns (kept, rejected);
    kept is sorted by first_seen.

    A clip the operator identified by hand ("fixed") is settled: it is placed
    first and never rejected. Overruling a typed-in identity with vote counts
    turned the operator's answer back into a guess — and if two fixed clips of
    one number overlap, that contradiction is the operator's to see, not ours
    to silently resolve.
    """
    def _strength(c):
        return (bool(c.get("fixed")),
                c.get("num_share", 0) * c.get("num_votes", 0), c.get("toi", 0))
    kept, kept_ivs, rejected = [], [], []
    for c in sorted(clips, key=_strength, reverse=True):
        s = c.get("first_seen", 0); e = s + c.get("toi", 0)
        if c.get("fixed"):
            kept.append(c); kept_ivs.append((s, e))
            continue
        overlaps = any(s < ke and ks < e for ks, ke in kept_ivs)
        weak = (c.get("num_votes", 0) < weak_votes
                or c.get("num_share", 0) < weak_share)
        if overlaps or (weak and kept):
            rejected.append(c)
        else:
            kept.append(c); kept_ivs.append((s, e))
    kept.sort(key=lambda c: c.get("first_seen", 0))
    return kept, rejected


def _dump_number_debug(records, result_dir):
    """Write the number-decision trace for logic-vs-model diagnosis.

    - number_debug.jsonl : one line per number read (raw YOLO digits + reason).
    - number_debug.json  : per-track summary — what the model detected vs what
      the logic chose — so a wrong number is quickly attributable:
        * model fault  -> YOLO rarely/never detected the right digit(s)
        * logic fault  -> right digits detected, but the pick / gate was wrong
    """
    try:
        with open(result_dir / "number_debug.jsonl", "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        per_track = {}
        for r in records:
            pid = r.get("pid")
            t = per_track.setdefault(str(pid), {
                "reads": 0, "team": r.get("team"),
                "digit_conf_sum": {}, "digit_count": {},
                "chosen": {}, "reasons": {},
            })
            t["reads"] += 1
            for dig, conf, _rel in r.get("dets", []):
                k = str(dig)
                t["digit_conf_sum"][k] = t["digit_conf_sum"].get(k, 0.0) + conf
                t["digit_count"][k] = t["digit_count"].get(k, 0) + 1
            ck = str(r.get("num"))
            t["chosen"][ck] = t["chosen"].get(ck, 0) + 1
            reason = r.get("reason", "?").split("(")[0].split(":")[0]
            t["reasons"][reason] = t["reasons"].get(reason, 0) + 1

        for t in per_track.values():
            t["digit_avg_conf"] = {
                k: round(t["digit_conf_sum"][k] / t["digit_count"][k], 2)
                for k in t["digit_count"]
            }
            del t["digit_conf_sum"]

        with open(result_dir / "number_debug.json", "w", encoding="utf-8") as f:
            json.dump({"per_track": per_track}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass  # diagnostics must never break a run


# ---------------------------------------------------------------------------
# Video Processing Job (background thread)
# ---------------------------------------------------------------------------

def set_msg(job, key, text, **vars):
    """The status line the job page shows: an i18n key with its variables, plus
    a plain fallback. The page is bilingual; a message written here is not."""
    job["message"] = text
    job["msg"] = {"k": key, "v": vars}


def process_video_job(job_id, video_path, config_data):
    """Run full detection + tracking pipeline in background."""
    job = processing_jobs[job_id]
    job["status"] = "running"
    set_msg(job, "job_m_models", "Загрузка моделей...")

    try:
        from ultralytics import YOLO
    except ImportError:
        job["status"] = "error"
        set_msg(job, "job_m_noultra", "ultralytics не установлен. Установите: pip install ultralytics")
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
        # NumbersRec stays PyTorch: it runs on a variable-size BATCH of player
        # crops per frame, and a CoreML export fixes batch=1, which breaks the
        # batched call (returns 1 result for N crops). Measured, not assumed.
        number_path = str(MODELS_DIR / "NumbersRec.pt")
        # Prefer BoT-SORT (GMC for camera pan/zoom + appearance ReID through
        # occlusions); fall back to plain ByteTrack, then the default tracker.
        tracker_stable = str(TRACKERS_DIR / "tracker_stable.yaml")
        tracker_fast = str(TRACKERS_DIR / "tracker_fast.yaml")
        tracker_orig = str(TRACKERS_DIR / "tracker.yaml")
        tracker_path = (tracker_stable if os.path.exists(tracker_stable)
                        else tracker_fast if os.path.exists(tracker_fast)
                        else tracker_orig)

        for p, label in [(detection_path, os.path.basename(detection_path)),
                         (number_path, "NumbersRec.pt"), (tracker_path, os.path.basename(tracker_path))]:
            if not os.path.exists(p):
                job["status"] = "error"
                set_msg(job, "job_m_nomodel", f"Модель не найдена: {label}", name=label)
                return

        use_coreml = detection_path.endswith(".mlpackage")
        if use_coreml:
            model_det = YOLO(detection_path, task="detect")
        else:
            model_det = YOLO(detection_path).to(device)
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
        # Training-data mining: collect digit crops + YOLO pre-annotations.
        candidates_dir = result_dir / "dataset_candidates"
        (candidates_dir / "images").mkdir(parents=True, exist_ok=True)
        (candidates_dir / "labels").mkdir(parents=True, exist_ok=True)
        identifier._collect_dir = candidates_dir

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            job["status"] = "error"
            set_msg(job, "job_m_novideo", "Не удалось открыть видеофайл")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        total_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Downsample every video to ~30 fps regardless of source fps: process only
        # every Nth source frame. Halves the work on 60fps broadcast and makes
        # clips lighter, with no loss for tracking at 30fps. `fps` below is the
        # EFFECTIVE rate used for all timing (track_buffer, TOI, clip output).
        TARGET_FPS = 30
        frame_stride = max(round(src_fps / TARGET_FPS), 1)
        fps = src_fps / frame_stride
        total_frames = total_src // frame_stride

        set_msg(job, "job_m_processing", "Обработка видео...")
        job["total_frames"] = total_frames

        # Speeds and distances are NOT computed (no rink calibration, no
        # LinesDetection model). The only movement check kept is the kinematic
        # SWAP GUARD: a box jumping more than this many of its OWN heights in one
        # frame (after camera-pan removal) is the tracker leaping onto a different
        # player, so we split the track. Height-relative + per-frame keeps it
        # robust across resolution AND fps (a px/sec threshold false-fires at 60fps).
        SWAP_BOX_HEIGHTS = 2.0             # jump > 2 body-heights/frame = tracker swapped players

        # Player data
        players_data = defaultdict(lambda: {
            "time_on_ice": 0, "last_seen": None, "last_position": None,
            "first_seen": None, "first_position": None,
            "player_name": [], "player_number": [], "player_number_conf": [],
            "player_team": [], "frames": [],
        })
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writers = {}

        # Track splitting: when a cross-team jump is detected, we create a
        # new virtual PID so the old track data stays clean.
        pid_remap = {}        # tracker_pid -> current virtual_pid
        next_virtual_pid = 100000  # virtual PIDs start high to avoid collisions
        # Swap-guard telemetry: how many suspected ID swaps we converted to splits.
        split_stats = {"team": 0, "kinematic": 0, "number": 0, "crossings": 0}
        cross_window = int(fps * identifier.CROSS_WINDOW_SEC)
        # Baseline toggle for A/B measurement of the swap guards.
        _swap_guards_off = bool(os.environ.get("PSF_SWAP_GUARDS_OFF"))

        # --- Speed optimization ---
        # DET_STRIDE: full identification every Nth frame (saves ~10ms on non-ID frames).
        # All frames are processed through YOLO tracker for tracking continuity.
        # CoreML on Apple Neural Engine provides ~2x speedup over MPS PyTorch.
        FRAME_SKIP = 1    # process every (downsampled-to-30fps) frame
        DET_STRIDE = 1    # identify on EVERY frame (video is already 30fps) for accuracy
        DET_IMGSZ = 640   # full resolution
        clip_fps = fps     # output clips at the effective (30) fps

        # --- Async video writing: offload I/O to a background thread ---
        import queue as _queue
        write_queue = _queue.Queue(maxsize=200)
        _write_stop = False

        # Per-team marker colour (broadcast style, same as Live control): from the
        # config jersey colour, falling back to the app palette for neutral kits.
        from live import APP_ORANGE_RGB, APP_CYAN_RGB

        def _team_bgr():
            def cfg_rgb(team):
                for nm in identifier.teams.get(team, []):
                    c = identifier.BASIC_COLORS.get(nm)
                    if c and (max(c) - min(c)) >= 45:
                        return tuple(int(v) for v in c)
                return None
            fb = {identifier._light_team: APP_ORANGE_RGB, identifier._dark_team: APP_CYAN_RGB}
            rgb = {"Home": cfg_rgb("Home") or fb["Home"], "Away": cfg_rgb("Away") or fb["Away"]}
            if rgb["Home"] == rgb["Away"]:
                rgb["Away"] = APP_ORANGE_RGB if rgb["Home"] != APP_ORANGE_RGB else APP_CYAN_RGB
            return {t: (c[2], c[1], c[0]) for t, c in rgb.items()}   # BGR for cv2
        team_bgr = _team_bgr()

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
                vpid, frame_data, box, team = item
                if vpid not in writers:
                    vp = str(videos_dir / f"player_{vpid}.mp4")
                    writers[vpid] = cv2.VideoWriter(vp, fourcc, clip_fps, (width, height))
                fc = frame_data.copy()
                # Broadcast-style highlight: a ground ellipse at the feet in the
                # team colour, NO text (no track/jersey number) — just the marker.
                color = team_bgr.get(team, (0, 200, 255))
                x1, y1, x2, y2 = box
                cx, bw = (x1 + x2) // 2, (x2 - x1)
                ew = max(int(bw * 0.5), 10)
                eh = max(int(ew * 0.35), 4)
                cv2.ellipse(fc, (cx, y2), (ew, eh), 0, 0, 360, color, 2, cv2.LINE_AA)
                writers[vpid].write(fc)

        wt = Thread(target=_writer_thread, daemon=True)
        wt.start()

        # DO NOT seek "back to the start" here. Nothing has read from `cap` yet,
        # and on some files cap.set(CAP_PROP_POS_FRAMES, 0) hands back a frame a
        # whole GOP later while still reporting position 1 — measured at 100
        # frames (4 s) on a 25 fps clip. Everything downstream is timed by the
        # frame counter, so those 4 s shifted every clip, every TOI and the whole
        # review overlay against the video the operator watches.
        frame_count = 0
        set_msg(job, "job_m_processing", "Обработка видео...")
        t_start = time.time()

        last_detections = []    # cached detections from last YOLO run
        last_id_results = {}    # cached identification results
        overlay_frames = []     # [t, [[clip id, x1, y1, x2, y2], ...]] per frame
        overlay_wh = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280,
                      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720)
        _prof = {"track": 0, "ident": 0, "data": 0, "n": 0}

        # Balance between LONG tracks and NOT swapping players. In hockey every
        # player wears the same kit, so ReID is weak and a LONG buffer makes a
        # lost track re-link onto a look-alike (id jumps between players). Keep the
        # buffer moderate and let a new player get a fresh id rather than inherit a
        # stale one. Same builder as Live — see effective_tracker().
        tracker_path = effective_tracker(tracker_path, result_dir, fps)

        # Prefetch: decode frames in a background thread so the accelerator never
        # waits on cap.read(). A small bounded queue overlaps decode with
        # inference and applies backpressure (no unbounded RAM growth). Only this
        # thread touches `cap` from here on.
        frame_q = _queue.Queue(maxsize=8)
        _read_stop = Event()

        def _reader():
            while not _read_stop.is_set():
                # skip (stride-1) frames cheaply (grab = no decode), then decode one
                skipped_ok = True
                for _ in range(frame_stride - 1):
                    if not cap.grab():
                        skipped_ok = False
                        break
                if not skipped_ok:
                    break
                ok, fr = cap.read()
                if not ok:
                    break
                # The decoder's own timestamp for this frame. Counting frames
                # from zero assumes the file starts at zero and loses nothing at
                # the head — on a real broadcast cut it did neither, and every
                # clip time drifted a second against the video in review.
                pos = cap.get(cv2.CAP_PROP_POS_MSEC) or 0
                while not _read_stop.is_set():
                    try:
                        frame_q.put((fr, pos / 1000.0), timeout=0.5)
                        break
                    except _queue.Full:
                        continue
            frame_q.put(None)        # sentinel: end of stream

        rt = Thread(target=_reader, daemon=True)
        rt.start()

        while True:
            item = frame_q.get()
            if item is None:         # reader reached end of video
                break
            frame, frame_t = item
            frame_count += 1
            current_time = frame_t if frame_t > 0 else frame_count / fps

            if frame_count % 30 == 0:
                job["progress"] = round(frame_count / max(total_frames, 1) * 100, 1)
                job["processed_frames"] = frame_count

            is_det_frame = (frame_count % DET_STRIDE == 1) or DET_STRIDE == 1

            _t0 = time.time()
            # Pin CUDA+FP16 explicitly on Windows/Linux NVIDIA; MPS/CoreML/CPU
            # keep ultralytics' own device auto-select (already correct).
            _dev_kw = {"device": device, "half": use_half} if device == "cuda" else {}
            results = model_det.track(
                frame, conf=0.6, persist=True, tracker=tracker_path,
                max_det=16, verbose=False, imgsz=DET_IMGSZ, **_dev_kw,
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
                # --- Crossing detection ---
                # When two boxes overlap, a swap can happen; flag both tracks
                # for post-crossing scrutiny (the number guard then trusts a
                # single contradicting read instead of requiring several).
                if not _swap_guards_off and len(detections) >= 2:
                    for _a in range(len(detections)):
                        pa, ax1, ay1, ax2, ay2 = detections[_a]
                        for _b in range(_a + 1, len(detections)):
                            pb, bx1_, by1_, bx2_, by2_ = detections[_b]
                            ix1 = max(ax1, bx1_); iy1 = max(ay1, by1_)
                            ix2 = min(ax2, bx2_); iy2 = min(ay2, by2_)
                            iw = ix2 - ix1; ih = iy2 - iy1
                            if iw <= 0 or ih <= 0:
                                continue
                            inter = iw * ih
                            union = ((ax2 - ax1) * (ay2 - ay1)
                                     + (bx2_ - bx1_) * (by2_ - by1_) - inter)
                            if union > 0 and inter / union >= identifier.CROSS_IOU:
                                until = frame_count + cross_window
                                identifier._crossed_until[pa] = until
                                identifier._crossed_until[pb] = until
                                split_stats["crossings"] += 1

                # Full identification only on detection frames; otherwise reuse cache
                if is_det_frame:
                    id_results, team_switches, number_switches = identifier.identify_batch(frame, detections, frame_count)
                    last_id_results = id_results

                    # Both team-colour and number swaps split the track into a
                    # fresh virtual PID so a contaminated identity can't persist.
                    if _swap_guards_off:
                        number_switches = set()
                    for raw_pid, kind in [(p, "team") for p in team_switches] + \
                                         [(p, "number") for p in number_switches if p not in team_switches]:
                        old_vpid = pid_remap.get(raw_pid, raw_pid)
                        if old_vpid in writers:
                            writers[old_vpid].release()
                            del writers[old_vpid]
                        pid_remap[raw_pid] = next_virtual_pid
                        next_virtual_pid += 1
                        split_stats[kind] += 1
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

                frame_boxes = []
                for pid, bx1, by1, bx2, by2 in detections:
                    name, number, team = id_results.get(pid, ("Unknown", None, "Unknown"))

                    vpid = pid_remap.get(pid, pid)
                    pd_entry = players_data[vpid]
                    cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2

                    # --- Kinematic swap guard ---
                    # A box that teleports beyond human skating speed is almost
                    # certainly the tracker jumping onto a DIFFERENT player. We
                    # convert the suspected swap into a split (new virtual PID)
                    # so a contaminated identity never persists — a split is
                    # recoverable by TrackMerger, a swap is not. Camera pan is
                    # already removed via the median residual (cam_dx/dy), so a
                    # large residual is a per-track outlier, not a pan. This is
                    # the same-team swap case the team-colour splitter misses.
                    if pd_entry["last_seen"] is not None and pd_entry["last_position"] is not None:
                        _td = current_time - pd_entry["last_seen"]
                        if _td > 0:
                            _lx, _ly = pd_entry["last_position"]
                            _res_px = math.hypot((cx - _lx) - cam_dx, (cy - _ly) - cam_dy)
                            # Scale-invariant swap test: measure the jump in units of
                            # the player's OWN height, per frame elapsed. A real swap
                            # lands on a player several body-heights away; box jitter
                            # is a small fraction of a height. This does NOT blow up at
                            # high fps the way a px/sec threshold does (which produced
                            # ~180 false splits/min at 60fps).
                            _box_h = max(by2 - by1, 1)
                            _frames = max(_td * fps, 1.0)          # usually 1 (all frames)
                            _limit = SWAP_BOX_HEIGHTS * _box_h * _frames
                            if _res_px > _limit and not _swap_guards_off:
                                if vpid in writers:
                                    writers[vpid].release()
                                    del writers[vpid]
                                vpid = next_virtual_pid
                                next_virtual_pid += 1
                                pid_remap[pid] = vpid
                                pd_entry = players_data[vpid]
                                split_stats["kinematic"] += 1

                    if is_det_frame:
                        if name not in (None, "None", "Unknown"):
                            pd_entry["player_name"].append(name)
                            pd_entry["player_number"].append(number)
                            # confidence of THIS frame's number read (for weighted voting)
                            _lg = identifier._last_guess.get(pid)
                            pd_entry["player_number_conf"].append(
                                _lg[1] if (_lg and _lg[0] == number) else 0.5)
                        if team:
                            pd_entry["player_team"].append(team)

                    if pd_entry["first_seen"] is None:
                        pd_entry["first_seen"] = current_time
                        pd_entry["first_position"] = (cx, cy)
                    if pd_entry["last_seen"] is not None:
                        pd_entry["time_on_ice"] += current_time - pd_entry["last_seen"]

                    pd_entry["last_seen"] = current_time
                    pd_entry["last_position"] = (cx, cy)

                    # Where this track was on this frame — review draws the same
                    # markers on the original video that live control draws live.
                    frame_boxes.append([vpid, bx1, by1, bx2, by2])

                    # Async video write (team drives the highlight colour)
                    try:
                        write_queue.put_nowait((vpid, frame, (bx1, by1, bx2, by2), team))
                    except _queue.Full:
                        pass
                overlay_frames.append([round(current_time, 2), frame_boxes])
                _t3 = time.time()

                _prof["track"] += (_t1 - _t0)
                _prof["ident"] += (_t2 - _t1)
                _prof["data"] += (_t3 - _t2)
                _prof["n"] += 1

        _read_stop.set()             # stop the prefetch reader and release cap
        rt.join(timeout=5)
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
        # Swap proxy: a track whose number votes are bimodal (>=2 numbers each
        # holding a real share) most likely contains two players — i.e. a swap
        # slipped through. Lower is better; use it to gauge the swap guards.
        def _contaminated(pdata):
            bad = 0
            for _pid, _d in pdata.items():
                votes = [v for v in _d.get("player_number", []) if v is not None]
                if len(votes) < 6:
                    continue
                c = Counter(votes)
                strong = [num for num, cnt in c.items() if cnt >= 0.25 * len(votes)]
                if len(strong) >= 2:
                    bad += 1
            return bad
        contaminated = _contaminated(players_data)

        profile_info = None
        if _prof["n"] > 0:
            n = _prof["n"]
            profile_info = {
                "yolo_track_ms": round(_prof["track"] / n * 1000, 1),
                "identify_ms": round(_prof["ident"] / n * 1000, 1),
                "data_write_ms": round(_prof["data"] / n * 1000, 1),
                "overhead_s": round(t_elapsed - _prof["track"] - _prof["ident"] - _prof["data"], 1),
                "processed_frames": frame_count,
                "splits_team": split_stats["team"],
                "splits_kinematic": split_stats["kinematic"],
                "splits_number": split_stats["number"],
                "crossings": split_stats["crossings"],
                "contaminated_tracks": contaminated,
            }
        job["profile"] = profile_info
        if profile_info:
            print(f"[Job {job_id}] {frame_count} frames in {t_elapsed:.1f}s = {processing_fps:.1f} FPS | "
                  f"track={profile_info['yolo_track_ms']}ms ident={profile_info['identify_ms']}ms "
                  f"data={profile_info['data_write_ms']}ms | splits team={split_stats['team']} "
                  f"kin={split_stats['kinematic']} num={split_stats['number']} | "
                  f"crossings={split_stats['crossings']} contaminated={contaminated}", flush=True)

        # Merge fragmented tracks
        set_msg(job, "job_m_merging", "Слияние фрагментированных треков...")
        merger = TrackMerger(fps)
        merged_data, pid_mapping = merger.merge(dict(players_data))

        n_before = len(players_data)
        n_after = len(merged_data)
        set_msg(job, "job_m_merged", f"Слияние: {n_before} треков -> {n_after} игроков",
                a=n_before, b=n_after)

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
            # Confusion-aware + confidence-weighted number resolution.
            number = TrackMerger.resolve_number(stats["player_number"],
                                                confs=stats.get("player_number_conf"))
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
        # Goalies are identified ONLY by explicit config numbers now — the old
        # movement heuristic needed speed, which is no longer computed.
        goalie_pids = set()
        for pid, stats in merged_data.items():
            info = player_info[pid]
            num_str = str(info["number"]) if info["number"] is not None else None
            if num_str:
                if info["team"] == "Home" and num_str in goalies_home:
                    goalie_pids.add(pid)
                elif info["team"] == "Away" and num_str in goalies_away:
                    goalie_pids.add(pid)

        # --- Build review manifest (no auto-merge) ---
        set_msg(job, "job_m_manifest", "Формирование манифеста для ревизии...")

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
                clip_number = TrackMerger.resolve_number(
                    pd_entry.get("player_number", []), min_count=2,
                    confs=pd_entry.get("player_number_conf"))
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
                # Skip short clips (< 1s) — noise, not worth reviewing
                clip_toi = pd_entry.get("time_on_ice", 0)
                if clip_toi < 1.0:
                    continue

                # Confidence in this clip's number: the share of the track's own
                # reads backing the number we actually label it with (NOT the most
                # frequent one — resolve_number weighs by detector confidence, so
                # those differ). dedupe_by_number ranks clips on this too.
                _nraw = pd_entry.get("player_number", [])
                _nvotes = [v for v in _nraw if v is not None]
                _nshare = (sum(1 for v in _nvotes if v == clip_number) / len(_nvotes)
                           if _nvotes and clip_number is not None else 0.0)
                _nconf, _ = clip_confidence(_nraw, pd_entry.get("player_number_conf"),
                                            clip_number)
                clips.append({
                    "track_id": raw_pid,
                    "file": f"player_{raw_pid}.mp4",
                    "toi": round(clip_toi, 1),
                    "first_seen": round(pd_entry.get("first_seen", 0), 1),
                    "auto_team": clip_team,
                    "auto_number": clip_number,
                    "auto_name": clip_name,
                    "num_votes": len(_nvotes),
                    "num_share": round(_nshare, 2),
                    "num_conf": _nconf,
                })

            if not clips:
                continue

            entry = {
                "canonical_pid": canonical_pid,
                "auto_team": info["team"],
                "auto_number": info["number"],
                "auto_name": info["name"],
                "total_toi": round(union_toi(clips), 1),
                "clips": clips,
            }
            if is_identified:
                groups.append(entry)
            else:
                unidentified.append(entry)

        # --- Post-merge: consolidate groups with same identity ---
        # ONE card per (team, number): to the reviewer, "#34 Ilyenko" is a single
        # player even when the system produced several concurrent tracks for that
        # number. We group them all together (the reviewer prunes the wrong clips
        # inside the card). TOI stays honest because union_toi counts overlapping
        # clips once, so a merged card can never exceed the real time on ice.
        consolidated = {}  # key (team, number) -> entry
        leftover = []      # groups without team+number go as-is
        for g in groups:
            key = (g["auto_team"], g["auto_number"])
            if key[0] and key[1] is not None:
                if key in consolidated:
                    consolidated[key]["clips"].extend(g["clips"])
                    consolidated[key]["total_toi"] = round(
                        union_toi(consolidated[key]["clips"]), 1)
                else:
                    consolidated[key] = g
            else:
                leftover.append(g)
        # --- Physical constraint (Fix 1) + weak-number gate (Fix 2) ---
        # Only ONE player can wear a given number on the ice at a time, so clips
        # of the same (team, number) that OVERLAP in time cannot all be that
        # player — a wrong-number fragment got merged in. Keep the strongest
        # (num_share x votes, then TOI), non-overlapping timeline; eject the rest
        # — plus any weak-number fragment once a stronger one is kept — to
        # "unidentified", where the reviewer assigns them instead of them
        # polluting one player (e.g. Valentsov #9: 6 concurrent -> 1 real).
        ejected_clips = []
        for g in consolidated.values():
            kept, rejected = dedupe_by_number(g["clips"])
            for c in rejected:                    # physically impossible / weak -> review as unidentified
                c["auto_number"] = None
                c["auto_name"] = None
            ejected_clips.extend(rejected)
            g["clips"] = kept
            g["total_toi"] = round(union_toi(kept), 1)
        consolidated = {k: g for k, g in consolidated.items() if g["clips"]}
        if ejected_clips:
            ejected_clips.sort(key=lambda c: c.get("first_seen", 0))
            unidentified.append({
                "canonical_pid": "ejected",
                "auto_team": None, "auto_number": None, "auto_name": None,
                "total_toi": round(union_toi(ejected_clips), 1),
                "clips": ejected_clips,
            })
        groups = list(consolidated.values()) + leftover

        # Diagnostics: how fragmented is the result (clips the reviewer must watch).
        _clips_id = sum(len(g["clips"]) for g in groups)
        _clips_unid = sum(len(g["clips"]) for g in unidentified)
        _n_players = len([g for g in groups if g.get("auto_number") is not None])
        _clips_per_player = round(_clips_id / _n_players, 1) if _n_players else 0
        if isinstance(job.get("profile"), dict):
            job["profile"].update({"clips_identified": _clips_id, "clips_unidentified": _clips_unid,
                                   "players": _n_players, "clips_per_player": _clips_per_player})
        print(f"[Job {job_id}] clips: {_clips_id} identified + {_clips_unid} unid | "
              f"{_n_players} players | {_clips_per_player} clips/player", flush=True)

        # On-ice shifts (same timeline as Live control). Each clip is a contiguous
        # track [first_seen, first_seen+toi]; overlapping/adjacent clips for one
        # (team, number) merge into solid segments, the rest is bridged/split by
        # the shared shift builder.
        from live import build_shifts, FLICKER_GAP
        presence = {}
        names_by_key = {}
        for g in groups:
            tm, num = g.get("auto_team"), g.get("auto_number")
            if not tm or num is None:
                continue
            key = (tm, str(num))
            names_by_key[key] = g.get("auto_name") or ""
            for c in g["clips"]:
                s = c.get("first_seen", 0)
                presence.setdefault(key, []).append([s, s + c.get("toi", 0)])
        for key, iv in presence.items():
            iv.sort()
            merged = []
            for s, e in iv:
                if merged and s - merged[-1][1] <= FLICKER_GAP:
                    merged[-1][1] = max(merged[-1][1], e)
                else:
                    merged.append([s, e])
            presence[key] = merged

        try:
            (result_dir / "overlay.json").write_text(json.dumps(
                {"w": overlay_wh[0], "h": overlay_wh[1], "frames": overlay_frames}),
                encoding="utf-8")
        except Exception:
            pass                   # an overlay is a nicety; the review is not

        manifest = {
            "fps": fps,
            "source_video": video_path,       # review replays the whole match
            "team_colors": live_mod.team_hex(identifier),
            "groups": groups,
            "unidentified": unidentified,
            "shifts": build_shifts(presence, names_by_key),
            "profile": job.get("profile"),     # timing + fragmentation diagnostics
            "roster": {
                "Home": home_squad,
                "Away": away_squad,
            },
            # Pairs/lines as configured, so the shift list can be read by unit
            # instead of as one long alphabetical column.
            "lines": squad_lines(config_data),
        }
        manifest_path = result_dir / "review_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        # --- Number-decision debug dump (logic-vs-model diagnosis) ---
        if identifier._num_debug:
            _dump_number_debug(identifier._num_debug, result_dir)

        job["status"] = "review"
        set_msg(job, "job_m_ready", f"Обработано {frame_count} кадров. Готово к ревизии.",
                n=frame_count)
        job["result_path"] = str(result_dir)
        job["progress"] = 100

    except Exception as e:
        job["status"] = "error"
        set_msg(job, "job_m_error", f"Ошибка: {e}", err=str(e))
        try:
            _read_stop.set()         # don't leak the prefetch thread on error
        except NameError:
            pass


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
        return redirect(url_for("analysis_page"))

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
        "message": "В очереди...", "msg": {"k": "job_m_queued", "v": {}},
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
                "message": "Готово к ревизии.", "msg": {"k": "job_m_review", "v": {}},
                "result_path": str(result_dir),
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
                "message": "Готово к ревизии.", "msg": {"k": "job_m_review", "v": {}},
                "result_path": str(result_dir),
                "stats": None, "total_frames": 0, "processed_frames": 0, "video": "",
            }
            job = processing_jobs[job_id]
        else:
            return redirect(url_for("index"))
    # Both modes are reviewed the same way the operator worked in live control:
    # the whole video with the tracks marked, not a queue of clip cards.
    return render_template("review_live.html", job_id=job_id)


@app.route("/api/job/<job_id>/shifts", methods=["POST"])
def api_job_shifts(job_id):
    """The shift timeline for a set of review decisions.

    Built with the SAME builder the live session uses, from the clips' own
    intervals — so what review draws and what the finalized result holds cannot
    drift apart, and an edit shows up on the chart immediately."""
    manifest_path = RESULTS_FOLDER / job_id / "review_manifest.json"
    if not manifest_path.exists():
        return jsonify({"error": "Manifest not found"}), 404
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    body = request.get_json(force=True) or {}
    # Either the bare decision map, or it wrapped together with the manual cuts
    # that split a clip into parts (the page sends the second form).
    decisions = body.get("decisions") if isinstance(body.get("decisions"), dict) else body
    splits = body.get("splits") or {} if isinstance(body, dict) else {}
    split_times = body.get("split_times") or {} if isinstance(body, dict) else {}
    from live import build_shifts
    roster = manifest.get("roster", {})
    presence, names = {}, {}
    groups = [(g, False) for g in manifest.get("groups", [])]
    groups += [(g, True) for g in manifest.get("unidentified", [])]
    for group, unid in groups:
        for whole in group.get("clips", []):
            for ckey, clip in clip_parts(whole, splits, None, split_times):
                d = decisions.get(ckey) or {}
                action = d.get("action") or ("ignore" if unid else "approve")
                if action in ("reject", "ignore"):
                    continue
                if action in ("reassign", "assign"):
                    team, number = d.get("target_team"), d.get("target_number")
                    name = d.get("target_name") or ""
                else:
                    team, number = group.get("auto_team"), group.get("auto_number")
                    name = group.get("auto_name") or ""
                if not team or number in (None, ""):
                    continue
                key = (team, str(number))
                iv = clip.get("iv") or [[clip.get("first_seen", 0),
                                         clip.get("first_seen", 0) + clip.get("toi", 0)]]
                presence.setdefault(key, []).extend([float(a), float(b)] for a, b in iv)
                names[key] = name or (roster.get(team, {}) or {}).get(str(number), "")
    return jsonify({"shifts": build_shifts(presence, names)})


@app.route("/api/job/<job_id>/source")
def api_job_source(job_id):
    """The original video of a run, so review can replay the moment a player was
    on the ice. Served with range support (send_file does it) for seeking."""
    manifest = RESULTS_FOLDER / job_id / "review_manifest.json"
    src = ""
    if manifest.exists():
        with open(manifest, encoding="utf-8") as f:
            src = (json.load(f) or {}).get("source_video", "")
    if not src:
        src = (processing_jobs.get(job_id) or {}).get("video", "")
    path = Path(src)
    if not path.is_absolute():
        path = UPLOAD_FOLDER / path.name
    if not src or not path.exists():
        return jsonify({"error": "Исходное видео не найдено"}), 404
    return send_file(str(path), conditional=True)


@app.route("/result/<job_id>")
def result_page(job_id):
    """Final result of a finalized run: per-player videos (sorted) + shifts."""
    if not (RESULTS_FOLDER / job_id / "final_result.json").exists():
        # not finalized yet -> send to review (or index if nothing there)
        if (RESULTS_FOLDER / job_id / "review_manifest.json").exists():
            return redirect(url_for("review_page", job_id=job_id))
        return redirect(url_for("index"))
    return render_template("result.html", job_id=job_id)


@app.route("/api/job/<job_id>/result")
def api_result_get(job_id):
    p = RESULTS_FOLDER / job_id / "final_result.json"
    if not p.exists():
        return jsonify({"error": "Not finalized"}), 404
    with open(p, "r", encoding="utf-8") as f:
        return jsonify(json.load(f))


@app.route("/results")
def results_page():
    """New Results section: finalized runs, each opening the per-player result."""
    return render_template("results.html")


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


@app.route("/api/job/<job_id>/cut", methods=["POST"])
def api_review_cut(job_id):
    """Write a reviewer's manual cuts of one track out as real clip files.

    A part used to be the whole file replayed between two offsets, so the player's
    scrubber still showed the full clip and the halves looked identical. Each part
    gets its own file instead, and its length is then the length the reviewer sees.
    Posting no cuts removes the part files (the reviewer merged them back).
    """
    result_dir = RESULTS_FOLDER / job_id
    videos_dir = result_dir / "player_videos"
    if not videos_dir.is_dir():
        return jsonify({"error": "Job not found"}), 404

    body = request.get_json() or {}
    tid = str(body.get("track_id", ""))
    if not tid.isdigit():
        return jsonify({"error": "Bad track_id"}), 400
    src = videos_dir / f"player_{tid}.mp4"
    if not src.exists():
        return jsonify({"error": "Clip not found"}), 404

    for stale in videos_dir.glob(f"player_{tid}__p*.mp4"):   # always rebuilt whole
        stale.unlink()
    cuts = sorted({round(float(c), 2) for c in (body.get("cuts") or []) if float(c) > 0})
    if not cuts:
        return jsonify({"files": []})

    fps = 25
    manifest_path = result_dir / "review_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            fps = json.load(f).get("fps", 25) or 25

    bounds = [0.0] + cuts + [None]     # last part runs to the end of the file
    files = []
    for i in range(len(bounds) - 1):
        dst = videos_dir / f"player_{tid}__p{i}.mp4"
        frames = write_clip_range(src, dst, fps, bounds[i] or 0.0, bounds[i + 1])
        if not frames:                 # a cut past the end of the footage
            dst.unlink(missing_ok=True)
            for done in files:
                (videos_dir / done["file"]).unlink(missing_ok=True)
            return jsonify({"error": "Cut falls outside the clip"}), 400
        files.append({"file": dst.name, "seconds": round(frames / fps, 1)})
    return jsonify({"files": files})


@app.route("/api/job/<job_id>/finalize", methods=["POST"])
def api_finalize(job_id):
    result_dir = RESULTS_FOLDER / job_id
    job = processing_jobs.get(job_id)
    if not job:
        if not (result_dir / "review_manifest.json").exists():
            return jsonify({"error": "Job not found"}), 404
        processing_jobs[job_id] = {
            "status": "review", "progress": 100, "message": "Готово к ревизии.",
            "msg": {"k": "job_m_review", "v": {}}, "result_path": str(result_dir),
            "stats": None, "total_frames": 0, "processed_frames": 0, "video": "",
        }
        job = processing_jobs[job_id]
    manifest_path = result_dir / "review_manifest.json"
    decisions_path = result_dir / "review_decisions.json"
    if not manifest_path.exists() or not decisions_path.exists():
        return jsonify({"error": "Missing manifest or decisions"}), 400

    # Pressing the button again while it runs used to start a SECOND render of
    # every player over the first one's files.
    if job.get("status") == "finalizing":
        return jsonify({"status": "ok", "already": True})

    job["status"] = "finalizing"
    set_msg(job, "job_m_finalizing", "Finalizing…")
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
        splits = decisions.get("splits", {})            # manual mid-clip cuts
        split_files = decisions.get("split_files", {})   # ...written out as real files
        split_times = decisions.get("split_times", {})   # ...and where they fall in the match

        # Build final player groups: player_key -> list of clips
        # player_key = "Team_Number_Name"
        final_groups = {}  # key -> {"team", "number", "name", "clips": [...]}

        def player_key(team, number, name):
            return f"{team}_{number}_{name}"

        def add_clip(key, team, number, name, clip):
            if key not in final_groups:
                final_groups[key] = {"team": team, "number": number, "name": name, "clips": []}
            final_groups[key]["clips"].append(clip)

        # Accuracy counters. calib answers the long-run question the per-clip
        # confidence raises: when the system says 90%, is it right 90% of the time?
        calib = defaultdict(lambda: {"n": 0, "correct": 0})
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

            for key, clip in [kc for c in group["clips"]
                              for kc in clip_parts(c, splits, split_files, split_times)]:
                action = grp_dec.get(key, {}).get("action", "approve")
                total_reviewed += 1

                if clip.get("num_conf") is not None:
                    band = calib[conf_band(clip["num_conf"])]
                    band["n"] += 1
                    band["correct"] += action == "approve"

                if action == "approve":
                    add_clip(grp_key, grp_team, grp_number, grp_name, clip)
                    correct += 1
                elif action == "reject":
                    rejected += 1
                elif action == "reassign":
                    dec = grp_dec[key]
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
            for key, clip in [kc for c in group["clips"]
                              for kc in clip_parts(c, splits, split_files, split_times)]:
                dec = unid_decisions.get(key, {})
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

        # Merge videos per final group. Every clip is decoded and re-encoded
        # here, so this is minutes of work — report it per player, or the caller
        # (Live control above all) has nothing to show but a frozen screen.
        final_videos_dir = result_dir / "player_videos_final"
        final_videos_dir.mkdir(exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        stats_list = []
        final_players = []          # per-player final video + metrics (for the Results page)

        n_groups = len(final_groups)
        job["total_frames"] = n_groups
        for done, (key, group) in enumerate(final_groups.items()):
            job["processed_frames"] = done
            set_msg(job, "job_m_render", f"Rendering player videos… {done + 1}/{n_groups}",
                    a=done + 1, b=n_groups)
            clips = sorted(group["clips"], key=lambda c: c.get("first_seen", 0))
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
            for clip in clips:
                src = str(videos_dir / clip["file"])
                if not os.path.exists(src):
                    continue
                # A cut part that was never written to its own file still carries a
                # frame range of the original; otherwise this takes the whole file.
                src_cap = cv2.VideoCapture(src)
                f_from = int(round(float(clip.get("start") or 0) * fps))
                f_to = int(round(clip["end"] * fps)) if clip.get("end") is not None else None
                idx = 0
                while f_to is None or idx < f_to:
                    ret, fr = src_cap.read()
                    if not ret:
                        break
                    if idx >= f_from:
                        if out_writer is None:
                            h, w = fr.shape[:2]
                            out_writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
                        out_writer.write(fr)
                    idx += 1
                src_cap.release()

            # TOI = union of clip intervals, never the sum (one body, one clock)
            total_toi = union_toi(clips)

            if out_writer is not None:
                out_writer.release()

            stats_list.append({
                "Name": name,
                "Number": number,
                "Team": team,
                "Time (s)": round(total_toi),
            })
            if out_writer is not None:           # a video was actually written
                final_players.append({
                    "team": team, "number": number, "name": name,
                    "file": fname, "toi": round(total_toi),
                })

        # Generate Excel
        job["processed_frames"] = n_groups
        set_msg(job, "job_m_stats", "Writing statistics…")
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
            # Measured accuracy per confidence band — compare across runs to see
            # whether the reported confidence is honest.
            "calibration": [
                {"band": label, "n": calib[label]["n"],
                 "correct_pct": round(calib[label]["correct"] / calib[label]["n"] * 100, 1)}
                for _, label in CONF_BANDS if calib[label]["n"]
            ],
        }
        with open(result_dir / "accuracy_report.json", "w", encoding="utf-8") as f:
            json.dump(accuracy, f, ensure_ascii=False, indent=2)

        # Final result for the Results page: players sorted by (team, number) +
        # the same On-ice shifts timeline shown in Live control.
        def _numkey(n):
            try:
                return (0, int(n))
            except (TypeError, ValueError):
                return (1, str(n))
        final_players.sort(key=lambda p: (p["team"] or "", _numkey(p["number"])))

        # On-ice shifts recomputed from the FINAL clip assignments (after review
        # decisions), so the timeline matches the per-player videos exactly.
        from live import build_shifts, FLICKER_GAP
        fpres, fnames = {}, {}
        for group in final_groups.values():
            tm, num = group["team"], group["number"]
            if not tm or num is None or tm == "Unknown":
                continue
            fk = (tm, str(num))
            fnames[fk] = group.get("name") or ""
            for c in group["clips"]:
                s = c.get("first_seen", 0)
                fpres.setdefault(fk, []).append([s, s + c.get("toi", 0)])
        for fk, iv in fpres.items():
            iv.sort()
            merged = []
            for s, e in iv:
                if merged and s - merged[-1][1] <= FLICKER_GAP:
                    merged[-1][1] = max(merged[-1][1], e)
                else:
                    merged.append([s, e])
            fpres[fk] = merged

        with open(result_dir / "final_result.json", "w", encoding="utf-8") as f:
            json.dump({"players": final_players, "shifts": build_shifts(fpres, fnames),
                       "accuracy": accuracy, "fps": fps,
                       "lines": manifest.get("lines") or {}}, f, ensure_ascii=False, indent=2)

        job["status"] = "completed"
        set_msg(job, "job_m_done", f"Done! Auto-labeling accuracy: {accuracy_pct}%",
                pct=accuracy_pct)
        job["stats"] = stats_list
        job["accuracy"] = accuracy

    except Exception as e:
        job["status"] = "error"
        set_msg(job, "job_m_fin_error", f"Finalization error: {e}", err=str(e))


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
    for name in ["Player_Detection.pt", "NumbersRec.pt"]:
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

    jobs.sort(key=lambda j: j.get("created", 0), reverse=True)   # newest first, by date
    return jsonify(jobs)


@app.route("/api/job/<job_id>", methods=["DELETE"])
def api_job_delete(job_id):
    """Delete a run from history (Archive). Guards against path traversal."""
    target = (RESULTS_FOLDER / job_id).resolve()
    if target.parent != RESULTS_FOLDER.resolve() or not target.is_dir():
        return jsonify({"error": "Invalid job"}), 400
    shutil.rmtree(target, ignore_errors=True)
    processing_jobs.pop(job_id, None)
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# Training-data mining (NumbersRec fine-tuning) — Roboflow-ready
# ---------------------------------------------------------------------------
NUM_CLASSES = [str(i) for i in range(10)]


def _count_images(d):
    img = d / "images"
    return len(list(img.glob("*.jpg"))) if img.exists() else 0


@app.route("/dataset")
def dataset_page():
    """Training-data mining tab."""
    return render_template("dataset.html")


@app.route("/api/dataset/overview")
def api_dataset_overview():
    """Global staging count + runs whose crops are not yet added."""
    staged = _count_images(TRAINING_FOLDER)
    pending = []
    if RESULTS_FOLDER.exists():
        for d in sorted(RESULTS_FOLDER.iterdir(), reverse=True):
            cand = d / "dataset_candidates"
            n = _count_images(cand)
            if n > 0 and not (cand / ".added").exists():
                pending.append({"job_id": d.name, "count": n,
                                "created": d.stat().st_mtime})
    return jsonify({"staged": staged, "pending": pending})


@app.route("/api/dataset/images")
def api_dataset_images():
    """Paginated list of staged image names for the gallery."""
    img_dir = TRAINING_FOLDER / "images"
    names = sorted([p.name for p in img_dir.glob("*.jpg")]) if img_dir.exists() else []
    offset = int(request.args.get("offset", 0))
    limit = min(int(request.args.get("limit", 120)), 500)
    return jsonify({"total": len(names), "names": names[offset:offset + limit]})


@app.route("/api/dataset/img/<path:name>")
def api_dataset_img(name):
    """Serve a staged crop (optionally a specific run's candidate via ?job=)."""
    job = request.args.get("job")
    base = (RESULTS_FOLDER / job / "dataset_candidates") if job else TRAINING_FOLDER
    return send_from_directory(str(base / "images"), name)


@app.route("/api/dataset/add/<job_id>", methods=["POST"])
def api_dataset_add(job_id):
    """Copy a run's mined crops+labels into the global staging set."""
    cand = RESULTS_FOLDER / job_id / "dataset_candidates"
    if not (cand / "images").exists():
        return jsonify({"error": "no candidates for this run"}), 404
    added = 0
    for img in (cand / "images").glob("*.jpg"):
        lbl = cand / "labels" / f"{img.stem}.txt"
        dst_img = TRAINING_FOLDER / "images" / f"{job_id}_{img.name}"
        dst_lbl = TRAINING_FOLDER / "labels" / f"{job_id}_{img.stem}.txt"
        try:
            shutil.copy2(img, dst_img)
            if lbl.exists():
                shutil.copy2(lbl, dst_lbl)
            added += 1
        except Exception:
            continue
    (cand / ".added").write_text(str(added))
    return jsonify({"status": "ok", "added": added,
                    "staged": _count_images(TRAINING_FOLDER)})


@app.route("/api/dataset/delete", methods=["POST"])
def api_dataset_delete():
    """Remove specific staged crops (+ their labels) — manual garbage cleanup."""
    data = request.get_json(force=True) or {}
    names = data.get("names", [])
    img_dir = TRAINING_FOLDER / "images"
    lbl_dir = TRAINING_FOLDER / "labels"
    removed = 0
    for name in names:
        base = Path(name).name  # strip any path component
        img = img_dir / base
        if img.exists() and img.parent == img_dir:
            try:
                img.unlink()
                lbl = lbl_dir / f"{Path(base).stem}.txt"
                if lbl.exists():
                    lbl.unlink()
                removed += 1
            except Exception:
                continue
    return jsonify({"status": "ok", "removed": removed,
                    "staged": _count_images(TRAINING_FOLDER)})


@app.route("/api/dataset/clear", methods=["POST"])
def api_dataset_clear():
    """Empty the global staging set (after a successful Roboflow upload)."""
    for sub in ("images", "labels"):
        d = TRAINING_FOLDER / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    # allow re-adding runs
    for r in RESULTS_FOLDER.iterdir():
        marker = r / "dataset_candidates" / ".added"
        if marker.exists():
            marker.unlink()
    return jsonify({"status": "ok"})


def _build_dataset_zip(base_dir, download_name):
    """Pack a {images,labels} dir into a Roboflow-ready YOLOv8 zip response."""
    import io
    import zipfile
    img_dir = base_dir / "images"
    lbl_dir = base_dir / "labels"
    names_yaml = "[" + ", ".join(f"'{c}'" for c in NUM_CLASSES) + "]"
    data_yaml = ("train: train/images\nval: train/images\n"
                 f"nc: {len(NUM_CLASSES)}\nnames: {names_yaml}\n")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("data.yaml", data_yaml)
        for img in sorted(img_dir.glob("*.jpg")):
            z.write(img, f"train/images/{img.name}")
            lbl = lbl_dir / f"{img.stem}.txt"
            if lbl.exists():
                z.write(lbl, f"train/labels/{lbl.name}")
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True,
                     download_name=download_name)


@app.route("/api/dataset/export")
def api_dataset_export():
    """Download the staging set as a Roboflow-ready YOLOv8 zip."""
    return _build_dataset_zip(TRAINING_FOLDER, "numbersrec_dataset.zip")


@app.route("/api/dataset/archive", methods=["POST"])
def api_dataset_archive():
    """Snapshot the current staging set into a timestamped archive and empty
    staging. Use after uploading to Roboflow: future additions start fresh, so
    re-exporting never re-includes already-uploaded frames. Unlike Clear, the
    per-run .added markers are KEPT (archived runs must not be re-added)."""
    import time as _time
    if _count_images(TRAINING_FOLDER) == 0:
        return jsonify({"error": "датасет пуст — нечего архивировать"}), 400
    stamp = _time.strftime("%Y%m%d_%H%M%S")
    arch = TRAINING_FOLDER / "archive" / stamp
    arch.mkdir(parents=True, exist_ok=True)
    shutil.move(str(TRAINING_FOLDER / "images"), str(arch / "images"))
    shutil.move(str(TRAINING_FOLDER / "labels"), str(arch / "labels"))
    (TRAINING_FOLDER / "images").mkdir(parents=True, exist_ok=True)
    (TRAINING_FOLDER / "labels").mkdir(parents=True, exist_ok=True)
    return jsonify({"status": "ok", "archive": stamp, "count": _count_images(arch)})


@app.route("/api/dataset/archives")
def api_dataset_archives():
    """List archived batches (name, count, date)."""
    root = TRAINING_FOLDER / "archive"
    out = []
    if root.exists():
        for d in sorted(root.iterdir(), reverse=True):
            if d.is_dir():
                out.append({"name": d.name, "count": _count_images(d),
                            "created": d.stat().st_mtime})
    return jsonify(out)


@app.route("/api/dataset/archive_export/<name>")
def api_dataset_archive_export(name):
    """Re-download an archived batch as a Roboflow-ready zip."""
    arch = TRAINING_FOLDER / "archive" / Path(name).name
    if not (arch / "images").exists():
        return jsonify({"error": "archive not found"}), 404
    return _build_dataset_zip(arch, f"numbersrec_{Path(name).name}.zip")


@app.route("/api/dataset/archive_delete/<name>", methods=["POST"])
def api_dataset_archive_delete(name):
    """Permanently delete an archived batch."""
    arch = TRAINING_FOLDER / "archive" / Path(name).name
    if arch.exists() and arch.parent == TRAINING_FOLDER / "archive":
        shutil.rmtree(arch)
    return jsonify({"status": "ok"})


@app.route("/api/dataset/roboflow", methods=["POST"])
def api_dataset_roboflow():
    """Best-effort direct upload to an existing Roboflow project."""
    data = request.get_json(force=True) or {}
    api_key = data.get("api_key", "").strip()
    workspace = data.get("workspace", "").strip()
    project = data.get("project", "").strip()
    if not (api_key and workspace and project):
        return jsonify({"error": "api_key, workspace и project обязательны"}), 400
    try:
        from roboflow import Roboflow
    except ImportError:
        return jsonify({"error": "roboflow SDK не установлен. Установи "
                        "`pip install roboflow` или используй экспорт zip."}), 400
    try:
        rf = Roboflow(api_key=api_key)
        proj = rf.workspace(workspace).project(project)
        img_dir = TRAINING_FOLDER / "images"
        lbl_dir = TRAINING_FOLDER / "labels"
        uploaded = 0
        for img in sorted(img_dir.glob("*.jpg")):
            lbl = lbl_dir / f"{img.stem}.txt"
            proj.upload(image_path=str(img),
                        annotation_path=str(lbl) if lbl.exists() else None)
            uploaded += 1
        return jsonify({"status": "ok", "uploaded": uploaded})
    except Exception as e:
        return jsonify({"error": f"Roboflow: {e}"}), 500


# ---------------------------------------------------------------------------
# Live review — play the video through the tracker, correct any track inline
# while it runs, log every correction as ground truth. One session at a time.
# ---------------------------------------------------------------------------
import live as live_mod

live_session = None  # module-global: single active LiveSession


def _load_live_pipeline(config_data):
    """Build (model_det, identifier, tracker_path, det_imgsz) for a live session.
    ponytail: ~25 lines duplicated from process_video_job's loader; extracting a
    shared helper would touch the batch path — not worth the risk for a drop-in."""
    from ultralytics import YOLO
    try:
        import torch
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                  else "cpu")
    except ImportError:
        device = "cpu"

    det_coreml = str(MODELS_DIR / "Player_Detection.mlpackage")
    det_pt = str(MODELS_DIR / "Player_Detection.pt")
    det_path = det_coreml if os.path.exists(det_coreml) and device == "mps" else det_pt
    num_path = str(MODELS_DIR / "NumbersRec.pt")
    # SAME tracker as Automate. Live used to run a looser config of its own
    # (match_thresh 0.80 and appearance_thresh 0.30, against 0.85/0.25 here)
    # because it processed ~8 of every 60 frames and needed ReID to bridge the
    # gaps. It now processes every frame at ~30fps exactly like Automate, so that
    # looseness only bought extra id switches on identical hockey kits.
    tracker_path = next((str(TRACKERS_DIR / t) for t in
                         ("tracker_stable.yaml", "tracker_fast.yaml", "tracker.yaml")
                         if os.path.exists(TRACKERS_DIR / t)), str(TRACKERS_DIR / "tracker.yaml"))
    for p in (det_path, num_path):
        if not os.path.exists(p):
            raise FileNotFoundError(os.path.basename(p))

    model_det = YOLO(det_path, task="detect") if det_path.endswith(".mlpackage") else YOLO(det_path).to(device)
    model_num = YOLO(num_path).to(device)

    # FP16 on CUDA (Windows/Linux NVIDIA) — ~2x speedup, matches Automate mode.
    # CoreML (Apple) runs on the Neural Engine; MPS/CPU stay FP32.
    use_half = device == "cuda"
    if use_half:
        model_det.model.half()
        model_num.model.half()
    print(f"[Live] Device: {device}, CoreML det: {det_path.endswith('.mlpackage')}, FP16: {use_half}", flush=True)

    teams_colors_info = config_data.get("teams_colors_info", {"Home": ["blue"], "Away": ["white", "red"]})
    basic_colors = {}
    for name, rgb in config_data.get("basic_colors", BASIC_COLORS).items():
        basic_colors[name] = tuple(rgb) if isinstance(rgb, list) else rgb
    identifier = PlayerIdentifier(teams_colors_info, config_data.get("home_team", {}),
                                  config_data.get("away_team", {}), basic_colors or BASIC_COLORS, model_num)
    return model_det, identifier, tracker_path, 640, device, use_half


@app.route("/analysis")
def analysis_page():
    """One start page for both modes: a checkbox picks live control or automatic."""
    return render_template("analysis.html")


@app.route("/live")
def live_page():
    return redirect(url_for("analysis_page"))


@app.route("/live/start", methods=["POST"])
def live_start():
    global live_session
    if live_session and live_session.status in ("playing", "paused"):
        live_session.stop()

    video_file = request.files.get("video")
    if not video_file or not allowed_file(video_file.filename, ALLOWED_VIDEO):
        return jsonify({"error": "Загрузите видеофайл (mp4, avi, mov, mkv)"}), 400
    filename = secure_filename(video_file.filename)
    video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    video_file.save(video_path)

    config_data = {}
    ctext = request.form.get("config_json", "")
    if ctext.strip():
        try:
            config_data = json.loads(ctext)
        except json.JSONDecodeError:
            return jsonify({"error": "Невалидная конфигурация (JSON)"}), 400
    try:
        model_det, identifier, tracker_path, det_imgsz, device, use_half = _load_live_pipeline(config_data)
    except FileNotFoundError as e:
        return jsonify({"error": f"Модель не найдена: {e}"}), 500
    except ImportError:
        return jsonify({"error": "ultralytics не установлен"}), 500

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return jsonify({"error": "Не удалось открыть видеофайл"}), 400
    src_fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    total_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Downsample to ~30fps like Automate mode (process every Nth source frame).
    frame_stride = max(round(src_fps / 30), 1)
    fps = src_fps / frame_stride
    total = total_src // frame_stride

    sid = hashlib.md5(f"live_{filename}_{time.time()}".encode()).hexdigest()[:12]
    out_dir = RESULTS_FOLDER / f"live_{sid}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tracker_path = effective_tracker(tracker_path, out_dir, fps)
    live_session = live_mod.LiveSession(cap, model_det, identifier, tracker_path,
                                        fps, total, out_dir, det_imgsz,
                                        lines=squad_lines(config_data), video_path=video_path,
                                        frame_stride=frame_stride, device=device, half=use_half)
    live_session.start()
    return jsonify({"status": "ok", "session": sid, "total": total})


@app.route("/live/state")
def live_state():
    if not live_session:
        return jsonify({"status": "idle"})
    return jsonify(live_session.state())


@app.route("/live/frame")
def live_frame():
    if not live_session or not live_session.frame_jpeg:
        return ("", 204)
    from flask import Response
    return Response(live_session.frame_jpeg, mimetype="image/jpeg")


@app.route("/live/pause", methods=["POST"])
def live_pause():
    if live_session:
        live_session.pause()
    return jsonify({"status": "ok"})


@app.route("/live/resume", methods=["POST"])
def live_resume():
    if live_session:
        live_session.resume()
    return jsonify({"status": "ok"})


@app.route("/live/stop", methods=["POST"])
def live_stop():
    if live_session:
        live_session.stop()
        return jsonify({"status": "ok", "job_id": live_session.job_id})
    return jsonify({"status": "ok"})


@app.route("/live/override", methods=["POST"])
def live_override():
    """Edit or remove an already-applied correction, live."""
    if not live_session:
        return jsonify({"error": "нет сессии"}), 400
    d = request.get_json(force=True) or {}
    try:
        pid = int(d.get("pid"))
    except (TypeError, ValueError):
        return jsonify({"error": "bad pid"}), 400
    live_session.set_override(pid, d.get("number"), d.get("team"), remove=bool(d.get("remove")))
    return jsonify({"status": "ok"})


@app.route("/live/cut", methods=["POST"])
def live_cut():
    """Split a track at the current frame: the tracker jumped to another player."""
    if not live_session:
        return jsonify({"error": "нет сессии"}), 400
    try:
        pid = int((request.get_json(force=True) or {}).get("pid"))
    except (TypeError, ValueError):
        return jsonify({"error": "bad pid"}), 400
    if not live_session.cut_track(pid):
        return jsonify({"error": "too_short"}), 400
    return jsonify({"status": "ok"})


@app.route("/live/pending", methods=["POST"])
def live_pending():
    """Skip a question about a track that left the frame unanswered."""
    if not live_session:
        return jsonify({"error": "нет сессии"}), 400
    try:
        pid = int((request.get_json(force=True) or {}).get("pid"))
    except (TypeError, ValueError):
        return jsonify({"error": "bad pid"}), 400
    live_session.dismiss_pending(pid)
    return jsonify({"status": "ok"})


@app.route("/live/ignore", methods=["POST"])
def live_ignore():
    """Leave a track out of the count entirely (the goalie, a linesman)."""
    if not live_session:
        return jsonify({"error": "нет сессии"}), 400
    d = request.get_json(force=True) or {}
    try:
        pid = int(d.get("pid"))
    except (TypeError, ValueError):
        return jsonify({"error": "bad pid"}), 400
    live_session.ignore_track(pid, on=d.get("on", True))
    return jsonify({"status": "ok"})


@app.route("/live/shift", methods=["POST"])
def live_shift():
    """Review/fix a player the model was confident about (never asked): remove a
    phantom, or reassign its tracks to another number/team."""
    if not live_session:
        return jsonify({"error": "нет сессии"}), 400
    d = request.get_json(force=True) or {}
    action = d.get("action")
    if action not in ("remove", "reassign"):
        return jsonify({"error": "bad action"}), 400
    live_session.edit_shift(d.get("team"), d.get("number"), action,
                            new_number=d.get("new_number"), new_team=d.get("new_team"))
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    app.run(debug=args.debug, host="0.0.0.0", port=args.port, threaded=True)
