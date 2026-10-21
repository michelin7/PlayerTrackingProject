# Player Scouting & Fitness (PS&F)

Semi-automated ice hockey video analytics system for individual player scouting and fitness assessment — powered by computer vision, no wearable sensors required.

## Overview

PS&F extracts per-player video clips and physical performance metrics (TOI, distance, average and maximum skating speed) from standard broadcast hockey footage. It replaces expensive GPS/LPS tracking hardware with a camera-based pipeline: YOLO detection, ByteTrack tracking, jersey color classification, number recognition, and a human-in-the-loop review step.

## Key Features

- **Video Scouting** — automatic per-player clip generation for individual on-ice review at original video quality and frame rate
- **Fitness Metrics** — distance covered, average and maximum skating speed (km/h) calculated after the human review step
- **No Sensors Needed** — works from any broadcast-quality video feed without GPS/LPS hardware
- **Team Classification** — HSV pixel-affinity scoring with dual jersey colors (primary + secondary) and adaptive ice-surface filtering
- **Number Recognition** — YOLO-based OCR with confusion-aware voting (e.g. #7 vs #47) and forward-line/D-pair context disambiguation
- **Track Integrity** — ByteTrack cross-team jump detection with virtual PID splitting; 3-pass TrackMerger for fragmented tracks
- **Perspective-Corrected Speed** — per-frame speed scaled by vertical position in frame (near/far side of rink); physically capped at 46.8 km/h; smoothed over a 0.5 s sliding window
- **Semi-Automated Review** — the system labels, the human verifies and corrects; auto-labeling accuracy is measured per session
- **Profiling Dashboard** — per-phase timing (YOLO track / identification / data+write) visible on the job status page
- **Bilingual UI** — Russian / English interface with localStorage language preference

## Processing Pipeline

```
Configuration → Detection & Tracking → Identification → Review → Results
```

1. **Configuration** — team rosters, jersey colors (primary + secondary), D-pairs (up to 4×2), forward lines (up to 4×3), goalie numbers
2. **Detection & Tracking** — YOLO model accelerated via CoreML on Apple Neural Engine + ByteTrack (IoU-only, no ReID); all frames processed for full-quality clips
3. **Identification** — team classification by jersey color affinity; YOLO number OCR every 3rd frame with cache; line-mate context for ambiguous numbers
4. **Review** — per-clip approve / reject / reassign; unidentified tracks can be manually assigned
5. **Results** — TOI, distance, avg/max speed (km/h), merged per-player video clips, Excel report, accuracy report

## Tech Stack

| Component | Technology |
|---|---|
| Backend | Flask 3.x (Python 3.10+) |
| Detection | YOLOv8 — custom trained `Player_Detection.pt` |
| Line Calibration | YOLOv8 — `LinesDetection.pt` (rink line segmentation) |
| Number OCR | YOLOv8 — `NumbersRec.pt` |
| Tracking | ByteTrack (`tracker_fast.yaml`, IoU-only, no ReID) |
| Inference acceleration | CoreML / Apple Neural Engine (primary), MPS, CUDA |
| Speed estimation | Perspective-corrected pixels-to-meters, 0.5 s sliding window, hard cap 13 m/s |
| Video writing | Async background thread, original FPS, H.264 |
| Frontend | Vanilla JS + CSS + Chart.js; `i18n.js` for RU/EN |

## Project Structure

```
├── webapp/
│   ├── app.py                  # Main Flask application (~2200 lines)
│   ├── requirements.txt        # Python dependencies
│   ├── static/
│   │   ├── css/style.css       # UI styles (light theme)
│   │   └── js/
│   │       ├── app.js          # Common helpers
│   │       └── i18n.js         # RU/EN translation strings + toggle logic
│   └── templates/
│       ├── base.html           # Base layout: navbar, language toggle, footer
│       ├── index.html          # Landing page (features, pipeline, model status)
│       ├── config.html         # Team configuration editor (rosters, colors, lines)
│       ├── upload.html         # Video upload + config selection with roster preview
│       ├── job.html            # Processing status, progress bar, profiling metrics
│       ├── review.html         # Per-clip review interface
│       ├── results_list.html   # Processed jobs list
│       ├── dashboard.html      # Per-dataset statistics dashboard
│       └── stats.html          # Aggregated stats across sessions
├── models/                     # YOLO model weights (not in repo — see Setup)
│   ├── Player_Detection.pt     # ~21 MB
│   ├── LinesDetection.pt       # ~19 MB
│   └── NumbersRec.pt           # ~21 MB
├── trackers/
│   ├── tracker_fast.yaml       # ByteTrack, IoU-only (used by default)
│   ├── tracker.yaml            # ByteTrack with ReID
│   └── tracker_stable.yaml     # Conservative thresholds variant
├── requirements.txt
└── README.md
```

## Install (end users)

Download the installer for your system from the **Releases** page, then follow the
steps below. The app is not signed with a paid developer certificate, so each
system asks for one confirmation the first time — once per computer, never again.

### macOS

Two builds are published; pick the one matching your Mac. Apple menu  → **About
This Mac**: a *Chip* line (M1-M4) means Apple Silicon, a *Processor* line with
Intel means the Intel build.

| Mac | File |
| --- | --- |
| Apple Silicon (M1-M4) | `PSF-macOS-arm64.dmg` |
| Intel | `PSF-macOS-intel.dmg` |

1. Open the DMG and drag **Player Scouting & Fitness** into **Applications**.
2. Open **Applications** and double-click the app. macOS says it cannot verify the
   developer — click **Done**.
3. Open **System Settings → Privacy & Security**, scroll to the bottom and click
   **Open Anyway**, then **Open**.

From then on the app starts with a normal double-click.

### Windows 10 / 11

1. Run `PSF-Setup.exe`. Windows SmartScreen may warn about an unknown publisher —
   click **More info → Run anyway**.
2. Click through the installer. No administrator password is needed: the app
   installs for the current user only.
3. Start **Player Scouting & Fitness** from the Start menu or the desktop shortcut.

The installer adds the Microsoft WebView2 runtime if it is missing, which the app
uses to draw its window.

### No warnings at all

The warnings above come from the quarantine flag that browsers attach to
downloads. Copying the installer from a USB stick or over a local network skips
them entirely — useful when handing the app to a club.

### Where your data lives

Videos, results, and team configurations are stored outside the app, so updating
or reinstalling it never touches them:

- macOS — `~/Library/Application Support/PSF`
- Windows — `%LOCALAPPDATA%\PSF`

## Setup (from source)

### Prerequisites

- Python 3.10+
- macOS with Apple Silicon (for CoreML/Neural Engine acceleration), or NVIDIA GPU (CUDA), or CPU fallback

### 1. Clone the repository

```bash
git clone https://github.com/michelin7/PlayerTrackingProject.git
cd PlayerTrackingProject
```

### 2. Install dependencies

```bash
pip install -r webapp/requirements.txt
```

### 3. Place models

Download the three YOLO model files and put them in the `models/` directory:

| File | Description | Size |
|---|---|---|
| `Player_Detection.pt` | Player detection model | ~21 MB |
| `LinesDetection.pt` | Rink line segmentation for calibration | ~19 MB |
| `NumbersRec.pt` | Jersey number recognition (OCR) | ~21 MB |

> Models are not included in the repository due to file size. Contact the author or train custom models using [Ultralytics YOLOv8](https://docs.ultralytics.com/).

### 4. Run

```bash
cd webapp
python app.py
```

App starts at **http://localhost:5050**.

### 5. (Optional) CoreML acceleration on Apple Silicon

The app automatically detects and uses a pre-exported CoreML model (`Player_Detection.mlpackage`) if present. To export it:

```bash
pip install coremltools>=9.0
python -c "from ultralytics import YOLO; YOLO('models/Player_Detection.pt').export(format='coreml', imgsz=640, nms=False)"
```

This gives ~2× inference speedup on the Neural Engine (~24 ms/frame vs ~53 ms on MPS).

## Usage

1. **Create a configuration** — `/config`: set team names, rosters, jersey colors, D-pairs, forward lines, goalie numbers; save for reuse
2. **Upload video** — `/upload`: select a saved config, upload a game period video (any length)
3. **Monitor processing** — the job page shows progress, FPS, and per-phase timing breakdown
4. **Review** — approve, reject, or reassign each automatically labeled clip; manually assign unidentified tracks
5. **Finalize** — generates merged per-player videos, Excel stats, and an accuracy report

## Performance

| Metric | Value |
|---|---|
| Processing speed — Apple M3, CoreML | ~30 FPS |
| Processing speed — Apple M3, MPS (PyTorch) | ~14 FPS |
| YOLO track time — CoreML | ~24 ms/frame |
| YOLO track time — MPS | ~53 ms/frame |
| Auto-labeling accuracy (typical, before review) | 70–85% |
| Max player speed cap | 46.8 km/h (13 m/s) |
| Speed smoothing window | 0.5 s |
| Perspective correction factor (PERSPECTIVE_K) | 0.6 |

## Configuration Format

Team configurations are stored as JSON in `webapp/configs/`. Example structure:

```json
{
  "name": "Dynamo vs Lokomotiv",
  "teams_colors_info": {
    "Home": ["blue", "white"],
    "Away": ["red", "white"]
  },
  "home_team": { "71": "Вей", "97": "Панарин" },
  "away_team": { "10": "Красковский", "55": "Воронков" },
  "home_defense_pairs": [["71", "44"], ["8", "55"]],
  "home_forward_lines": [["97", "23", "18"], ["10", "7", "91"]],
  "away_defense_pairs": [["10", "55"]],
  "away_forward_lines": [["21", "47", "9"]],
  "goalies_home": [30],
  "goalies_away": [40]
}
```

## Author

**Medvedev M.D.** — Research project, 2025
