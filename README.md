# Player Tracking

Semi-automated ice hockey video analytics: per-player video clips and time on ice
from ordinary broadcast footage, with a human verifying the machine's work. No
wearable sensors.

## Overview

Player Tracking turns a period of hockey video into per-player clips and a
time-on-ice report. A YOLO detector and a BoT-SORT tracker follow every skater,
team and jersey number are read from the picture, and the operator verifies the
result — either live while the video plays, or afterwards in a review page.

It replaces GPS/LPS vests for the scouting and ice-time part of the job. It does
not measure distance or skating speed; see [Not implemented](#not-implemented).

## Key Features

- **Two working modes** — *live control*, where the operator corrects labels as
  the video plays, and *automate*, a batch pre-pass followed by review. Both end
  in the same review page and produce the same results
- **Video Scouting** — per-player clips at the source resolution and frame rate
- **Time on Ice** — per-player TOI and a shift timeline, derived from the tracks
  the operator confirmed
- **Team Classification** — by jersey brightness: hockey requires one light and
  one dark team, so a torso crop's median brightness is split against two online
  cluster centres that adapt as the game goes on. Steadier than matching hues,
  which drift with arena lighting and broadcast colour grading
- **Number Recognition** — YOLO digit detection on every frame, voted over the
  whole track, constrained to the configured roster and to one player per number;
  forward-line and D-pair context resolves the ambiguous reads
- **Track Integrity** — a number-change guard and a kinematic guard (a box
  jumping more than two of its own heights in one frame is the tracker leaping to
  another player) split contaminated tracks; a two-pass merger reassembles
  fragments of the same player
- **Review on the video itself** — the whole match with tracks drawn over it:
  reassign a track to another player, cut it where the tracker jumped, drop it.
  Numbers are picked from the roster, never typed
- **Accuracy Report** — every session measures how much of the automatic
  labelling survived review
- **Bilingual UI** — Russian / English, remembered per browser
- **Desktop app** — ships as a macOS DMG and a Windows installer; no Python on
  the user's machine

## Not implemented

- **Distance and skating speed.** Both need the rink calibrated to real-world
  metres, which needs rink-line detection wired into the pipeline. The Excel
  report therefore carries `Name`, `Number`, `Team`, `Time (s)` and nothing else.
  A `LinesDetection.pt` model exists from earlier work but is not used by any
  code path.
- **Cross-team swap splitting** is implemented but switched off by default
  (`PlayerIdentifier.TEAM_SPLIT`): the splits it produced cost more review time
  than the swaps they fixed.

## Processing Pipeline

```
Configuration → Detection & Tracking → Identification → Review → Results
```

1. **Configuration** — rosters, which team wears the light jersey, D-pairs
   (up to 4×2), forward lines (up to 4×3), goalie numbers
2. **Detection & Tracking** — YOLO on every frame, CoreML on the Apple Neural
   Engine where available; BoT-SORT with appearance ReID and camera-motion
   compensation, so a track survives a pan or a short occlusion
3. **Identification** — team from torso brightness; the jersey number read every
   frame and decided by weighted voting over the track rather than per frame
4. **Review** — reassign, cut or drop tracks on the video itself; unidentified
   tracks can be given a number from the roster
5. **Results** — merged per-player clips, TOI, shift timeline, Excel report,
   accuracy report

## Tech Stack

| Component | Technology |
|---|---|
| Backend | Flask 3.x (Python 3.10) |
| Desktop shell | pywebview window + PyInstaller bundle |
| Detection | YOLO — custom trained `Player_Detection.pt` |
| Number OCR | YOLO — `NumbersRec.pt` |
| Tracking | BoT-SORT with ReID and GMC (`trackers/tracker_stable.yaml`) |
| Team classification | Jersey brightness against two online EMA cluster centres |
| Inference acceleration | CoreML / Apple Neural Engine, MPS, CUDA, CPU fallback |
| Video writing | OpenCV H.264 (`avc1`), background writer thread |
| Frontend | Vanilla JS + CSS + Chart.js; `i18n.js` for RU/EN |

## Project Structure

```
├── webapp/
│   ├── app.py                  # Flask application: pipeline, review, results
│   ├── live.py                 # Live-control session (decode, draw, edit, save)
│   ├── launcher.py             # Desktop entry point: native window + --selftest
│   ├── requirements.txt
│   ├── packaging/              # PyInstaller spec, Inno Setup script, DMG assets
│   ├── tools/                  # Icon and DMG-background generators, JS tests
│   ├── test_*.py               # 75 unit tests (plain pytest, no fixtures)
│   ├── static/
│   │   ├── css/style.css
│   │   └── js/{app.js, i18n.js}
│   └── templates/
│       ├── base.html           # Layout: navbar, language toggle, footer
│       ├── index.html          # Landing page
│       ├── config.html         # Team configuration editor
│       ├── analysis.html       # Video analysis start page (live / automate)
│       ├── job.html            # Processing status and profiling
│       ├── review_live.html    # Review: tracks drawn over the whole video
│       ├── result.html         # One finished session
│       ├── results.html        # Session list
│       ├── dashboard.html      # Per-dataset statistics
│       ├── dataset.html        # Training-data staging
│       └── stats.html          # Aggregated statistics
├── models/
│   ├── Player_Detection.pt     # 21 MB — in the repository
│   ├── Player_Detection.mlpackage  # 21 MB — CoreML export, macOS
│   └── NumbersRec.pt           # 19 MB — in the repository
├── trackers/
│   ├── tracker_stable.yaml     # BoT-SORT + ReID (used by default)
│   ├── tracker_fast.yaml       # Looser thresholds
│   └── tracker.yaml            # Fallback
├── .github/workflows/build.yml # Builds the installers on a tag
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
| Apple Silicon (M1-M4) | `PlayerTracking-macOS-arm64.dmg` |
| Intel | `PlayerTracking-macOS-intel.dmg` |

1. Open the DMG and drag **Player Tracking** into **Applications**.
2. Open **Applications** and double-click the app. macOS says it cannot verify the
   developer — click **Done**.
3. Open **System Settings → Privacy & Security**, scroll to the bottom and click
   **Open Anyway**, then **Open**.

From then on the app starts with a normal double-click.

### Windows 10 / 11

1. Run `PlayerTracking-Setup.exe`. Windows SmartScreen may warn about an unknown publisher —
   click **More info → Run anyway**.
2. Click through the installer. No administrator password is needed: the app
   installs for the current user only.
3. Start **Player Tracking** from the Start menu or the desktop shortcut.

The installer adds the Microsoft WebView2 runtime if it is missing, which the app
uses to draw its window.

### No warnings at all

The warnings above come from the quarantine flag that browsers attach to
downloads. Copying the installer from a USB stick or over a local network skips
them entirely — useful when handing the app to a club.

### Where your data lives

Videos, results, and team configurations are stored outside the app, so updating
or reinstalling it never touches them:

- macOS — `~/Library/Application Support/PlayerTracking`
- Windows — `%LOCALAPPDATA%\PlayerTracking`

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

### 3. Models

The two models the app loads are tracked in this repository, so a clone has
everything it needs:

| File | Description | Size |
|---|---|---|
| `Player_Detection.pt` | Player detection | 21 MB |
| `NumbersRec.pt` | Jersey number recognition | 19 MB |
| `Player_Detection.mlpackage` | CoreML export of the detector, used on Apple Silicon | 21 MB |

### 4. Run

```bash
cd webapp
python app.py
```

App starts at **http://localhost:5050**.

`python launcher.py` instead opens the same app in a desktop window, the way the
installed build runs; that path additionally needs `pip install pywebview`, which
`requirements.txt` leaves out so a headless install stays free of GUI packages.

### 5. (Optional) re-export the CoreML model

The detector is loaded from `Player_Detection.mlpackage` whenever the app runs
on MPS, and that export is already in the repository. To rebuild it:

```bash
pip install coremltools>=9.0
python -c "from ultralytics import YOLO; YOLO('models/Player_Detection.pt').export(format='coreml', imgsz=640, nms=False)"
```

## Usage

1. **Create a configuration** — `/config`: team names, rosters, jersey colours,
   D-pairs, forward lines, goalie numbers. Saved for reuse
2. **Start the analysis** — `/analysis`: pick the video and the configuration.
   The checkbox chooses the mode: *live control* corrects labels while the video
   plays, unchecked runs the batch pre-pass first
3. **Review** — the whole video with the tracks drawn over it. Reassign a track
   to another player, cut it where the tracker jumped to someone else, or drop
   it. Numbers come from the configured roster
4. **Finalize** — merged per-player videos, the Excel report and the accuracy
   report are written out

## Performance

Measured on an **Apple M3** with the CoreML detector, on a 112-second 1080p clip
at 25 fps (2800 frames), reading the jersey number on every frame:

| Phase | Time |
|---|---|
| YOLO detection + tracking | 77.8 ms/frame |
| Identification (team + number) | 54.2 ms/frame |
| Data and clip writing | 0.1 ms/frame |
| **Throughput** | **≈ 7.5 frames/s — about 6.5 minutes for a 2-minute clip** |

Accuracy on that run: **83.9 %** of the automatic labels survived review — 43
clips, 26 correct, 4 corrected, 12 unidentified tracks assigned by hand, 1
rejected. Expect this to move with footage quality, camera work, and how clearly
the two teams' jerseys differ in brightness.

These figures are slower than earlier versions of this project by design: the
jersey number is now read on *every* frame instead of every third, and tracking
runs BoT-SORT with appearance ReID instead of IoU-only ByteTrack. Both cost
machine time and both cut the number of tracks a human has to fix afterwards,
which is the expensive part.

Every session records its own numbers: they are on the job page and in
`review_manifest.json` under `profile`.

## Configuration Format

Team configurations are JSON. Running from source they live in `webapp/configs/`;
in the installed app they sit in the user data directory listed above. The colours
below decide which side is the light team — the classifier works on brightness,
not on the colour names themselves.

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

**Medvedev M.D.** — Master's research project, 2025–2026
