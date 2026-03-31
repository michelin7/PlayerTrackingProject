# Player Scouting & Fitness (PS&F)

Semi-automated ice hockey video analytics system for individual player scouting and fitness assessment — powered by computer vision, no wearable sensors required.

## Overview

PS&F extracts per-player video clips and physical performance metrics (TOI, distance, average/top skating speed) from standard broadcast hockey footage. It replaces expensive GPS/LPS tracking hardware with a camera-based pipeline: YOLO detection, ByteTrack tracking, jersey color classification, number recognition, and a human-in-the-loop review step.

## Key Features

- **Video Scouting** — automatic per-player clip generation for individual on-ice review
- **Fitness Metrics** — distance covered, average and maximum skating speed (km/h) with perspective correction
- **No Sensors Needed** — works from any broadcast-quality video feed
- **Team Classification** — HSV pixel-affinity scoring with dual jersey colors and adaptive ice filtering
- **Number Recognition** — YOLO-based OCR with confusion-aware voting (e.g. 7 vs 47) and line-context disambiguation
- **Semi-Automated Review** — the system labels, the human reviews and corrects; accuracy is measured
- **Bilingual UI** — Russian / English interface

## Processing Pipeline

```
Configuration → Detection & Tracking → Identification → Review → Results
```

1. **Configuration** — team rosters, jersey colors (primary + secondary), D-pairs, forward lines, goalie numbers
2. **Detection & Tracking** — YOLO model (CoreML-accelerated on Apple Silicon) + ByteTrack (IoU association)
3. **Identification** — team classification by jersey color affinity + number recognition + line-mate context
4. **Review** — verify auto-labeling, correct misidentifications, assign unidentified tracks
5. **Results** — TOI, distance, avg/max speed, individual player video clips, Excel report

## Tech Stack

| Component | Technology |
|---|---|
| Backend | Flask (Python) |
| Detection | YOLOv8 (ultralytics) |
| Tracking | ByteTrack (IoU-only, no ReID) |
| Inference | CoreML (Apple Neural Engine) / MPS / CUDA |
| Calibration | YOLO line segmentation → pixels-to-meters |
| Speed Estimation | Perspective-corrected, sliding-window smoothed, capped at 46.8 km/h |
| Frontend | Vanilla JS, CSS, Chart.js |

## Project Structure

```
├── webapp/
│   ├── app.py                  # Main Flask application (~2200 lines)
│   ├── requirements.txt        # Python dependencies
│   ├── static/
│   │   ├── css/style.css       # UI styles
│   │   └── js/
│   │       ├── app.js          # Common JS
│   │       └── i18n.js         # RU/EN translations
│   └── templates/
│       ├── base.html           # Base layout with nav
│       ├── index.html          # Landing page
│       ├── config.html         # Team configuration editor
│       ├── upload.html         # Video upload + config selection
│       ├── job.html            # Processing status + profiling
│       ├── review.html         # Review interface
│       ├── results_list.html   # Processed jobs list
│       ├── dashboard.html      # Statistics dashboard
│       └── stats.html          # Aggregated stats
├── models/                     # YOLO models (not in repo — see Setup)
│   ├── Player_Detection.pt
│   ├── LinesDetection.pt
│   └── NumbersRec.pt
├── trackers/                   # ByteTrack configuration
│   ├── tracker.yaml
│   ├── tracker_fast.yaml
│   └── tracker_stable.yaml
├── requirements.txt
└── README.md
```

## Setup

### Prerequisites

- Python 3.10+
- macOS with Apple Silicon (for CoreML acceleration) or NVIDIA GPU (CUDA)

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/player-scouting-fitness.git
cd player-scouting-fitness
```

### 2. Install dependencies

```bash
pip install -r webapp/requirements.txt
```

### 3. Download models

Place the following YOLO model files into the `models/` directory:

| File | Description | Size |
|---|---|---|
| `Player_Detection.pt` | Player detection model | ~21 MB |
| `LinesDetection.pt` | Rink line segmentation model | ~19 MB |
| `NumbersRec.pt` | Jersey number recognition model | ~21 MB |

> Models are not included in the repository due to size. Contact the author or train your own using [Ultralytics YOLOv8](https://docs.ultralytics.com/).

### 4. Run the application

```bash
cd webapp
python app.py
```

The app starts at **http://localhost:5050**.

### 5. (Optional) CoreML acceleration

On Apple Silicon, the app automatically exports `Player_Detection.pt` to CoreML format (`.mlpackage`) for ~2x inference speedup via the Neural Engine. This happens on first run if `coremltools` is installed:

```bash
pip install coremltools>=9.0
```

## Usage

1. **Create a configuration** — go to `/config`, set team rosters, jersey colors, D-pairs, and forward lines
2. **Upload video** — go to `/upload`, select a config and upload a game period video
3. **Wait for processing** — the job page shows progress and profiling data
4. **Review results** — verify auto-labeled player clips, correct mistakes, assign unidentified tracks
5. **Finalize** — generates merged per-player videos, Excel stats (TOI, distance, avg/max speed), and accuracy report

## Speed & Accuracy

| Metric | Value |
|---|---|
| Processing speed (Apple M3, CoreML) | ~30 FPS |
| Processing speed (Apple M3, MPS) | ~14 FPS |
| Auto-labeling accuracy (typical) | 70–85% (before review) |
| Max realistic player speed cap | 46.8 km/h (13 m/s) |
| Speed smoothing window | 0.5 sec |

## Author

**Medvedev M.D.** — Research project, 2025
