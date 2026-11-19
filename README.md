# Football Video Tracker

An understandable, offline-first football video analysis project. The current CLI reads a
local match clip, detects visible players and officials, assigns ByteTrack IDs, classifies
outfield shirt appearance into two anonymous teams, calibrates wide broadcast views from
pitch landmarks, and writes an annotated MP4 with a synchronized tactical pitch underneath.
With an optional roster, it also makes conservative, explicitly tentative player-name
suggestions from repeated jersey-number reads.
Ball detection and conservative short-term association are also available as an opt-in mode;
possession is deliberately not inferred yet.

This repository contains the first five milestones of a one-week demo build started on
18 November 2026. Track IDs such as `T7` are temporary tracker IDs, **not real player
identities**.

## Current status

| Capability | Status |
| --- | --- |
| Local video CLI | Working |
| Pretrained person or football-object detection | Working |
| ByteTrack IDs and movement trails | Working |
| Camera-cut tracker reset | Working, histogram-based baseline |
| MP4 export at source resolution and nominal FPS | Working for constant-frame-rate input |
| Team shirt classification and team colours | Working baseline with explicit uncertainty |
| Goalkeeper/referee treatment | Separate role colours; goalkeepers are not forced into an outfield-shirt cluster |
| Pitch calibration and tactical view | Working wide-shot baseline with per-frame trust gates |
| Jersey number / roster suggestions | Optional EasyOCR baseline; multi-frame consensus, roster-backed, and deliberately abstains on distant footage |
| Ball detection/tracking | Optional baseline; current-frame markers only, with uncertainty hiding |
| Possession, speed, distance, and spacing | Not available yet |
| Dashboard | Deliberately deferred until the video pipeline is reliable |

`team-a` and `team-b` are anonymous appearance clusters, not club names. Referees,
goalkeepers, and uncertain tracks receive separate labels and colours. Track IDs such as
`T7` remain temporary tracking IDs, never real player identities.

## Why this implementation

- **Ultralytics YOLO** gives a small pretrained fallback (`yolo11n.pt`) and also loads the
  football-specific YOLOv8 checkpoint used by Roboflow Sports.
- **Supervision ByteTrack 0.25.1** follows the public Roboflow Sports soccer example and
  provides a compact, model-independent tracking baseline.
- **OpenCV** keeps frame decoding, annotations, and encoding explicit enough to learn from.
- A small **NumPy two-cluster classifier** learns robust Lab-colour statistics from upper-
  torso crops. It avoids another heavyweight model in the first demo, abstains when the
  clusters or a crop are ambiguous, and smooths decisions per track.
- **MPS** is selected automatically on Apple Silicon, with `--device mps` available when you
  want to require it.
- A camera cut clears ByteTrack motion state and starts a new segment. Display IDs remain
  globally unique, so an ID is never silently reused for a different person after the cut.
- A second pretrained **pitch-landmark YOLO model** locates 32 indexed markings. OpenCV
  RANSAC fits a fresh image-to-pitch homography on every frame, so a pan updates rather than
  reusing stale geometry.
- Calibration is accepted only when landmark confidence, inlier agreement, image/pitch
  coverage, and pixel reprojection error pass. Failed frames show an empty tactical pitch
  with the reason; a cut is always invalid for that frame.
- Output is written to a temporary file and atomically moved into place only after success.
- Labels use a greedy nearby-placement pass to reduce collisions. Dense groups can still
  overlap; this is a readability baseline, not a full label-layout solver.
- **EasyOCR** is an optional, replaceable scene-text baseline. It only sees sharp central-
  torso crops from sufficiently large outfield-player tracks. A name is suggested only after
  at least three accepted reads agree and the exact `(team-a|team-b, number)` roster entry
  exists. Soccer-specific OCR is a future upgrade, not hidden behind this baseline.
- The Roboflow football checkpoint already contains a `ball` class, so `--track-ball` reuses
  the existing YOLO inference pass rather than loading a second model. A marker is shown only
  after consecutive, motion-consistent detections; misses and ambiguous candidates are hidden
  instead of extrapolated. Ball association state is cleared at every camera cut.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for credits and licence notes. The
design is informed by the
[Roboflow Sports soccer example](https://github.com/roboflow/sports/tree/main/examples/soccer),
but this smaller implementation is written for this project.

## MacBook Air M4 setup

Python 3.12 is recommended. The current Homebrew `python3` may be 3.14, which this project's
PyTorch stack does not target.

```bash
cd /path/to/football_tracker

# Install this once if python3.12 is not already available.
brew install python@3.12

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .

mkdir -p inputs outputs models
```

The first Ultralytics run may download model weights. Models, input videos, output videos,
caches, and local environments are ignored by Git.

Install the OCR extra only if you plan to pass `--roster`:

```bash
python -m pip install -e '.[ocr]'
```

The first roster run downloads EasyOCR weights into `models/easyocr/`. YOLO remains on MPS;
the EasyOCR 1.7 baseline runs on CPU on Apple Silicon because its MPS path was not stable in
verification.

### Quick start with the small generic model

Put a clip at `inputs/match.mp4`, then run:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 football-tracker \
  inputs/match.mp4 \
  --output outputs/match-tracked.mp4 \
  --device mps \
  --image-size 960
```

The generic model tracks COCO's `person` class. It cannot distinguish players, goalkeepers,
and referees, so every detected person is treated as an outfield player. Use
`--no-team-classification` with this fallback if officials or sideline staff would pollute
the two shirt clusters.

### Better football classes with the Roboflow Sports checkpoint

Roboflow's upstream `examples/soccer/setup.sh` publishes the checkpoint download. This is
the same URL, saved outside Git:

```bash
curl -fL \
  'https://drive.usercontent.google.com/download?id=17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q&export=download&confirm=t' \
  -o models/football-player-detection.pt

PYTORCH_ENABLE_MPS_FALLBACK=1 football-tracker \
  inputs/match.mp4 \
  --output outputs/match-football-tracked.mp4 \
  --model models/football-player-detection.pt \
  --device mps \
  --image-size 960
```

For that checkpoint, the CLI automatically selects `goalkeeper`, `player`, and `referee`
and excludes `ball`. Use `--classes 1,2,3` only if automatic class-name inference does not
match a different custom model.

### Optionally track the ball

The same football checkpoint reports classes `ball`, `goalkeeper`, `player`, and `referee`.
Enable its ball class without changing the default replay behavior:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 football-tracker \
  inputs/match.mp4 \
  --output outputs/match-ball-replay.mp4 \
  --model models/football-player-detection.pt \
  --pitch-model models/football-pitch-detection.pt \
  --track-ball \
  --ball-confidence 0.35 \
  --device mps \
  --image-size 960
```

The default requires two consecutive reads before drawing the ball. A lost or ambiguous ball
is hidden immediately on both views; the retained gap is association memory only and never
draws a predicted position. The tactical dot is additionally gated by trusted calibration.
Because the homography models the ground plane, an airborne ball's tactical location is only
an approximation. `--ball-minimum-hits` and `--ball-maximum-gap` expose the main association
controls. Possession estimation is intentionally deferred.

### Add the synchronized tactical pitch

Roboflow's setup script also publishes the pitch-landmark checkpoint. Save it outside Git:

```bash
curl -fL \
  'https://drive.usercontent.google.com/download?id=1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf&export=download&confirm=t' \
  -o models/football-pitch-detection.pt

PYTORCH_ENABLE_MPS_FALLBACK=1 football-tracker \
  inputs/match.mp4 \
  --output outputs/match-replay.mp4 \
  --model models/football-player-detection.pt \
  --pitch-model models/football-pitch-detection.pt \
  --device mps \
  --image-size 960
```

`--pitch-model` enables the lower panel. The pitch model keeps its published 640-pixel
inference size by default; raising it to 1280 reduced usable landmarks on the verified sample.
Use `--pitch-panel-height` to override the automatic height, and use the confidence,
minimum-keypoint, and maximum-error options only when validating a different camera domain.

### Team-label behaviour

Team classification is enabled by default. Before tracking, the CLI samples one frame per
second, extracts colour features only from detections whose model class is `player`, and fits
two shirt-appearance clusters for the clip. During tracking:

- outfield players become `team-a`, `team-b`, or `unknown` after temporal smoothing;
- referees retain the detector role and use a dedicated yellow annotation;
- goalkeepers use `goalkeeper/unknown` and a dedicated magenta annotation, because their
  different shirt should not be guessed into an outfield cluster;
- weak calibration or ambiguous crops stay `unknown` instead of receiving a forced team;
- camera cuts reset tracking and per-track shirt votes, so new tracks briefly return to
  `unknown` while evidence accumulates.

The cluster letters are stable within one run but have no semantic home/away meaning. Tune
sampling and abstention with `--team-sample-seconds` and `--team-confidence`, or disable the
calibration pass with `--no-team-classification`.

### Optional roster and tentative name suggestions

First run the clip without a roster and inspect which real club the shirt classifier called
`team-a` and `team-b`. Then create a CSV with that **explicit anonymous-team mapping**. The
same shirt number can safely occur once for each opposing team:

```csv
team,number,name
team-a,10,Ada Example
team-a,14,Alex Example
team-b,10,Bea Example
team-b,7,Ben Example
```

Run the same replay command with the roster:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 football-tracker \
  inputs/match.mp4 \
  --output outputs/match-roster-replay.mp4 \
  --model models/football-player-detection.pt \
  --pitch-model models/football-pitch-detection.pt \
  --roster inputs/roster.csv \
  --device mps \
  --image-size 960
```

Without `--roster`, the existing replay path is unchanged and EasyOCR is not imported or
loaded. With a roster, an unresolved outfield player is labelled only with the temporary ID,
such as `T7`. A consensus match is shown as a suggestion, such as `T7 Alex Example?`, on both
the footage and tactical pitch. The question mark is intentional: this is not a confirmed
identity. Referees and goalkeepers retain their separate role labels and colours.

The default gate samples a track every five frames, rejects detected players shorter than
120 pixels or blurred torso crops, and requires three compatible accepted OCR reads with at
least 75% support. An unknown team, conflicting reads, a number absent from that team's
roster, or weak image evidence produces no name. A camera cut clears every track's number
evidence, so post-cut suggestions must be earned again. `--jersey-sample-interval`,
`--jersey-minimum-votes`, `--jersey-minimum-confidence`, and
`--jersey-minimum-player-height` expose the main validation controls.

General-purpose OCR is not expected to recover tiny broadcast digits reliably. For useful
evaluation footage, record or obtain 1080p/4K medium or close broadcast shots where at least
two players are back-facing for several sharp frames and each torso/number occupies tens of
pixels. Keep the actual match roster alongside the clip. A future SoccerNet/PARSeq-style
tracklet model can replace `JerseyNumberOCR` without changing the replay pipeline.

Use `football-tracker --help` for confidence, tracker buffer, camera-cut, class, device, and
codec options. If MPS is unavailable, use `--device cpu`; it will be slower.

## Timing and media behaviour

The processor decodes every frame and writes exactly one composite frame at the source-
reported FPS. Without a pitch model, source resolution is retained. With a pitch model, the
source-width tactical panel is appended below, so output height increases while frame count,
FPS, and constant-frame-rate duration stay unchanged. OpenCV output is video-only: source
audio is not copied, and variable-frame-rate timestamps are normalized to the reported
average FPS. Audio remux and true per-frame timestamps are a later export improvement.

The default `mp4v` codec was verified as a playable MP4 on this machine. If a local OpenCV
build cannot create it, try `--codec avc1`.

## Verification

Run all checks:

```bash
source .venv/bin/activate
pytest
ruff check src tests
```

The pipeline was run on real Roboflow Sports football footage. The original 3-second check
and pitch-panel checks on a six-second pan and a 12-second constructed same-match camera cut
preserved input frame count, FPS, and duration. See
[docs/verification.md](docs/verification.md) for the exact commands, measured results,
observed ID behaviour, and visual limitations.

## Project layout

```text
src/football_tracker/
  cli.py          argument parsing and early validation
  detection.py    Ultralytics adapter and football-class inference
  ball.py         confidence filtering and conservative ball association
  classification.py  shirt features, uncertainty-aware clustering, and track smoothing
  identity.py     roster validation, jersey crops, replaceable OCR, and track consensus
  pitch.py        pitch geometry, landmark adapter, RANSAC fit, and quality gates
  tactical.py     top-down pitch, role markers, and vertical replay composition
  scene.py        camera-cut score
  tracking.py     ByteTrack lifecycle and global display IDs
  annotation.py   boxes, foot markers, labels, and image-space trails
  video.py        explicit reader and atomic source-FPS writer
  pipeline.py     end-to-end orchestration
tests/            unit tests plus a generated-video MP4 integration test
```

## One-week roadmap

1. **18 Nov — complete:** detection, tracking, cut resets, trails, timed MP4 export.
2. **18 Nov — complete:** uncertainty-aware shirt classification and separate role handling.
3. **19 Nov — complete:** pitch landmarks, per-frame homography gates, and cut invalidation.
4. **19 Nov — complete:** synchronized pitch using only visible, calibrated foot positions.
5. **19 Nov — baseline complete:** optional jersey OCR and tentative roster suggestions;
   readable close-shot validation footage is still required.
6. **19 Nov — baseline complete:** optional ball detector/tracker with current-frame-only
   replay markers; possession remains deferred.
7. **23 Nov:** validation clips, ID/calibration quality gates, then speed/distance/spacing only
   where metric calibration supports them.
8. **24 Nov:** demo polish and a small dashboard after the annotated video remains reliable.

No tactical position will be fabricated outside the camera view. Statistics will be marked
unavailable when detection, tracking, or calibration quality is insufficient.
