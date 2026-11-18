# Football Video Tracker

An understandable, offline-first football video analysis project. The current CLI reads a
local match clip, detects visible players and officials, assigns ByteTrack IDs, draws short
image-space movement trails, and writes an annotated MP4 with one output frame per decoded
input frame.

This is the first milestone of a one-week demo build started on 18 November 2026. Track IDs
such as `T7` are temporary tracker IDs, **not real player identities**.

## Current status

| Capability | Status |
| --- | --- |
| Local video CLI | Working |
| Pretrained person or football-object detection | Working |
| ByteTrack IDs and movement trails | Working |
| Camera-cut tracker reset | Working, histogram-based baseline |
| MP4 export at source resolution and nominal FPS | Working for constant-frame-rate input |
| Team shirt classification and team colours | Next milestone |
| Goalkeeper/referee treatment | Detected as distinct classes with the football model; team logic pending |
| Pitch calibration and tactical view | Not implemented yet |
| Ball, possession, speed, distance, and spacing | Not available yet |
| Dashboard | Deliberately deferred until the video pipeline is reliable |

The colours in today's output are deterministic **track colours**, not team colours.

## Why this implementation

- **Ultralytics YOLO** gives a small pretrained fallback (`yolo11n.pt`) and also loads the
  football-specific YOLOv8 checkpoint used by Roboflow Sports.
- **Supervision ByteTrack 0.25.1** follows the public Roboflow Sports soccer example and
  provides a compact, model-independent tracking baseline.
- **OpenCV** keeps frame decoding, annotations, and encoding explicit enough to learn from.
- **MPS** is selected automatically on Apple Silicon, with `--device mps` available when you
  want to require it.
- A camera cut clears ByteTrack motion state and starts a new segment. Display IDs remain
  globally unique, so an ID is never silently reused for a different person after the cut.
- Output is written to a temporary file and atomically moved into place only after success.

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
and referees.

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

Use `football-tracker --help` for confidence, tracker buffer, camera-cut, class, device, and
codec options. If MPS is unavailable, use `--device cpu`; it will be slower.

## Timing and media behaviour

The processor decodes every frame, writes exactly one annotated frame for it, and configures
the output with the source-reported FPS and resolution. This preserves timing for normal
constant-frame-rate match clips. OpenCV output is video-only: source audio is not copied,
and variable-frame-rate timestamps are normalized to the reported average FPS. Audio remux
and true per-frame timestamps are a later export improvement.

The default `mp4v` codec was verified as a playable MP4 on this machine. If a local OpenCV
build cannot create it, try `--codec avc1`.

## Verification

Run all checks:

```bash
source .venv/bin/activate
pytest
ruff check src tests
```

The milestone was also run on a real Roboflow Sports football sample. The Git-ignored output
contained 75/75 frames at 1920×1080, 25 FPS, and 3.0 seconds, matching the trimmed input.
See [docs/verification.md](docs/verification.md) for the exact command, measured result, and
honest visual limitations.

## Project layout

```text
src/football_tracker/
  cli.py          argument parsing and early validation
  detection.py    Ultralytics adapter and football-class inference
  scene.py        camera-cut score
  tracking.py     ByteTrack lifecycle and global display IDs
  annotation.py   boxes, foot markers, labels, and image-space trails
  video.py        explicit reader and atomic source-FPS writer
  pipeline.py     end-to-end orchestration
tests/            unit tests plus a generated-video MP4 integration test
```

## One-week roadmap

1. **18 Nov — complete:** detection, tracking, cut resets, trails, timed MP4 export.
2. **19 Nov:** shirt-appearance team classifier; preserve separate goalkeeper/referee roles.
3. **20 Nov:** pitch landmark model and per-shot homography quality checks.
4. **21 Nov:** synchronized top-down pitch using only visible, calibrated foot positions.
5. **22 Nov:** ball detector/tracker and uncertainty-aware estimated possession.
6. **23 Nov:** validation clips, ID/calibration quality gates, then speed/distance/spacing only
   where metric calibration supports them.
7. **24 Nov:** demo polish and a small dashboard after the annotated video remains reliable.

No tactical position will be fabricated outside the camera view. Statistics will be marked
unavailable when detection, tracking, or calibration quality is insufficient.
