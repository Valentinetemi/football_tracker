# Milestone verification — 18 November 2026

This report records what was actually run. The sample, checkpoint, inspection images, and
generated video are Git-ignored and are not redistributed by this repository.

## Source

- Upstream project: [Roboflow Sports soccer example](https://github.com/roboflow/sports/tree/main/examples/soccer)
- Upstream sample: `2e57b9_0.mp4`, downloaded from the ID in Roboflow's `setup.sh`
- Verification clip: first 3 seconds of that sample
- Football detector: `football-player-detection.pt`, also from upstream `setup.sh`

Input clip metadata measured with `ffprobe`:

```text
codec_name=h264
width=1920
height=1080
avg_frame_rate=25/1
duration=3.000000
nb_frames=75
```

## Command run

The managed agent sandbox intermittently timed out while reading large dependency files
from the Desktop virtual environment, so this verification-only environment was installed
under `/private/tmp`. A normal local Mac run should use the `.venv` commands in the README.

```bash
MPLCONFIGDIR=/private/tmp/football_tracker_verify_cache \
XDG_CACHE_HOME=/private/tmp/football_tracker_verify_cache \
PYTORCH_ENABLE_MPS_FALLBACK=1 \
/private/tmp/football_tracker_verify_20261118/bin/football-tracker \
  data/roboflow-sample-3s.mp4 \
  --output outputs/roboflow-sample-3s-tracked.mp4 \
  --model data/football-player-detection.pt \
  --device mps \
  --image-size 960 \
  --confidence 0.25
```

Observed CLI result:

```text
Wrote 75 frames at 25.000 FPS to outputs/roboflow-sample-3s-tracked.mp4 (24 tracks, 0 camera-cut resets).
```

After the first-frame warm-up, progress was approximately 6.5 frames/second on the available
Apple Silicon MPS device. That is an observation from this three-second clip, not a general
performance guarantee.

Output metadata measured with `ffprobe`:

```text
codec_name=mpeg4
width=1920
height=1080
avg_frame_rate=25/1
duration=3.000000
nb_frames=75
```

## Visual inspection

Frames inspected around 1.5 and 2.5 seconds showed:

- bounding boxes and foot ellipses aligned with visible football people;
- `player` and `referee` class labels with detector confidence;
- the same IDs on several continuously visible people across the two inspected frames,
  including T2, T3, T4, and T6;
- longer image-space trails in the later frame;
- 24 visible tracks and no camera-cut reset in this short shot.

Observed limitations:

- labels overlap in the dense midfield area;
- per-track colours are not team assignments;
- a three-second, single-shot clip does not establish long-match ID stability;
- no audio is present in the generated MP4;
- this run does not validate cuts, occlusion recovery, team classification, calibration,
  tactical positions, possession, or metric statistics.

## Automated checks

The test suite includes CLI/config validation, football-class inference, scene-cut decisions,
global ID isolation after resets, and a generated-video integration check. The integration
check confirms one output frame per input frame at the same resolution and FPS.
