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

## Team-classification and camera-pan verification

This second check used another real sample listed by the Roboflow Sports soccer setup:
`0bfacc_0.mp4`. To exercise cut handling without mixing teams or kits, the ignored 12-second
fixture concatenated the source's first six seconds with seconds 12–18 from the **same
match**, cropping the second section to 1280×720 and scaling it back to 1920×1080. The source
contains broadcast pans; the wide-to-tight transition at exactly 6.0 seconds is constructed
and must not be described as an organic broadcast cut.

Input fixture metadata measured with `ffprobe`:

```text
codec_name=h264
width=1920
height=1080
avg_frame_rate=25/1
duration=12.000000
nb_frames=300
```

Command run:

```bash
MPLCONFIGDIR=/private/tmp/football_tracker_verify_cache \
XDG_CACHE_HOME=/private/tmp/football_tracker_verify_cache \
PYTORCH_ENABLE_MPS_FALLBACK=1 \
/private/tmp/football_tracker_verify_20261118/bin/football-tracker \
  data/roboflow-pan-cut-12s.mp4 \
  --output outputs/roboflow-pan-cut-12s-teams.mp4 \
  --model data/football-player-detection.pt \
  --device mps \
  --image-size 960 \
  --confidence 0.25 \
  --team-sample-seconds 1.0
```

Observed CLI result:

```text
Wrote 300 frames at 25.000 FPS to outputs/roboflow-pan-cut-12s-teams.mp4 (50 tracks, 1 camera-cut resets). Team classification ready (goalkeeper/unknown=1, referee=5, team-a=20, team-b=24).
```

Those counts report each unique track's latest label; transient `unknown` labels during
track warm-up are visible in the video but are not a separate final-count bucket.

The calibration pass used 12 sampled frames. After warm-up it ran at approximately 2.4
samples/second, and the full tracking pass ran at approximately 6.1 frames/second on the
available Apple Silicon MPS device. These are observations from this fixture, not general
performance guarantees.

Output metadata measured with `ffprobe`:

```text
codec_name=mpeg4
width=1920
height=1080
r_frame_rate=25/1
avg_frame_rate=25/1
duration=12.000000
nb_frames=300
```

### Manual frame inspection

Contact sheets were inspected at frames 50, 60, 70, 80, 90, and 100 before the cut; frames
150–155 during reset recovery; and frames 175, 185, 195, 205, 215, and 225 after the cut.
Those images and the video remain ignored because generated media must not enter Git.

Observed tracking behaviour:

- IDs on the continuously visible people checked in both pan sequences remained stable; no
  definite within-shot ID switch was observed in those sampled frames.
- Frame 150 detected the constructed cut with a scene score of 0.31 and reset ByteTrack.
  Frame 151 still had no confirmed tracks. This one-to-two-frame annotation gap is an
  observed reset/reacquisition limitation.
- All reappearing people received new globally unique IDs in segment 2. This is intentional
  cut isolation, not identity continuity and not counted as a within-shot ID switch.
- The reported 50 unique tracks therefore does **not** mean 50 people were visible. Typical
  inspected frames contained 22–23 tracks, with different global IDs on either side of the
  reset.

Observed classification behaviour and errors:

- `team-a` consistently covered the white/black outfield shirts and `team-b` the red shirts
  in the inspected frames. No confirmed outfield team swap or per-track colour flip was
  observed in this visual sample.
- The two frames after track confirmation labelled outfield tracks `unknown`; most became
  team-labelled on frame 154 after three compatible votes, while one track was still
  `unknown` on frame 155. These are intended uncertainty abstentions, not forced guesses.
- Referees remained a separate yellow role. One detector-labelled goalkeeper was retained as
  `goalkeeper/unknown`; the implementation intentionally does not infer its team from the
  outfield-shirt clusters.
- No labelled ground truth was available, so this inspection cannot establish classification
  accuracy or prove that no switches occurred between sampled frames. It found no confirmed
  shirt-classification error; an evaluation set with player and role labels is still needed.
- Greedy label placement reduced collisions compared with always drawing above each box, but
  some labels still overlap in tight player groups.

## Current automated checks

After the team and camera-cut changes, 22 tests completed successfully and
`ruff check src tests` reported no issues. The added tests cover shirt-feature extraction,
repeatable two-cluster fitting, uncertainty rejection, role separation, temporal voting,
team colours, label placement, CLI options, and the wide-to-tight cut regression fixture.
