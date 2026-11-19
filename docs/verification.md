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

At the end of the team and camera-cut milestone, 22 tests completed successfully and
`ruff check src tests` reported no issues. The added tests cover shirt-feature extraction,
repeatable two-cluster fitting, uncertainty rejection, role separation, temporal voting,
team colours, label placement, CLI options, and the wide-to-tight cut regression fixture.

## Pitch calibration and tactical replay — 19 November 2026

This milestone used Roboflow Sports' published `football-pitch-detection.pt` checkpoint and
the same match footage as the team verification. The model supplies 32 indexed pitch
landmarks. The project fits pitch-to-image geometry with OpenCV RANSAC, inverts it for player
foot projection, and rejects the frame unless landmark count, spatial coverage, inlier ratio,
and pixel reprojection error pass. No previous homography is carried across a rejected frame.

Two Git-ignored fixtures were checked:

- `roboflow-pan-6s.mp4`: the first six seconds of `0bfacc_0.mp4`, containing a continuous
  wide broadcast pan;
- `roboflow-pan-cut-12s.mp4`: the previously documented same-match fixture with a constructed
  wide-to-tight cut at frame 150 / 6.0 seconds.

The managed workspace's Desktop virtual environment again stalled on large dependency and
model files. Verification therefore used a temporary Python 3.12 environment and temporary
copies of the ignored models/inputs. A normal Mac run should use the `.venv` and `models/`
paths documented in the README.

### Wide-pan command and result

```bash
MPLCONFIGDIR=/private/tmp/football_tracker_pitch_cache \
XDG_CACHE_HOME=/private/tmp/football_tracker_pitch_cache \
PYTORCH_ENABLE_MPS_FALLBACK=1 \
/private/tmp/football_tracker_verify_20261119/bin/football-tracker \
  /private/tmp/roboflow-pan-6s.mp4 \
  --output outputs/pitch-pan-6s.mp4 \
  --model /private/tmp/football-player-detection.pt \
  --pitch-model /private/tmp/football-pitch-detection.pt \
  --device mps \
  --image-size 960 \
  --confidence 0.25 \
  --team-sample-seconds 1.0
```

Observed result:

```text
Wrote 150 frames at 25.000 FPS to outputs/pitch-pan-6s.mp4 (26 tracks, 0 camera-cut resets). Team classification ready (goalkeeper/unknown=1, referee=3, team-a=10, team-b=12). Pitch calibration trusted on 150/150 frames with 3259 visible positions projected.
```

Output metadata:

```text
codec_name=mpeg4
width=1920
height=1490
r_frame_rate=25/1
avg_frame_rate=25/1
duration=6.000000
nb_frames=150
```

The source was 1920×1080; the extra 410 rows are the tactical panel. Frame count, FPS, and
duration match the input. The combined player/pitch tracking pass averaged approximately 4.1
frames/second on the available Apple Silicon MPS device.

Frames 0, 25, 50, 75, 100, and 125 were inspected. Frame 0 had a trusted calibration but no
confirmed ByteTrack detections yet, so the pitch correctly contained no dots. Subsequent
frames showed only current visible tracks: team-coloured player circles, yellow referee
squares, and a magenta goalkeeper diamond, each labelled with its temporary `T` ID.

The landmark fit updated throughout the pan and never dropped out in this six-second shot.
No large systematic projection drift was apparent in the contact sheet. Small frame-to-frame
dot jitter was visible, especially near the touchlines where foot positions are farther from
the strongest midfield landmarks. There is no labelled pitch-position ground truth, so this
visual observation is not a metric accuracy claim.

### Camera-cut command and result

The same command was run with `/private/tmp/roboflow-pan-cut-12s.mp4` and output
`outputs/pitch-pan-cut-12s.mp4`.

Observed result:

```text
Wrote 300 frames at 25.000 FPS to outputs/pitch-pan-cut-12s.mp4 (50 tracks, 1 camera-cut resets). Team classification ready (goalkeeper/unknown=1, referee=5, team-a=20, team-b=24). Pitch calibration trusted on 224/300 frames with 4847 visible positions projected.
```

Output metadata:

```text
codec_name=mpeg4
width=1920
height=1490
r_frame_rate=25/1
avg_frame_rate=25/1
duration=12.000000
nb_frames=300
```

A pitch-only diagnostic measured the exact calibration intervals:

```text
trusted                                      224 frames  0-149, 151-224
camera cut                                     1 frame   150
too few landmarks agreed with one camera view 75 frames  225-299
trusted median landmark error                            0.70-3.89 px
trusted usable landmarks                                7-9 per frame
```

Observed cut and dropout behaviour:

- Frame 150 immediately discarded the old transform and displayed an empty pitch with
  `CALIBRATION UNAVAILABLE — camera cut`; no stale dots crossed the edit.
- Frame 151 earned a fresh transform, but ByteTrack had not confirmed new tracks yet, so the
  pitch remained dot-free. Frame 152 showed the newly confirmed tracks as grey `unknown`
  markers while shirt votes accumulated.
- The tighter second shot remained trusted through frame 224. From frame 225 (9.0 seconds)
  to the end, fewer than six landmarks agreed with a single homography; all dots were hidden
  for those 75 frames. This is the largest observed dropout and is preferable to extrapolating
  player positions from a weak or stale transform.
- Frames 190, 205, 220, and 224 showed mild jitter but no obvious gross drift immediately
  before rejection. Pixel landmark error measures internal fit, not true player-location
  error, so metric statistics remain unavailable.

`3259` and `4847` are cumulative current-frame projections, not unique people and not
fabricated off-screen positions. A detection is omitted if calibration is unavailable or its
projected foot falls outside the pitch bounds.

### Replay verification checks

- Both generated MP4 files decoded from start to finish with FFmpeg and no reported errors.
- The complete suite finished with 32 passing tests.
- `ruff check src tests` reported no issues.
- Added coverage includes synthetic homography recovery, landmark/coverage rejection, explicit
  cut invalidation, current-frame-only projection, role-aware tactical markers, vertical frame
  composition, and preservation of source frame count and FPS when the panel is enabled.

## Jersey-number and roster suggestions — 19 November 2026

The optional roster was built from
[FC Bayern's official lineup](https://fcbayern.com/en/matches/profis/bundesliga/2020-2021/vfb-stuttgart-fc-bayern-muenchen-28-11-2020/lineup)
for the 28 November 2020 Stuttgart–Bayern match. Visual shirt classification maps Stuttgart's
white kit to `team-a`
and Bayern's red kit to `team-b`. The matching CSV included both teams' starting numbers;
overlapping numbers 1, 5, 7, 8, and 24 intentionally map to different names by team.

The optional EasyOCR 1.7.2 extra was installed only in the temporary verification environment.
Its automatic model download failed because that Python installation lacked a usable CA chain,
so the two official EasyOCR release archives were downloaded with certificate-checking `curl`.
The extracted weight MD5 values matched EasyOCR's published configuration. No weights, roster,
clip, or generated video were added to Git.

EasyOCR initialization with `gpu="mps"` reproducibly exited with code 139 before tracking.
The adapter now uses quantized CPU OCR when the main detector device is MPS; YOLO and pitch
inference remain on MPS. A standalone initialization and the two real replay runs below then
completed.

### Six-second wide pan

The roster-enabled version of the existing wide-pan command added:

```bash
--roster /private/tmp/stuttgart-bayern-roster.csv \
--jersey-model-directory /private/tmp/football_tracker_easyocr_models
```

Observed result:

```text
Wrote 150 frames at 25.000 FPS to outputs/jersey-wide-pan-6s.mp4 (26 tracks, 0 camera-cut resets). Team classification ready (goalkeeper/unknown=1, referee=3, team-a=10, team-b=12). Pitch calibration trusted on 150/150 frames with 3259 visible positions projected. Jersey suggestions: 0 tracks from 0 accepted reads; 22 unresolved.
```

Output metadata remained 1920×1490, 150 frames, 25 FPS, and 6.000 seconds. Frame 75 was
inspected: outfield labels showed only `T` IDs on both views, while referees retained their
role label. No name was silently guessed.

### Four-second cut fixture

A 100-frame fixture containing two seconds before and two seconds after the already documented
constructed cut was run with the same roster, OCR, detector, and pitch model. Observed result:

```text
Wrote 100 frames at 25.000 FPS to outputs/jersey-cut-4s.mp4 (46 tracks, 1 camera-cut resets). Team classification ready (goalkeeper/unknown=1, referee=5, team-a=20, team-b=20). Pitch calibration trusted on 99/100 frames with 2131 visible positions projected. Jersey suggestions: 0 tracks from 0 accepted reads across 0 clear crops; 40 unresolved.
```

Output metadata remained 1920×1490, 100 frames, 25 FPS, and 4.000 seconds. Frames 49, 50,
51, and 60 were inspected. The cut frame had no tracks or tactical dots, the next frame still
had no confirmed tracks, and newly acquired segment-2 IDs appeared later with no carried name
evidence. Unit coverage separately proves that even a pre-cut three-vote suggestion needs
three fresh compatible reads on a new post-cut track.

### Honest recognition result

- Correct suggestions: **0**.
- Wrong suggestions: **0**, because the system abstained; this is not an accuracy claim.
- Unresolved outfield tracks: **22** in the wide pan and **40** across the two cut segments.
- Accepted number reads: **0** in both real runs.

Detector measurements on representative frames found player boxes roughly 49–131 pixels tall.
The number itself occupies only a small fraction of those pixels and is often front-facing,
blurred, or occluded. A few single-frame shapes look number-like to a human at full resolution,
but the current footage does not contain two numbers readable consistently across several
clear track crops. It therefore cannot satisfy the requested two-player recognition test, and
claiming correct names from it would be misleading.

The needed follow-up footage is 1080p or 4K medium/close action with at least two known players
back-facing for several consecutive sharp frames, limited occlusion, and torso/number regions
that occupy tens of pixels, plus the actual roster. The current EasyOCR adapter is a baseline;
a soccer-specific tracklet recognizer should be evaluated on that footage before relying on
name suggestions.

At the end of this milestone, all **47 tests** passed and `ruff check src tests` reported no
issues. The generated six-second pan and four-second cut MP4s also decoded fully with FFmpeg.
