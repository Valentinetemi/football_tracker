# Football Video Tracker

School and club coaches sit down with match film and try to answer simple questions: where
was this player, what was happening around them, and which moment is worth stopping for at
training. Paid analysis tools assume a budget, a tactical analyst, or a cloud account. This
project is a local alternative for that review.

It reads a match clip on your own computer, detects the people the camera can see, follows
them with temporary track IDs, separates two anonymous shirt colours, and — when the pitch
markings support it — places those players on a top-down pitch. An optional mode can mark the
ball when the detector and a short motion check agree. A local dashboard lets you follow one
player, scrub the clip, and tag moments in the browser.

Track labels such as `T7` are temporary tracker IDs. They are not real player identities. A
suggested name always keeps its question mark. The pitch is left empty when calibration is
not trusted, and the ball is omitted when it was not reliably seen. Nothing in this repository
invents a position to fill a gap.

This is a STEM2Connect AI Hackathon demo. It is not a finished scouting product, and it does
not claim measured tracking accuracy. A 60-second operational note lives in
[docs/stem2connect-validation.md](docs/stem2connect-validation.md). That note records what one
run did. It is not an accuracy benchmark.

## Who it is for

- Coaches and club volunteers who already have match footage and do not have a paid analysis
  subscription.
- Students who want to see how detection, tracking, and calibration behave on a real sport
  video, including the frames where the system should abstain.
- Hackathon judges, who can open the local dashboard without downloading YOLO weights or
  rerunning a match.

It is not a live officiating tool. It does not decide possession, and it does not turn a
shirt number into a confirmed identity.

## Judge quickstart

You need Python 3.11, 3.12, or 3.13. From the repository root:

```bash
python -m pip install -e . && python -m football_tracker.dashboard
```

That command serves the annotated clip and the replay JSON. It does not download model
weights and it does not run the match again. A fresh clone does not contain those two
files, so the command stops until they are downloaded into `outputs/` (see
[Demo files](#demo-files)). The message names the missing paths and how to pass `--video`
and `--data`. After the files are in place, open
[http://127.0.0.1:8765](http://127.0.0.1:8765).

The page reads `track` and `time` query parameters, so a prepared moment has this shape:

```text
http://127.0.0.1:8765/?track=TRACK_ID&time=SECONDS
```

`track` selects that temporary ID and keeps it selected even if the player leaves the frame.
`time` seeks the video, in seconds. To choose a real moment from the release replay, run:

```bash
node dashboard/verify-replay.mjs outputs/stem2connect-replay-data.json
```

Use `follow_demo.track_id` and `follow_demo.end_seconds` from the printed JSON.

**TODO (owner):** this is the shape of the link, not a working moment. After the release
replay is on disk, run the command above and replace the line below with the printed
`follow_demo.track_id` and `follow_demo.end_seconds`. Judges should not be sent `TRACK_ID`.

```text
http://127.0.0.1:8765/?track=TRACK_ID&time=SECONDS
```

## Demo files

The annotated video and the replay JSON are generated files. They are gitignored, so a fresh
clone does not contain them. The project owner will upload them as a GitHub Release.

| Release asset | Place it at |
| --- | --- |
| `stem2connect-annotated-60s.mp4` | `outputs/stem2connect-annotated-60s.mp4` |
| `stem2connect-replay-data.json` | `outputs/stem2connect-replay-data.json` |

**TODO (owner):** publish the release, then replace the placeholder URLs below with the real
asset links.

```text
https://github.com/Valentinetemi/football_tracker/releases/download/TODO-tag/stem2connect-annotated-60s.mp4
https://github.com/Valentinetemi/football_tracker/releases/download/TODO-tag/stem2connect-replay-data.json
```

After the release exists, replace `TODO-tag` in these commands with that tag and run them.
Each command already uses the full asset URL, so do not add the filename a second time.

```bash
mkdir -p outputs
curl -fL -o outputs/stem2connect-annotated-60s.mp4 \
  "https://github.com/Valentinetemi/football_tracker/releases/download/TODO-tag/stem2connect-annotated-60s.mp4"
curl -fL -o outputs/stem2connect-replay-data.json \
  "https://github.com/Valentinetemi/football_tracker/releases/download/TODO-tag/stem2connect-replay-data.json"
```

You can also point the server at any other annotated MP4 and replay JSON:

```bash
python -m football_tracker.dashboard \
  --video path/to/annotated.mp4 \
  --data path/to/replay-data.json
```

Run that command from the repository root so it can see the `dashboard/` assets, or pass
`--static-directory`.

OpenCV's default `mp4v` output often will not play in a browser. Prefer H.264. The validated
STEM2Connect renders described in the validation note were H.264, so use that annotated file
for the dashboard. When you generate a new file and the local OpenCV build can encode H.264,
pass `--codec avc1`. If a build cannot open `avc1`, `--codec mp4v` still writes a file that
desktop players may open, but the dashboard page may show the playback error instead of the
picture.

The interactive pitch draws the ball only when the replay JSON includes a per-frame `ball`
object with `visible: true` and the frame's calibration is trusted. Replay files exported
before that field existed omit it, so the pitch stays without a ball until the clip is
exported again. The annotated video from the earlier run can still show the ball marker.
Producing a new JSON means running the pipeline; this repository does not ship that rerun.

## What the dashboard shows

- The annotated broadcast, with seeking kept in step with the pitch.
- An interactive pitch. Click a marker or pick a track from the list.
- Team A and Team B as anonymous shirt clusters, plus GK, referee, and the ball. The colour
  legend is on the pitch panel. GK is the dashboard label for a goalkeeper whose team is not
  inferred. The replay JSON still stores `goalkeeper/unknown` for that case.
- A suggested name, when the replay includes `suggested_name`, with the `?` left in place.
- The same selected track when that player disappears. The status says they are not visible.
  The pitch does not move the highlight to someone else, and it does not invent a coordinate.
- Moment tags saved in this browser's `localStorage`. Tags do not change the replay JSON or
  the repository. Opening a tag jumps two seconds before the moment.

## Current status

| Capability | Status |
| --- | --- |
| Local video CLI | Working |
| Pretrained person or football-object detection | Working |
| ByteTrack IDs and movement trails | Working |
| Camera-cut tracker reset | Working, histogram-based baseline |
| MP4 export at source resolution and nominal FPS | Working for constant-frame-rate input |
| Team shirt classification and team colours | Working baseline with explicit uncertainty |
| Goalkeeper and referee treatment | Separate role colours; goalkeepers are not forced into an outfield-shirt cluster |
| Pitch calibration and tactical view | Working wide-shot baseline with per-frame trust gates |
| Jersey number and roster suggestions | Optional EasyOCR baseline; multi-frame consensus, roster-backed, and deliberately abstains on distant footage |
| Ball detection and tracking | Optional baseline; current-frame markers only, hidden when the ball is not reliable |
| Possession, speed, distance, and spacing | Not available yet |
| Local follow-player dashboard | Working, including moment tags. Needs the demo video and replay JSON, which are not in Git |

`team-a` and `team-b` are anonymous appearance clusters, not club names. Referees,
goalkeepers, and uncertain tracks receive separate labels and colours.

## Hackathon track

**TODO (owner): choose one primary track before you submit.** STEM2Connect asks for a
primary track. Pick the one that matches the story you want judges to read, and leave the
other one unchecked.

- [ ] Education
- [ ] Community Impact

Write one sentence on the Devpost form about why that track fits. Do not leave this choice
to a reviewer.

## AI disclosure

Detection and tracking use other people's pretrained models and libraries. The pitch gates,
shirt clustering, jersey consensus, and ball association are project code. Say which coding
assistants you used as well.

- **Detection:** Ultralytics YOLO. The default fallback is the small COCO checkpoint
  `yolo11n.pt`, which only has a person class. The football run uses the Roboflow Sports
  YOLOv8 player and pitch checkpoints. Those files are downloaded separately and are not in
  this repository.
- **Tracking:** Supervision 0.25.1 ByteTrack, following the public Roboflow Sports soccer
  example.
- **Everything around those models:** project code. OpenCV supplies video IO, annotations,
  and a RANSAC homography. A small NumPy two-cluster classifier separates shirt colours and
  abstains when a crop is ambiguous. Optional EasyOCR plus multi-frame voting can suggest a
  roster name. A conservative ball associator shows a marker only after consecutive,
  motion-consistent detections.
- **Coding assistants:** TODO (owner): name the assistants you used and what you used them
  for, such as implementation help, tests, or writing. If you used none, say so.

The tracker does not invent player positions, track IDs, or ball locations when a frame is
untrusted or the ball is not reliably detected.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for credits and licence notes.

## What I learned

TODO (owner): replace each bullet with your own words before submission. These headings are
only a scaffold.

### Before this project

- TODO

### What was difficult

- TODO

### What I would do differently

- TODO

## Limitations

- There is no labelled ground truth in this repository, so it cannot support a tracking,
  ball, classification, or calibration accuracy number.
- Display IDs are temporary. They change after a camera cut on purpose. A short-lived
  detection still consumes an ID.
- `team-a` and `team-b` are visual clusters for one run. They are not home and away, and a
  clip that joins two matches does not make those letters into club names.
- A roster name is a suggestion. The question mark is part of the result. Referees and
  goalkeepers do not receive a guessed team or a guessed name.
- The ball marker is the current measured detection after a short consistency check. Misses
  and ambiguous candidates are hidden. A hidden gap is not a predicted position. On the
  tactical pitch the ball also requires trusted calibration and a point that lands on the
  pitch. An airborne ball is only a ground-plane approximation.
- Calibration fails closed. A weak landmark fit, a camera cut, or a point outside the pitch
  produces no tactical position for that frame.
- The dashboard ball depends on the replay JSON containing the `ball` field. Older exports
  do not have it.
- OpenCV `mp4v` output may not play in the browser. Prefer H.264 / `avc1` for the dashboard
  video. Exported video has no audio, and variable-frame-rate input is normalized to the
  reported average FPS.
- Jersey OCR is a general-purpose baseline. It is not expected to read tiny broadcast
  digits. Possession, speed, distance, and spacing are not implemented.
- Dense labels can still overlap.

## Next steps

- Upload the H.264 annotated clip and the replay JSON as a GitHub Release, then replace the
  placeholder links and the prepared demo URL above.
- Re-export replay JSON with the current code if the interactive pitch should include the
  ball. That is a pipeline run; it is not part of this change.
- Try the same checks on organic broadcast cuts, dissolves, and replays. The published
  60-second note uses one constructed join between two samples.
- Add possession, speed, distance, or spacing only where the calibration and the tracks are
  good enough to support them, and mark those numbers unavailable when they are not.
- Replace the EasyOCR jersey baseline with a soccer-specific recognizer without changing the
  rule that a name stays tentative.

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
- Calibration is accepted only when landmark confidence, inlier agreement, image and pitch
  coverage, and pixel reprojection error pass. Failed frames show an empty tactical pitch
  with the reason. A cut is always invalid for that frame.
- Output is written to a temporary file and atomically moved into place only after success.
- Labels use a greedy nearby-placement pass to reduce collisions. Dense groups can still
  overlap; this is a readability baseline, not a full label-layout solver.
- **EasyOCR** is an optional, replaceable scene-text baseline. It only sees sharp central-
  torso crops from sufficiently large outfield-player tracks. A name is suggested only after
  at least three accepted reads agree and the exact `(team-a|team-b, number)` roster entry
  exists. Soccer-specific OCR is a future upgrade, not hidden behind this baseline.
- The Roboflow football checkpoint already contains a `ball` class, so `--track-ball` reuses
  the existing YOLO inference pass rather than loading a second model. A marker is shown only
  after consecutive, motion-consistent detections. Misses and ambiguous candidates are hidden
  instead of extrapolated. Ball association state is cleared at every camera cut.

The design is informed by the
[Roboflow Sports soccer example](https://github.com/roboflow/sports/tree/main/examples/soccer),
but this smaller implementation is written for this project.

## Run the full pipeline

Python 3.12 is recommended on a Mac. The current Homebrew `python3` may be 3.14, which this
project's PyTorch stack does not target.

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

For a file you want to open in the dashboard later, prefer H.264 when OpenCV can write it:

```bash
  --codec avc1
```

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
Enable its ball class without changing the default of leaving ball tracking off:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 football-tracker \
  inputs/match.mp4 \
  --output outputs/match-ball-replay.mp4 \
  --model models/football-player-detection.pt \
  --pitch-model models/football-pitch-detection.pt \
  --track-ball \
  --ball-confidence 0.35 \
  --codec avc1 \
  --device mps \
  --image-size 960
```

The default requires two consecutive reads before drawing the ball. A lost or ambiguous ball
is hidden immediately on both views. The retained gap is association memory only and never
draws a predicted position. The tactical dot is additionally gated by trusted calibration.
Because the homography models the ground plane, an airborne ball's tactical location is only
an approximation. `--ball-minimum-hits` and `--ball-maximum-gap` expose the main association
controls. Possession estimation is intentionally not implemented.

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
  --codec avc1 \
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
- weak evidence or ambiguous crops stay `unknown` instead of receiving a forced team;
- camera cuts reset tracking and per-track shirt votes, so new tracks briefly return to
  `unknown` while evidence accumulates.

The cluster letters are stable within one run but have no semantic home/away meaning. Tune
sampling and abstention with `--team-sample-seconds` and `--team-confidence`, or disable the
calibration pass with `--no-team-classification`.

### Optional roster and tentative name suggestions

First run the clip without a roster and inspect which real club the shirt classifier called
`team-a` and `team-b`. Then create a CSV with that explicit anonymous-team mapping. The same
shirt number can safely occur once for each opposing team:

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
  --codec avc1 \
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
evaluation footage, record or obtain 1080p or 4K medium or close broadcast shots where at
least two players are back-facing for several sharp frames and each torso and number occupies
tens of pixels. Keep the actual match roster alongside the clip. A future SoccerNet or
PARSeq-style tracklet model can replace `JerseyNumberOCR` without changing the replay
pipeline.

Use `football-tracker --help` for confidence, tracker buffer, camera-cut, class, device, and
codec options. If MPS is unavailable, use `--device cpu`. It will be slower.

### Write replay data for the dashboard

Replay data requires the same trusted pitch calibration used by the tactical panel:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 football-tracker \
  inputs/match.mp4 \
  --output outputs/match-tactical.mp4 \
  --model models/football-player-detection.pt \
  --pitch-model models/football-pitch-detection.pt \
  --track-ball \
  --codec avc1 \
  --device mps \
  --replay-data outputs/match-replay-data.json
```

Each frame's `ball` object is optional in the sense that older files omit it. New exports
always include it. `visible` is true only when `reliable` is true, calibration is trusted,
and the measured centre lands on the pitch. Otherwise `x_cm` and `y_cm` are null.

```bash
python -m football_tracker.dashboard \
  --video outputs/match-tactical.mp4 \
  --data outputs/match-replay-data.json
```

The tactical MP4 is taller than the broadcast because the pitch panel is stacked underneath.
For the dashboard, the annotated broadcast-only file is the easier video to watch. Either
file must be H.264 if you need browser playback.

## Timing and media behaviour

The processor decodes every frame and writes exactly one composite frame at the source-
reported FPS. Without a pitch model, source resolution is retained. With a pitch model, the
source-width tactical panel is appended below, so output height increases while frame count,
FPS, and constant-frame-rate duration stay unchanged. OpenCV output is video-only: source
audio is not copied, and variable-frame-rate timestamps are normalized to the reported
average FPS. Audio remux and true per-frame timestamps are a later export improvement.

The default codec remains `mp4v` so existing local players keep working. Browser playback of
the dashboard wants H.264, which you request with `--codec avc1` when the installed OpenCV
can create that writer.

## Verification

Run all checks from the repository root:

```bash
source .venv/bin/activate
pytest
node --test tests/test_dashboard_state.mjs
ruff check src tests
```

The pipeline was run on real Roboflow Sports football footage. The original 3-second check
and pitch-panel checks on a six-second pan and a 12-second constructed same-match camera cut
preserved input frame count, FPS, and duration. See
[docs/verification.md](docs/verification.md) for the exact commands, measured results,
observed ID behaviour, and visual limitations. The 60-second STEM2Connect note is
[docs/stem2connect-validation.md](docs/stem2connect-validation.md).

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
  replay.py       atomic per-frame replay-data export, including an optional ball
  dashboard.py    local range-aware dashboard server
  scene.py        camera-cut score
  tracking.py     ByteTrack lifecycle and global display IDs
  annotation.py   boxes, foot markers, labels, and image-space trails
  video.py        explicit reader and atomic source-FPS writer
  pipeline.py     end-to-end orchestration
tests/            unit tests plus a generated-video MP4 integration test
dashboard/        one-screen replay UI and actual-data verification script
```

No tactical position is fabricated outside the camera view. Statistics, if they are added
later, will be marked unavailable when detection, tracking, or calibration quality is
insufficient.
