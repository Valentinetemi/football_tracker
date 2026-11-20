# STEM2Connect real-footage validation — 20 November 2026

## Outcome

The existing demo artifacts complete a 60-second operational validation of player tracking,
ball tracking, pitch calibration, camera-cut handling, annotated footage, and tactical replay.
All three videos have verified 60.000-second timelines with 1,500 frames at 25 FPS, and the two
rendered outputs were fully decoded without errors during validation.

This is an operational validation, not an accuracy benchmark. The footage has no labelled
player, ball, or pitch ground truth, so this report does not claim MOTA, IDF1, precision, recall,
or metric position accuracy.

## Artifacts

| Artifact | Purpose |
| --- | --- |
| [`stem2connect-validation-input-60s.mp4`](../outputs/stem2connect-validation-input-60s.mp4) | 60-second real-footage fixture |
| [`stem2connect-annotated-60s.mp4`](../outputs/stem2connect-annotated-60s.mp4) | Player IDs, roles/teams, trails, and ball annotation |
| [`stem2connect-tactical-replay-60s.mp4`](../outputs/stem2connect-tactical-replay-60s.mp4) | Annotated footage plus calibrated tactical panel |
| [`stem2connect-validation-metrics.json`](../outputs/stem2connect-validation-metrics.json) | Per-frame observations and aggregate run metrics |
| [`stem2connect-overall-contact-sheet.jpg`](../outputs/stem2connect-overall-contact-sheet.jpg) | Samples across the complete fixture |
| [`stem2connect-cut-contact-sheet.jpg`](../outputs/stem2connect-cut-contact-sheet.jpg) | Frames around the constructed transition |
| [`stem2connect-calibration-contact-sheet.jpg`](../outputs/stem2connect-calibration-contact-sheet.jpg) | Trusted/untrusted calibration examples |
| [`stem2connect-ball-contact-sheet.jpg`](../outputs/stem2connect-ball-contact-sheet.jpg) | Ball-visible and ball-hidden examples |

The annotated video is the 720-pixel-high footage panel from the same tracking run used for
the tactical replay; it is not a separate inference pass.

## Footage and timing

The fixture concatenates the two 30-second real match samples `0bfacc_0.mp4` and
`2e57b9_0.mp4`, both from the Roboflow Sports soccer example. The join is frame 750 at
30.000 seconds. It deliberately exercises reset handling, but it is a cross-match constructed
transition rather than an organic broadcast cut.

Fresh `ffprobe` measurements of the existing files:

| File | Codec | Resolution | FPS | Frames | Duration | Bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Validation input | H.264 | 1280×720 | 25 | 1,500 | 60.000 s | 27,704,890 |
| Annotated output | H.264 | 1280×720 | 25 | 1,500 | 60.000 s | 25,750,773 |
| Tactical replay | H.264 | 1280×994 | 25 | 1,500 | 60.000 s | 22,961,904 |

The input and both outputs therefore preserve the expected frame count, frame rate, and
duration. No audio is present in the generated files.

## Run configuration

The metrics record an MPS run with the football player and pitch checkpoints, player
confidence 0.25, player inference size 1280, pitch inference size 640, team classification,
and ball tracking enabled at confidence 0.35. The equivalent project CLI shape is:

```bash
football-tracker outputs/stem2connect-validation-input-60s.mp4 \
  --output outputs/stem2connect-tactical-replay-60s.mp4 \
  --model data/football-player-detection.pt \
  --pitch-model data/football-pitch-detection.pt \
  --device mps \
  --image-size 1280 \
  --confidence 0.25 \
  --team-sample-seconds 1.0 \
  --track-ball \
  --ball-confidence 0.35
```

The validation runner instrumented the same pipeline to collect per-frame evidence; it did
not modify the tracking behavior.

## Player tracking and ID stability

The run produced 70 globally unique display IDs. Frames contained a mean of 22.99 tracks,
a median of 23, and a 95th percentile of 24. Of the 70 IDs, 60 were observed for at least one
second and 53 for at least five seconds. The median ID span was 748 frames and the median
number of missing frames inside an ID's span was zero.

Fifteen IDs listed in the metrics were present without an internal gap on every frame from
frame 2 through frame 749 (748 frames). The contact-sheet samples show the same displayed IDs
on continuously visible people, with no definite within-shot switch visible in those samples.
That is useful evidence of display-ID stability, but sparse visual samples cannot prove that
identity was correct on every frame.

The reset occurred at frame 750. Frames 750 and 751 had no confirmed person tracks; frame 752
reacquired 24 tracks with new IDs. IDs 1–27 occur only before the cut, IDs 28–70 only after it,
and the metrics report no ID reused across the two segments. This is deliberate cut isolation,
not cross-camera identity continuity. The 70 IDs must not be interpreted as 70 distinct people:
short-lived detections and post-cut IDs contribute to that total.

![Samples across the run](../outputs/stem2connect-overall-contact-sheet.jpg)

## Ball visibility

The detector produced ball candidates on 856 frames. The temporal tracker considered the ball
reliable enough to display on 700 frames, or 46.7% of the fixture. It recorded 444 missed
frames, 10 ambiguous frames, 97 recoveries, and one camera-cut reset. These are pipeline-state
counts, not detection precision or recall.

The ball marker is present in the contact sheet's visible samples and absent in its selected
hidden samples. Around the transition it is visible through frame 749, is cleared by the reset,
and returns at frame 759 after fresh evidence. The longest recorded hidden intervals include
two 52-frame spans, so ball visibility is intermittent and should not be presented as
continuous tracking.

![Ball-visible and ball-hidden samples](../outputs/stem2connect-ball-contact-sheet.jpg)

## Pitch calibration and tactical replay

Calibration was trusted on 1,466 of 1,500 frames (97.7% operational availability), producing
32,704 projected positions. The 34 unavailable frames comprise:

- 27 frames where too few landmarks agreed with one camera view;
- 6 frames where visible landmarks covered too little of the pitch; and
- 1 camera-cut frame.

On trusted frames, the median internal landmark reprojection error was 2.57 pixels and the
95th percentile was 4.51 pixels. This measures consistency between detected landmarks and the
fitted transform; it is not ground-truth player-location error. Player foot points are
projected onto the pitch plane, and an airborne ball's projected location is necessarily an
approximation.

The calibration contact sheet confirms fail-closed behavior: the tactical panel omits projected
dots when calibration is untrusted and resumes them after trust returns. No stale homography is
carried across the cut.

![Calibration dropout and recovery samples](../outputs/stem2connect-calibration-contact-sheet.jpg)

## Camera-cut behavior

The expected join and the single tracker reset both occur at frame 750. The cut contact sheet
shows the reset annotation at that frame, an empty tactical panel during reset, no confirmed
tracks on frame 751, new globally unique IDs by frame 752, and team colors plus a ball marker
again by frame 759. No additional reset was recorded.

![Frames around the constructed cut](../outputs/stem2connect-cut-contact-sheet.jpg)

## Team classification

The prepass collected 600 appearance samples and fitted two clusters of 303 and 297 samples;
the recorded separation was 0.746. Final per-track labels were 27 `team-a`, 15 `team-b`,
12 `referee`, 7 `goalkeeper/unknown`, and 9 `unknown`. These are latest labels for unique
tracks, not counts of people on the pitch.

The fixture combines two different matches and kits, while classification fits one global pair
of appearance clusters. Consequently, `team-a` and `team-b` are anonymous visual clusters and
are not semantic club identities across both halves. The run does not establish team-label
accuracy. Some `unknown` abstentions and dense label overlap remain visible in the renders.

## Processing performance

The instrumented run took 644.65 seconds (10 minutes 44.65 seconds) end to end and processed
2.327 frames per second. For a 25 FPS source, that is approximately 10.7 times slower than
real time on the available Apple Silicon MPS environment. The team prepass took 10.88 seconds
before the first tracked frame.

Across 1,530 player-detector calls, mean latency was 0.300 seconds, median 0.303 seconds, and
95th percentile 0.362 seconds. Across 1,499 pitch-detector calls, mean latency was 0.093 seconds,
median 0.094 seconds, and 95th percentile 0.111 seconds. These are measurements from this
fixture and environment, not a general performance guarantee.

## Automated checks

- Existing test result: `57 passed in 30.10s`.
- Current lint result: `ruff check .` → `All checks passed!`.
- No reproducible critical product bug was found, so no product-code fix was made.

## Evidence-backed limitations

- There is no labelled ground truth, so the run cannot support formal tracking, ball,
  classification, or calibration accuracy metrics.
- Display IDs are temporary tracker IDs, not verified player identities. They intentionally
  change after a camera cut, and short-lived detections inflate the unique-ID total.
- Ball display availability is 46.7%, with gaps as long as 52 frames in this fixture.
- Calibration is unavailable on 34 frames and only validates an internal landmark fit when it
  is trusted.
- The only cut is a constructed cross-match join; organic broadcast cuts, dissolves, replays,
  and long occlusions need additional real-match validation.
- Global two-cluster team fitting is not semantically valid across the two different matches.
- Dense labels can overlap. The outputs have no audio.
- Roster/OCR matching was not exercised. Possession, speed, distance, and spacing metrics are
  outside the current repository's supported output.
