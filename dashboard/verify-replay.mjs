import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";

import {
  buildTrackCatalog,
  buildTrailSegments,
  noteJumpTime,
  playerAtFrame,
} from "./replay-state.mjs";

const dataPath = process.argv[2] ?? "outputs/stem2connect-replay-data.json";
const replay = JSON.parse(await readFile(dataPath, "utf8"));
const {frames} = replay;
const fps = Number(replay.video.fps);
const trailLength = 75;

assert.equal(frames.length, 1500, "expected the complete 60-second fixture");
assert.equal(fps, 25, "expected the verified source frame rate");

const hasPosition = (player) =>
  player && Number.isFinite(player.x_cm) && Number.isFinite(player.y_cm);

let best = null;
for (const track of buildTrackCatalog(frames).filter((item) => item.role === "player")) {
  let run = [];
  let previousFrame = null;
  let previousSegment = null;
  const scoreRun = () => {
    if (run.length < trailLength) return;
    for (let start = 0; start <= run.length - trailLength; start += 1) {
      const window = run.slice(start, start + trailLength);
      const first = window[0];
      const last = window.at(-1);
      const displacement = Math.hypot(last.x_cm - first.x_cm, last.y_cm - first.y_cm);
      if (!best || displacement > best.displacement) {
        best = {
          trackId: track.track_id,
          team: track.team,
          startFrame: first.frame,
          endFrame: last.frame,
          displacement,
        };
      }
    }
  };

  for (const frame of frames) {
    const player = playerAtFrame(frame, track.track_id);
    const continuous =
      hasPosition(player) &&
      !frame.camera_cut &&
      previousFrame != null &&
      frame.frame === previousFrame + 1 &&
      frame.segment === previousSegment;
    if (!continuous) {
      scoreRun();
      run = [];
    }
    if (hasPosition(player) && !frame.camera_cut) {
      run.push({...player, frame: frame.frame});
      previousFrame = frame.frame;
      previousSegment = frame.segment;
    } else {
      previousFrame = null;
      previousSegment = null;
    }
  }
  scoreRun();
}

assert(best, "expected at least one continuous player trail");
const demoTrail = buildTrailSegments(frames, best.endFrame, best.trackId, trailLength);
assert.equal(demoTrail.at(-1).length, trailLength, "demo trail must be continuous");
assert.equal(demoTrail.at(-1).at(-1).frame, best.endFrame);

const backwardFrame = best.endFrame - 50;
const backwardTrail = buildTrailSegments(frames, backwardFrame, best.trackId, trailLength);
assert(backwardTrail.length > 0, "backward seek should rebuild an earlier trail");
assert.equal(backwardTrail.at(-1).at(-1).frame, backwardFrame);

const cutFrame = frames.find((frame) => frame.camera_cut);
assert(cutFrame, "expected a camera-cut frame");
const beforeCut = frames[cutFrame.frame - 1];
const selectedBeforeCut = beforeCut.players.find(hasPosition);
assert(selectedBeforeCut, "expected a selectable track before the cut");
assert.equal(playerAtFrame(cutFrame, selectedBeforeCut.track_id), null);
assert.equal(playerAtFrame(frames[cutFrame.frame + 2], selectedBeforeCut.track_id), null);
assert.deepEqual(
  buildTrailSegments(frames, cutFrame.frame + 2, selectedBeforeCut.track_id, trailLength),
  [],
  "a trail must not cross the cut into the new tracker segment",
);

const demoStartSeconds = best.startFrame / fps;
const demoEndSeconds = best.endFrame / fps;
assert.equal(noteJumpTime(demoEndSeconds), Math.max(0, demoEndSeconds - 2));

console.log(
  JSON.stringify(
    {
      frames: frames.length,
      fps,
      follow_demo: {
        track_id: best.trackId,
        team: best.team,
        start_frame: best.startFrame,
        start_seconds: demoStartSeconds,
        end_frame: best.endFrame,
        end_seconds: demoEndSeconds,
        continuous_trail_frames: trailLength,
      },
      backward_seek: {
        frame: backwardFrame,
        seconds: backwardFrame / fps,
        selected_track_id: best.trackId,
      },
      camera_cut: {
        frame: cutFrame.frame,
        seconds: cutFrame.frame / fps,
        selected_track_id: selectedBeforeCut.track_id,
        selection_preserved: true,
        visible_player_after_cut: null,
        trail_segments_after_cut: 0,
      },
      note_jump_seconds: noteJumpTime(demoEndSeconds),
    },
    null,
    2,
  ),
);
