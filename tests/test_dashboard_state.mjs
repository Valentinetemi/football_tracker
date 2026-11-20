import assert from "node:assert/strict";
import test from "node:test";

import {
  buildTrackCatalog,
  buildTrailSegments,
  findNearestPlayer,
  frameIndexForTime,
  noteJumpTime,
  playerAtFrame,
} from "../dashboard/replay-state.mjs";

const player = (trackId, x, y, team = "team-a") => ({
  track_id: trackId,
  x_cm: x,
  y_cm: y,
  role: "player",
  team,
  label: team ?? "unknown",
});

test("maps video time to a bounded source frame", () => {
  assert.equal(frameIndexForTime(12.04, 25, 1500), 301);
  assert.equal(frameIndexForTime(90, 25, 1500), 1499);
  assert.equal(frameIndexForTime(-2, 25, 1500), 0);
  assert.equal(noteJumpTime(14.5), 12.5);
  assert.equal(noteJumpTime(1), 0);
});

test("keeps a selected track stable when it is unavailable", () => {
  const frame = {frame: 10, players: [player(8, 500, 600)]};
  assert.equal(playerAtFrame(frame, 7), null);
  assert.equal(playerAtFrame(frame, 8).track_id, 8);
});

test("breaks trails at missing positions and camera-cut segments", () => {
  const frames = [
    {frame: 0, segment: 1, camera_cut: false, players: [player(7, 100, 100)]},
    {frame: 1, segment: 1, camera_cut: false, players: [player(7, 110, 100)]},
    {frame: 2, segment: 1, camera_cut: false, players: [player(7, null, null)]},
    {frame: 3, segment: 1, camera_cut: false, players: [player(7, 130, 100)]},
    {frame: 4, segment: 2, camera_cut: true, players: []},
    {frame: 5, segment: 2, camera_cut: false, players: [player(28, 900, 900, "team-b")]},
  ];
  assert.deepEqual(
    buildTrailSegments(frames, 3, 7, 10).map((segment) => segment.map((point) => point.frame)),
    [[0, 1], [3]],
  );
  assert.deepEqual(buildTrailSegments(frames, 5, 7, 10), []);
});

test("builds track choices and supports pitch clicking", () => {
  const frames = [
    {
      frame: 0,
      segment: 1,
      camera_cut: false,
      players: [player(7, 100, 100, null), player(8, 800, 800, "team-b")],
    },
    {
      frame: 1,
      segment: 1,
      camera_cut: false,
      players: [player(7, 120, 120, "team-a")],
    },
  ];
  const catalog = buildTrackCatalog(frames);
  assert.deepEqual(catalog.map((track) => track.track_id), [7, 8]);
  assert.equal(catalog[0].team, "team-a");
  assert.equal(findNearestPlayer(frames[0].players, 790, 790, 100).track_id, 8);
  assert.equal(findNearestPlayer(frames[0].players, 4000, 4000, 100), null);
});
