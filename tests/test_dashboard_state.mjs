import assert from "node:assert/strict";
import test from "node:test";

import {
  ballPosition,
  buildTrackCatalog,
  buildTrailSegments,
  entityLabel,
  findNearestPlayer,
  followStatusText,
  formatSuggestedName,
  frameIndexForTime,
  noteJumpTime,
  playerAtFrame,
  trackMenuLabel,
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

test("labels goalkeepers as GK and keeps the question mark on a suggested name", () => {
  const goalkeeper = {
    track_id: 4,
    role: "goalkeeper",
    team: null,
    label: "goalkeeper/unknown",
    suggested_name: null,
  };
  assert.equal(entityLabel(goalkeeper), "GK");
  assert.equal(trackMenuLabel(goalkeeper), "T4 · GK");
  assert.equal(trackMenuLabel(goalkeeper).includes("goalkeeper/unknown"), false);

  const named = {
    track_id: 7,
    role: "player",
    team: "team-a",
    label: "team-a",
    suggested_name: "Ada Example",
  };
  assert.equal(formatSuggestedName("Ada Example"), "Ada Example?");
  assert.equal(formatSuggestedName("Ada Example?"), "Ada Example?");
  assert.equal(trackMenuLabel(named), "T7 · team-a · Ada Example?");
});

test("catalog keeps a suggested name and does not invent one", () => {
  const frames = [
    {
      frame: 0,
      players: [
        {
          track_id: 7,
          role: "player",
          team: null,
          label: "unknown",
          suggested_name: null,
        },
      ],
    },
    {
      frame: 1,
      players: [
        {
          track_id: 7,
          role: "player",
          team: "team-a",
          label: "team-a",
          suggested_name: "Ada",
        },
      ],
    },
  ];
  const catalog = buildTrackCatalog(frames);
  assert.equal(catalog[0].suggested_name, "Ada");
  assert.equal(catalog[0].team, "team-a");
  const unnamed = buildTrackCatalog([
    {frame: 0, players: [{track_id: 3, role: "referee", team: null, label: "referee"}]},
  ]);
  assert.equal(unnamed[0].suggested_name, null);
  assert.equal(trackMenuLabel(unnamed[0]), "T3 · referee");
});

test("draws a ball only when it is visible on a trusted calibration", () => {
  const visible = {
    calibration_trusted: true,
    ball: {visible: true, reliable: true, x_cm: 10, y_cm: 20},
  };
  assert.deepEqual(ballPosition(visible), {x_cm: 10, y_cm: 20});
  assert.equal(ballPosition({...visible, calibration_trusted: false}), null);
  assert.equal(
    ballPosition({
      calibration_trusted: true,
      ball: {visible: false, reliable: true, x_cm: 10, y_cm: 20},
    }),
    null,
  );
  assert.equal(
    ballPosition({
      calibration_trusted: true,
      ball: {visible: true, reliable: false, x_cm: 10, y_cm: 20},
    }),
    null,
  );
  assert.equal(
    ballPosition({
      calibration_trusted: true,
      ball: {visible: true, reliable: true, x_cm: null, y_cm: null},
    }),
    null,
  );
  assert.equal(ballPosition({calibration_trusted: true, players: []}), null);
});

test("keeps the follow status on the same track when that player disappears", () => {
  const hidden = {frame: 3, players: [{track_id: 8, x_cm: 10, y_cm: 10, role: "player", team: "team-b", label: "team-b"}]};
  const catalogTrack = {
    track_id: 7,
    role: "player",
    team: "team-a",
    label: "team-a",
    suggested_name: "Ada",
  };
  const text = followStatusText(7, hidden, catalogTrack);
  assert.match(text, /T7/);
  assert.match(text, /Ada\?/);
  assert.match(text, /Player not visible/);
  assert.equal(text.includes("T8"), false);
  const visible = {
    frame: 4,
    players: [
      {
        track_id: 7,
        x_cm: 1,
        y_cm: 2,
        role: "goalkeeper",
        team: null,
        label: "goalkeeper/unknown",
        suggested_name: null,
      },
    ],
  };
  assert.equal(followStatusText(7, visible, catalogTrack), "Following T7 · GK");
  const namedLater = {
    frame: 1,
    players: [
      {
        track_id: 7,
        x_cm: 1,
        y_cm: 2,
        role: "player",
        team: "team-a",
        label: "team-a",
        suggested_name: null,
      },
    ],
  };
  assert.equal(followStatusText(7, namedLater, catalogTrack), "Following T7 · team-a");
  assert.equal(
    followStatusText(null, visible, null),
    "Select a player on the pitch or from the list.",
  );
});
