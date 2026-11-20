import {
  ballPosition,
  buildTrackCatalog,
  buildTrailSegments,
  entityLabel,
  findNearestPlayer,
  followStatusText,
  formatSuggestedName,
  formatTimestamp,
  frameIndexForTime,
  noteJumpTime,
  playerAtFrame,
  trackMenuLabel,
} from "/replay-state.mjs";

const $ = (selector) => document.querySelector(selector);
const video = $("#match-video");
const canvas = $("#pitch-canvas");
const context = canvas.getContext("2d");
const playerSelect = $("#player-select");
const tagPlayer = $("#tag-player");
const clearSelection = $("#clear-selection");
const followStatus = $("#follow-status");
const timeLabel = $("#time-label");
const frameLabel = $("#frame-label");
const cutBadge = $("#cut-badge");
const pitchEmpty = $("#pitch-empty");
const videoNote = $("#video-note");
const timelineProgress = $("#timeline-progress");
const tagForm = $("#tag-form");
const tagTime = $("#tag-time");
const tagNote = $("#tag-note");
const tagMarkers = $("#tag-markers");
const tagList = $("#tag-list");

const STORAGE_KEY = "stem2connect-replay-notes-v1";
const TRAIL_FRAMES = 75;
const COLORS = {
  "team-a": "#3aa0ff",
  "team-b": "#ff5d73",
  referee: "#f5d451",
  goalkeeper: "#d56bff",
  unknown: "#aab6b1",
};

const response = await fetch("/replay-data.json", {cache: "no-store"});
if (!response.ok) throw new Error(`Replay data failed to load (${response.status}).`);
const replay = await response.json();
if (!Array.isArray(replay.frames) || replay.frames.length === 0) {
  throw new Error("Replay data contains no frames.");
}

const fps = Number(replay.video.fps);
const pitchLength = Number(replay.pitch.length_cm);
const pitchWidth = Number(replay.pitch.width_cm);
const catalog = buildTrackCatalog(replay.frames);
let selectedTrackId = null;
let currentFrameIndex = 0;
let playbackGeneration = 0;
let notes = loadNotes();
let lastPitchRect = null;

function hasPosition(player) {
  return player && Number.isFinite(player.x_cm) && Number.isFinite(player.y_cm);
}

function colorFor(player) {
  if (player?.team && COLORS[player.team]) return COLORS[player.team];
  if (entityLabel(player) === "GK") return COLORS.goalkeeper;
  return COLORS[player?.role] ?? COLORS.unknown;
}

function markerText(player) {
  if (entityLabel(player) === "GK") return `T${player.track_id} GK`;
  return `T${player.track_id}`;
}

function populateTrackSelectors() {
  for (const track of catalog) {
    const option = document.createElement("option");
    option.value = String(track.track_id);
    option.textContent = trackMenuLabel(track);
    playerSelect.append(option);

    const tagOption = option.cloneNode(true);
    tagPlayer.append(tagOption);
  }
}

function loadNotes() {
  try {
    const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]");
    if (!Array.isArray(stored)) return [];
    return stored
      .filter((note) => Number.isFinite(note.time) && typeof note.note === "string")
      .map((note) => ({
        id: String(note.id),
        time: Number(note.time),
        trackId: Number.isInteger(note.trackId) ? note.trackId : null,
        note: note.note.slice(0, 120),
      }))
      .sort((first, second) => first.time - second.time);
  } catch {
    return [];
  }
}

function saveNotes() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(notes));
}

function selectTrack(trackId) {
  selectedTrackId = trackId == null ? null : Number(trackId);
  playerSelect.value = selectedTrackId == null ? "" : String(selectedTrackId);
  if (selectedTrackId != null) tagPlayer.value = String(selectedTrackId);
  renderCurrentFrame();
}

function pitchRect(width, height) {
  const marginX = Math.max(22, width * 0.055);
  const marginY = Math.max(22, height * 0.09);
  const availableWidth = width - marginX * 2;
  const availableHeight = height - marginY * 2;
  const scale = Math.min(availableWidth / pitchLength, availableHeight / pitchWidth);
  const renderedWidth = pitchLength * scale;
  const renderedHeight = pitchWidth * scale;
  return {
    x: (width - renderedWidth) / 2,
    y: (height - renderedHeight) / 2,
    width: renderedWidth,
    height: renderedHeight,
    scale,
  };
}

function pitchPoint(rect, xCm, yCm) {
  return {x: rect.x + xCm * rect.scale, y: rect.y + yCm * rect.scale};
}

function drawPitch(rect) {
  context.fillStyle = "#0c1412";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#1f7a45";
  context.fillRect(rect.x, rect.y, rect.width, rect.height);
  context.strokeStyle = "rgba(246, 249, 239, 0.9)";
  context.lineWidth = Math.max(1.5, canvas.width / 650);
  context.strokeRect(rect.x, rect.y, rect.width, rect.height);

  const centre = pitchPoint(rect, pitchLength / 2, pitchWidth / 2);
  context.beginPath();
  context.moveTo(centre.x, rect.y);
  context.lineTo(centre.x, rect.y + rect.height);
  context.stroke();
  context.beginPath();
  context.arc(centre.x, centre.y, 915 * rect.scale, 0, Math.PI * 2);
  context.stroke();
  context.beginPath();
  context.arc(centre.x, centre.y, Math.max(2, 20 * rect.scale), 0, Math.PI * 2);
  context.fillStyle = "rgba(246, 249, 239, 0.9)";
  context.fill();

  const penaltyTop = (pitchWidth - 4100) / 2;
  const goalTop = (pitchWidth - 1832) / 2;
  for (const side of [0, 1]) {
    const direction = side === 0 ? 1 : -1;
    const edgeX = side === 0 ? 0 : pitchLength;
    const penaltyCorner = pitchPoint(rect, edgeX, penaltyTop);
    const penaltyFar = pitchPoint(rect, edgeX + direction * 2015, penaltyTop + 4100);
    context.strokeRect(
      Math.min(penaltyCorner.x, penaltyFar.x),
      penaltyCorner.y,
      Math.abs(penaltyFar.x - penaltyCorner.x),
      penaltyFar.y - penaltyCorner.y,
    );
    const goalCorner = pitchPoint(rect, edgeX, goalTop);
    const goalFar = pitchPoint(rect, edgeX + direction * 550, goalTop + 1832);
    context.strokeRect(
      Math.min(goalCorner.x, goalFar.x),
      goalCorner.y,
      Math.abs(goalFar.x - goalCorner.x),
      goalFar.y - goalCorner.y,
    );
  }
}

function drawTrail(rect, frame) {
  if (selectedTrackId == null) return;
  const segments = buildTrailSegments(
    replay.frames,
    currentFrameIndex,
    selectedTrackId,
    TRAIL_FRAMES,
  );
  const selectedPlayer = playerAtFrame(frame, selectedTrackId);
  const catalogTrack = catalog.find((track) => track.track_id === selectedTrackId);
  const trailColor = colorFor(selectedPlayer ?? catalogTrack);
  context.lineCap = "round";
  context.lineJoin = "round";
  for (const segment of segments) {
    if (segment.length < 2) continue;
    context.beginPath();
    segment.forEach((point, index) => {
      const canvasPoint = pitchPoint(rect, point.x_cm, point.y_cm);
      if (index === 0) context.moveTo(canvasPoint.x, canvasPoint.y);
      else context.lineTo(canvasPoint.x, canvasPoint.y);
    });
    context.strokeStyle = trailColor;
    context.globalAlpha = 0.72;
    context.lineWidth = Math.max(3, canvas.width / 320);
    context.stroke();
    context.globalAlpha = 1;
  }
}

function drawDiamond(point, radius) {
  context.beginPath();
  context.moveTo(point.x, point.y - radius);
  context.lineTo(point.x + radius, point.y);
  context.lineTo(point.x, point.y + radius);
  context.lineTo(point.x - radius, point.y);
  context.closePath();
  context.fill();
  context.stroke();
}

function drawLabel(text, x, y) {
  context.lineWidth = Math.max(3, canvas.width / 420);
  context.strokeStyle = "rgba(6, 12, 16, 0.85)";
  context.strokeText(text, x, y);
  context.fillStyle = "#f7fbf5";
  context.fillText(text, x, y);
}

function drawPlayers(rect, frame) {
  const radius = Math.max(6, canvas.width / 105);
  for (const player of frame.players) {
    if (!hasPosition(player)) continue;
    const point = pitchPoint(rect, player.x_cm, player.y_cm);
    const isSelected = player.track_id === selectedTrackId;
    const markerRadius = isSelected ? radius * 1.22 : radius;
    context.globalAlpha = selectedTrackId == null || isSelected ? 1 : 0.55;
    context.fillStyle = colorFor(player);
    context.strokeStyle = "#06100d";
    context.lineWidth = Math.max(2, canvas.width / 500);
    const label = entityLabel(player);
    if (label === "GK") {
      drawDiamond(point, markerRadius);
    } else if (label === "referee") {
      const size = markerRadius * 0.95;
      context.fillRect(point.x - size, point.y - size, size * 2, size * 2);
      context.strokeRect(point.x - size, point.y - size, size * 2, size * 2);
    } else {
      context.beginPath();
      context.arc(point.x, point.y, markerRadius, 0, Math.PI * 2);
      context.fill();
      context.stroke();
    }
    if (isSelected) {
      context.strokeStyle = "#f8ffdf";
      context.lineWidth = Math.max(2, canvas.width / 380);
      context.beginPath();
      context.arc(point.x, point.y, radius * 1.75, 0, Math.PI * 2);
      context.stroke();
    }
    context.font = `700 ${Math.max(11, canvas.width / 78)}px system-ui, sans-serif`;
    context.textBaseline = "alphabetic";
    drawLabel(markerText(player), point.x + markerRadius + 5, point.y - markerRadius);
    const suggestion = isSelected ? formatSuggestedName(player.suggested_name) : null;
    if (suggestion) {
      context.font = `650 ${Math.max(11, canvas.width / 86)}px system-ui, sans-serif`;
      drawLabel(suggestion, point.x + markerRadius + 5, point.y + 4);
    }
  }
  context.globalAlpha = 1;
}

function drawBall(rect, frame) {
  const position = ballPosition(frame);
  if (!position) return;
  const point = pitchPoint(rect, position.x_cm, position.y_cm);
  const radius = Math.max(5, canvas.width / 150);
  context.save();
  context.globalAlpha = 1;
  context.fillStyle = "#f4f1e8";
  context.strokeStyle = "#1a140c";
  context.lineWidth = Math.max(2, canvas.width / 480);
  context.beginPath();
  context.arc(point.x, point.y, radius, 0, Math.PI * 2);
  context.fill();
  context.stroke();
  context.fillStyle = "rgba(255, 255, 255, 0.9)";
  context.beginPath();
  context.arc(point.x - radius * 0.28, point.y - radius * 0.3, radius * 0.28, 0, Math.PI * 2);
  context.fill();
  context.font = `700 ${Math.max(10, canvas.width / 90)}px system-ui, sans-serif`;
  drawLabel("Ball", point.x + radius + 4, point.y - radius);
  context.restore();
}

function resizeCanvas() {
  const bounds = canvas.getBoundingClientRect();
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round(bounds.width * ratio));
  const height = Math.max(1, Math.round(bounds.height * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
}

function renderAtTime(seconds) {
  currentFrameIndex = frameIndexForTime(seconds, fps, replay.frames.length);
  renderCurrentFrame();
}

function renderCurrentFrame() {
  resizeCanvas();
  const frame = replay.frames[currentFrameIndex];
  const rect = pitchRect(canvas.width, canvas.height);
  lastPitchRect = rect;
  drawPitch(rect);
  drawTrail(rect, frame);
  drawPlayers(rect, frame);
  drawBall(rect, frame);

  const time = currentFrameIndex / fps;
  timeLabel.textContent = formatTimestamp(time);
  tagTime.textContent = formatTimestamp(video.currentTime || time);
  frameLabel.textContent = `Frame ${frame.frame} · Segment ${frame.segment}`;
  cutBadge.hidden = !frame.camera_cut;
  pitchEmpty.hidden = frame.calibration_trusted;
  pitchEmpty.textContent = frame.calibration_trusted
    ? ""
    : `Calibration unavailable${frame.calibration_reason ? ` · ${frame.calibration_reason}` : ""}`;
  timelineProgress.style.width = `${Math.min(100, (time / (replay.frames.length / fps)) * 100)}%`;

  const catalogTrack =
    selectedTrackId == null
      ? null
      : catalog.find((track) => track.track_id === selectedTrackId) ?? null;
  followStatus.textContent = followStatusText(selectedTrackId, frame, catalogTrack);
}

function schedulePlaybackLoop() {
  const generation = ++playbackGeneration;
  const tick = (_now, metadata) => {
    if (generation !== playbackGeneration) return;
    renderAtTime(metadata?.mediaTime ?? video.currentTime);
    if (video.paused || video.ended) return;
    if ("requestVideoFrameCallback" in video) video.requestVideoFrameCallback(tick);
    else requestAnimationFrame((now) => tick(now, null));
  };
  tick(performance.now(), {mediaTime: video.currentTime});
}

function renderNotes() {
  tagMarkers.replaceChildren();
  tagList.replaceChildren();
  const duration = replay.frames.length / fps;
  for (const note of notes) {
    const marker = document.createElement("button");
    marker.className = "tag-marker";
    marker.type = "button";
    marker.style.left = `${Math.min(100, (note.time / duration) * 100)}%`;
    marker.title = `${formatTimestamp(note.time)} · ${note.note}`;
    marker.addEventListener("click", () => openNote(note));
    tagMarkers.append(marker);

    const chip = document.createElement("div");
    chip.className = "tag-chip";
    chip.tabIndex = 0;
    chip.setAttribute("role", "button");
    const timestamp = document.createElement("strong");
    timestamp.textContent = formatTimestamp(note.time);
    const text = document.createElement("span");
    const player = note.trackId == null ? "" : `T${note.trackId} · `;
    text.textContent = `${player}${note.note}`;
    const remove = document.createElement("button");
    remove.className = "delete-tag";
    remove.type = "button";
    remove.textContent = "×";
    remove.title = "Delete tag";
    remove.addEventListener("click", (event) => {
      event.stopPropagation();
      notes = notes.filter((item) => item.id !== note.id);
      saveNotes();
      renderNotes();
    });
    chip.append(timestamp, text, remove);
    chip.addEventListener("click", () => openNote(note));
    chip.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") openNote(note);
    });
    tagList.append(chip);
  }
}

function openNote(note) {
  if (note.trackId != null) selectTrack(note.trackId);
  video.currentTime = noteJumpTime(note.time);
  renderAtTime(video.currentTime);
}

playerSelect.addEventListener("change", () => {
  selectTrack(playerSelect.value ? Number(playerSelect.value) : null);
});
clearSelection.addEventListener("click", () => selectTrack(null));

canvas.addEventListener("click", (event) => {
  if (!lastPitchRect) return;
  const bounds = canvas.getBoundingClientRect();
  const scaleX = canvas.width / bounds.width;
  const scaleY = canvas.height / bounds.height;
  const canvasX = (event.clientX - bounds.left) * scaleX;
  const canvasY = (event.clientY - bounds.top) * scaleY;
  const xCm = (canvasX - lastPitchRect.x) / lastPitchRect.scale;
  const yCm = (canvasY - lastPitchRect.y) / lastPitchRect.scale;
  if (xCm < 0 || xCm > pitchLength || yCm < 0 || yCm > pitchWidth) return;
  const players = replay.frames[currentFrameIndex].players.filter(hasPosition);
  const nearest = findNearestPlayer(players, xCm, yCm, 600);
  if (nearest) selectTrack(nearest.track_id);
});

tagForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const note = tagNote.value.trim();
  if (!note) return;
  notes.push({
    id: globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`,
    time: Math.min(replay.frames.length / fps, Math.max(0, video.currentTime)),
    trackId: tagPlayer.value ? Number(tagPlayer.value) : null,
    note,
  });
  notes.sort((first, second) => first.time - second.time);
  saveNotes();
  tagNote.value = "";
  renderNotes();
});

video.addEventListener("error", () => {
  videoNote.hidden = false;
});
video.addEventListener("play", schedulePlaybackLoop);
video.addEventListener("pause", () => {
  playbackGeneration += 1;
  renderAtTime(video.currentTime);
});
for (const eventName of ["loadedmetadata", "seeking", "seeked", "timeupdate", "ended"]) {
  video.addEventListener(eventName, () => renderAtTime(video.currentTime));
}

new ResizeObserver(() => renderCurrentFrame()).observe(canvas);
populateTrackSelectors();
renderNotes();

const initialParameters = new URLSearchParams(window.location.search);
const initialTrackId = Number(initialParameters.get("track"));
if (
  Number.isInteger(initialTrackId) &&
  catalog.some((track) => track.track_id === initialTrackId)
) {
  selectTrack(initialTrackId);
} else {
  renderCurrentFrame();
}

const initialTime = Number(initialParameters.get("time"));
if (Number.isFinite(initialTime) && initialTime >= 0) {
  const seekToInitialTime = () => {
    video.currentTime = Math.min(replay.frames.length / fps, initialTime);
    renderAtTime(video.currentTime);
  };
  if (video.readyState >= 1) seekToInitialTime();
  else video.addEventListener("loadedmetadata", seekToInitialTime, {once: true});
}
