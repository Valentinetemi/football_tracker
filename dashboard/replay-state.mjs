export function frameIndexForTime(currentTime, fps, frameCount) {
  if (!Number.isFinite(currentTime) || !Number.isFinite(fps) || fps <= 0 || frameCount <= 0) {
    return 0;
  }
  return Math.min(frameCount - 1, Math.max(0, Math.floor(currentTime * fps + 1e-6)));
}

function betterLabel(current, candidate) {
  if (!current) return candidate;
  if (current === "unknown" && candidate !== "unknown") return candidate;
  return current;
}

export function buildTrackCatalog(frames) {
  const tracks = new Map();
  for (const frame of frames) {
    for (const player of frame.players) {
      const existing = tracks.get(player.track_id) ?? {
        track_id: player.track_id,
        team: null,
        role: player.role,
        label: player.label,
        first_frame: frame.frame,
        last_frame: frame.frame,
      };
      existing.team = existing.team ?? player.team;
      existing.role = betterLabel(existing.role, player.role);
      existing.label = betterLabel(existing.label, player.label);
      existing.last_frame = frame.frame;
      tracks.set(player.track_id, existing);
    }
  }
  return [...tracks.values()].sort((first, second) => first.track_id - second.track_id);
}

export function playerAtFrame(frame, trackId) {
  if (!frame || trackId == null) return null;
  return frame.players.find((player) => player.track_id === trackId) ?? null;
}

export function buildTrailSegments(frames, currentIndex, trackId, maximumFrames = 75) {
  if (trackId == null || currentIndex < 0 || currentIndex >= frames.length) return [];
  const currentFrame = frames[currentIndex];
  const currentSegment = currentFrame.segment;
  const start = Math.max(0, currentIndex - maximumFrames + 1);
  const segments = [];
  let points = [];
  let previousFrameNumber = null;

  const finishSegment = () => {
    if (points.length) segments.push(points);
    points = [];
    previousFrameNumber = null;
  };

  for (let index = start; index <= currentIndex; index += 1) {
    const frame = frames[index];
    if (frame.segment !== currentSegment || frame.camera_cut) {
      finishSegment();
      continue;
    }
    const player = playerAtFrame(frame, trackId);
    if (!player || !Number.isFinite(player.x_cm) || !Number.isFinite(player.y_cm)) {
      finishSegment();
      continue;
    }
    if (previousFrameNumber != null && frame.frame !== previousFrameNumber + 1) {
      finishSegment();
    }
    points.push({frame: frame.frame, x_cm: player.x_cm, y_cm: player.y_cm});
    previousFrameNumber = frame.frame;
  }
  finishSegment();
  return segments;
}

export function findNearestPlayer(players, xCm, yCm, maximumDistanceCm) {
  let nearest = null;
  let nearestDistance = maximumDistanceCm;
  for (const player of players) {
    if (!Number.isFinite(player.x_cm) || !Number.isFinite(player.y_cm)) continue;
    const distance = Math.hypot(player.x_cm - xCm, player.y_cm - yCm);
    if (distance <= nearestDistance) {
      nearest = player;
      nearestDistance = distance;
    }
  }
  return nearest;
}

export function formatTimestamp(seconds) {
  const safeSeconds = Math.max(0, Number.isFinite(seconds) ? seconds : 0);
  const minutes = Math.floor(safeSeconds / 60);
  const remainder = safeSeconds - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${remainder.toFixed(2).padStart(5, "0")}`;
}

export function noteJumpTime(timestamp, leadSeconds = 2) {
  return Math.max(0, timestamp - leadSeconds);
}
