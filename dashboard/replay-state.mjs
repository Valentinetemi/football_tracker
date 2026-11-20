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
        suggested_name: null,
        first_frame: frame.frame,
        last_frame: frame.frame,
      };
      existing.team = existing.team ?? player.team;
      existing.role = betterLabel(existing.role, player.role);
      existing.label = betterLabel(existing.label, player.label);
      const suggestion = formatSuggestedName(player.suggested_name);
      if (suggestion && !existing.suggested_name) {
        existing.suggested_name = suggestion.endsWith("?")
          ? suggestion.slice(0, -1)
          : suggestion;
      }
      existing.last_frame = frame.frame;
      tracks.set(player.track_id, existing);
    }
  }
  return [...tracks.values()].sort((first, second) => first.track_id - second.track_id);
}

export function formatSuggestedName(name) {
  if (typeof name !== "string") return null;
  const trimmed = name.trim();
  if (!trimmed) return null;
  return trimmed.endsWith("?") ? trimmed : `${trimmed}?`;
}

export function entityLabel(entity) {
  if (!entity) return "unknown";
  const role = entity.role ?? "";
  const label = entity.label ?? "";
  if (role === "goalkeeper" || label === "goalkeeper/unknown" || label === "goalkeeper") {
    return "GK";
  }
  if (entity.team === "team-a" || entity.team === "team-b") return entity.team;
  if (role === "referee" || label === "referee") return "referee";
  if (label && label !== "unknown") return label;
  if (role && role !== "unknown") return role;
  return "unknown";
}

export function trackMenuLabel(track) {
  const suggestion = formatSuggestedName(track.suggested_name);
  const base = `T${track.track_id} · ${entityLabel(track)}`;
  return suggestion ? `${base} · ${suggestion}` : base;
}

export function followStatusText(selectedTrackId, frame, catalogTrack = null) {
  if (selectedTrackId == null) {
    return "Select a player on the pitch or from the list.";
  }
  const selectedPlayer = playerAtFrame(frame, selectedTrackId);
  const source = selectedPlayer ?? catalogTrack;
  const label = source ? entityLabel(source) : null;
  const suggestion = formatSuggestedName(
    selectedPlayer ? selectedPlayer.suggested_name : catalogTrack?.suggested_name,
  );
  const details = [label, suggestion].filter(Boolean).join(" · ");
  const prefix = details ? `T${selectedTrackId} · ${details}` : `T${selectedTrackId}`;
  const visible =
    selectedPlayer &&
    Number.isFinite(selectedPlayer.x_cm) &&
    Number.isFinite(selectedPlayer.y_cm);
  if (!visible) return `${prefix} · Player not visible.`;
  return `Following ${prefix}`;
}

export function ballPosition(frame) {
  const ball = frame?.ball;
  if (!frame?.calibration_trusted || !ball || ball.visible !== true) return null;
  if (ball.reliable === false) return null;
  if (!Number.isFinite(ball.x_cm) || !Number.isFinite(ball.y_cm)) return null;
  return {x_cm: ball.x_cm, y_cm: ball.y_cm};
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
