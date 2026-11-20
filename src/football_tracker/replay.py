"""Stream browser-friendly per-frame replay data beside the rendered video."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import TextIO

import supervision as sv

from football_tracker.classification import EntityAssignment, role_from_class_name
from football_tracker.identity import PlayerSuggestion
from football_tracker.pitch import PitchCalibration, SoccerPitch
from football_tracker.tactical import project_visible_entities


class ReplayDataExporter:
    """Write compact replay frames incrementally and publish the JSON atomically."""

    def __init__(
        self,
        path: Path,
        *,
        fps: float,
        frame_count: int | None,
        pitch: SoccerPitch,
        source_video: str,
    ) -> None:
        self.path = path
        self.fps = fps
        self.frame_count = frame_count
        self.pitch = pitch
        self.source_video = source_video
        self._handle: TextIO | None = None
        self._temporary_path: Path | None = None
        self._frames_written = 0

    def __enter__(self) -> ReplayDataExporter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self.path.parent,
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            delete=False,
        )
        self._handle = handle
        self._temporary_path = Path(handle.name)
        metadata = {
            "schema_version": 1,
            "video": {
                "source": self.source_video,
                "fps": self.fps,
                "frame_count": self.frame_count,
            },
            "pitch": {
                "length_cm": self.pitch.length,
                "width_cm": self.pitch.width,
            },
        }
        prefix = json.dumps(metadata, separators=(",", ":"))
        handle.write(prefix[:-1])
        handle.write(',"frames":[')
        return self

    def write_frame(
        self,
        *,
        frame_index: int,
        segment: int,
        camera_cut: bool,
        calibration: PitchCalibration,
        detections: sv.Detections,
        assignments: list[EntityAssignment] | None,
        suggestions: list[PlayerSuggestion | None] | None,
        class_name: Callable[[int], str],
    ) -> None:
        """Append one frame while keeping uncertain identity output explicitly anonymous."""
        if self._handle is None:
            raise RuntimeError("Replay data exporter is not open.")
        if frame_index != self._frames_written:
            raise ValueError("Replay frames must be written once in sequential order.")

        projected = {
            entity.detection_index: entity
            for entity in project_visible_entities(
                pitch=self.pitch,
                calibration=calibration,
                detections=detections,
            )
        }
        tracker_ids = detections.tracker_id
        players = []
        for index, track_id_value in enumerate(tracker_ids if tracker_ids is not None else ()):
            entity = projected.get(index)
            assignment = assignments[index] if assignments is not None else None
            if assignment is not None:
                role = assignment.role
                team = assignment.label if assignment.team in (0, 1) else None
                label = assignment.label
            else:
                class_id = (
                    int(detections.class_id[index]) if detections.class_id is not None else -1
                )
                role = role_from_class_name(class_name(class_id)) if class_id >= 0 else "unknown"
                team = None
                label = role
            suggestion = suggestions[index] if suggestions is not None else None
            players.append(
                {
                    "track_id": int(track_id_value),
                    "x_cm": round(entity.x, 2) if entity is not None else None,
                    "y_cm": round(entity.y, 2) if entity is not None else None,
                    "role": role,
                    "team": team,
                    "label": label,
                    # OCR roster matches remain suggestions, never verified identities.
                    "suggested_name": suggestion.name if suggestion is not None else None,
                }
            )

        record = {
            "frame": frame_index,
            "segment": segment,
            "camera_cut": camera_cut,
            "calibration_trusted": calibration.trusted,
            "calibration_reason": calibration.reason,
            "players": players,
        }
        if self._frames_written:
            self._handle.write(",")
        json.dump(record, self._handle, separators=(",", ":"))
        self._frames_written += 1

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._handle is None or self._temporary_path is None:
            return
        try:
            if exc_type is None:
                self._handle.write("]}")
                self._handle.flush()
                os.fsync(self._handle.fileno())
            self._handle.close()
            if exc_type is None:
                self._temporary_path.replace(self.path)
            else:
                self._temporary_path.unlink(missing_ok=True)
        finally:
            self._handle = None
            self._temporary_path = None
