"""ByteTrack lifecycle and globally unique display IDs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np
import supervision as sv


class Tracker(Protocol):
    def update_with_detections(self, detections: sv.Detections) -> sv.Detections: ...

    def reset(self) -> None: ...


class TrackManager:
    """Own ByteTrack state and isolate track identities across camera cuts."""

    def __init__(
        self,
        *,
        frame_rate: float,
        track_activation_threshold: float,
        lost_track_buffer: int,
        minimum_consecutive_frames: int,
        tracker_factory: Callable[[], Tracker] | None = None,
    ) -> None:
        if tracker_factory is None:
            def tracker_factory() -> sv.ByteTrack:
                return sv.ByteTrack(
                    track_activation_threshold=track_activation_threshold,
                    lost_track_buffer=lost_track_buffer,
                    frame_rate=max(1, round(frame_rate)),
                    minimum_consecutive_frames=minimum_consecutive_frames,
                )

        self._tracker = tracker_factory()
        self._local_to_display: dict[int, int] = {}
        self._next_display_id = 1
        self.segment = 1
        self.reset_count = 0

    def update(self, detections: sv.Detections) -> sv.Detections:
        """Track detections and replace local IDs with run-global display IDs."""
        tracked = self._tracker.update_with_detections(detections)
        if tracked.tracker_id is None:
            return tracked

        display_ids: list[int] = []
        for local_id_value in tracked.tracker_id:
            local_id = int(local_id_value)
            if local_id not in self._local_to_display:
                self._local_to_display[local_id] = self._next_display_id
                self._next_display_id += 1
            display_ids.append(self._local_to_display[local_id])

        tracked.tracker_id = np.asarray(display_ids, dtype=int)
        tracked.data["track_segment"] = np.full(len(tracked), self.segment, dtype=int)
        return tracked

    def reset_for_camera_cut(self) -> None:
        """Discard motion history while ensuring new tracks cannot reuse display IDs."""
        self._tracker.reset()
        self._local_to_display.clear()
        self.segment += 1
        self.reset_count += 1

    @property
    def display_track_count(self) -> int:
        return self._next_display_id - 1

