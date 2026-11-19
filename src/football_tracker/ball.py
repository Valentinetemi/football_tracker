"""Conservative ball candidate filtering and short-term motion association."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import supervision as sv

BALL_CLASS_NAMES = {"ball", "sports ball", "football", "soccer ball"}


def is_ball_class_name(name: str) -> bool:
    return name.strip().lower() in BALL_CLASS_NAMES


def split_ball_detections(
    detections: sv.Detections,
    *,
    class_name: Callable[[int], str],
    people_confidence: float,
    ball_confidence: float,
) -> tuple[sv.Detections, sv.Detections]:
    """Split ball candidates before people reach ByteTrack or shirt classification."""
    if detections.class_id is None:
        return detections, sv.Detections.empty()
    confidence = (
        detections.confidence
        if detections.confidence is not None
        else np.ones(len(detections), dtype=np.float32)
    )
    ball_mask = np.asarray(
        [is_ball_class_name(class_name(int(class_id))) for class_id in detections.class_id],
        dtype=bool,
    )
    people_mask = ~ball_mask & (confidence >= people_confidence)
    candidate_mask = ball_mask & (confidence >= ball_confidence)
    return detections[people_mask], detections[candidate_mask]


@dataclass(frozen=True, slots=True)
class BallObservation:
    """A current-frame measured ball position; never an extrapolated marker."""

    center: tuple[float, float]
    box: tuple[float, float, float, float]
    confidence: float


@dataclass(frozen=True, slots=True)
class BallTrackingSummary:
    candidate_frames: int
    reliable_frames: int
    missed_frames: int
    ambiguous_frames: int
    recoveries: int
    camera_cut_resets: int


class BallTracker:
    """Associate a unique ball candidate while hiding misses and ambiguous frames."""

    def __init__(
        self,
        *,
        minimum_hits: int = 2,
        maximum_gap: int = 8,
        maximum_distance_ratio: float = 0.10,
        ambiguity_margin: float = 0.25,
    ) -> None:
        self.minimum_hits = minimum_hits
        self.maximum_gap = maximum_gap
        self.maximum_distance_ratio = maximum_distance_ratio
        self.ambiguity_margin = ambiguity_margin
        self._last_center: np.ndarray | None = None
        self._velocity = np.zeros(2, dtype=np.float64)
        self._last_seen_frame: int | None = None
        self._consecutive_hits = 0
        self._ever_reliable_in_segment = False
        self._pending_recovery = False
        self._candidate_frames = 0
        self._reliable_frames = 0
        self._missed_frames = 0
        self._ambiguous_frames = 0
        self._recoveries = 0
        self._camera_cut_resets = 0

    @staticmethod
    def _candidate_values(
        detections: sv.Detections,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        boxes = np.asarray(detections.xyxy, dtype=np.float64)
        centers = np.column_stack(
            ((boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2)
        )
        confidence = (
            np.asarray(detections.confidence, dtype=np.float64)
            if detections.confidence is not None
            else np.ones(len(detections), dtype=np.float64)
        )
        return boxes, centers, confidence

    def _clear_motion(self) -> None:
        self._last_center = None
        self._velocity[:] = 0
        self._last_seen_frame = None
        self._consecutive_hits = 0
        self._pending_recovery = False

    def _accept(
        self,
        *,
        frame_index: int,
        box: np.ndarray,
        center: np.ndarray,
        confidence: float,
    ) -> BallObservation | None:
        gap = (
            frame_index - self._last_seen_frame
            if self._last_seen_frame is not None
            else None
        )
        if self._last_center is not None and gap is not None and gap > 0:
            measured_velocity = (center - self._last_center) / gap
            self._velocity = 0.6 * self._velocity + 0.4 * measured_velocity
        else:
            self._velocity[:] = 0

        if gap == 1:
            self._consecutive_hits += 1
        else:
            if gap is not None and self._ever_reliable_in_segment:
                self._pending_recovery = True
            self._consecutive_hits = 1
        self._last_center = center
        self._last_seen_frame = frame_index

        if self._consecutive_hits < self.minimum_hits:
            return None
        if self._pending_recovery:
            self._recoveries += 1
            self._pending_recovery = False
        self._ever_reliable_in_segment = True
        self._reliable_frames += 1
        return BallObservation(
            (float(center[0]), float(center[1])),
            tuple(float(value) for value in box),
            float(confidence),
        )

    def update(
        self,
        *,
        frame_index: int,
        candidates: sv.Detections,
        frame_size: tuple[int, int],
    ) -> BallObservation | None:
        """Return a marker only for a unique, current, motion-consistent detection."""
        if len(candidates) == 0:
            if self._last_seen_frame is not None:
                self._missed_frames += 1
                self._consecutive_hits = 0
                if frame_index - self._last_seen_frame > self.maximum_gap:
                    self._clear_motion()
            return None

        self._candidate_frames += 1
        boxes, centers, confidence = self._candidate_values(candidates)
        if self._last_center is None or self._last_seen_frame is None:
            if len(candidates) != 1:
                self._ambiguous_frames += 1
                return None
            return self._accept(
                frame_index=frame_index,
                box=boxes[0],
                center=centers[0],
                confidence=float(confidence[0]),
            )

        gap = frame_index - self._last_seen_frame
        if gap > self.maximum_gap:
            self._clear_motion()
            if len(candidates) != 1:
                self._ambiguous_frames += 1
                return None
            return self._accept(
                frame_index=frame_index,
                box=boxes[0],
                center=centers[0],
                confidence=float(confidence[0]),
            )

        frame_width, frame_height = frame_size
        base_gate = np.hypot(frame_width, frame_height) * self.maximum_distance_ratio
        gate = base_gate * (1.0 + 0.35 * max(0, gap - 1))
        predicted = self._last_center + self._velocity * gap
        distances = np.linalg.norm(centers - predicted, axis=1)
        eligible = np.flatnonzero(distances <= gate)
        if len(eligible) == 0:
            self._ambiguous_frames += 1
            self._consecutive_hits = 0
            return None

        ranked = eligible[np.argsort(distances[eligible])]
        if len(ranked) > 1:
            first, second = int(ranked[0]), int(ranked[1])
            distance_separation = distances[second] - distances[first]
            confidence_separation = confidence[first] - confidence[second]
            if (
                distance_separation < gate * self.ambiguity_margin
                and confidence_separation < 0.20
            ):
                self._ambiguous_frames += 1
                self._consecutive_hits = 0
                return None

        best = int(ranked[0])
        return self._accept(
            frame_index=frame_index,
            box=boxes[best],
            center=centers[best],
            confidence=float(confidence[best]),
        )

    def reset_for_camera_cut(self) -> None:
        self._clear_motion()
        self._ever_reliable_in_segment = False
        self._camera_cut_resets += 1

    def summary(self) -> BallTrackingSummary:
        return BallTrackingSummary(
            self._candidate_frames,
            self._reliable_frames,
            self._missed_frames,
            self._ambiguous_frames,
            self._recoveries,
            self._camera_cut_resets,
        )
