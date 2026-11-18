"""OpenCV annotations and short image-space movement trails."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable

import cv2
import numpy as np
import supervision as sv

TRACK_COLORS_BGR = (
    (90, 220, 255),
    (255, 160, 70),
    (90, 230, 120),
    (230, 120, 230),
    (80, 170, 255),
    (255, 210, 100),
    (180, 120, 255),
    (120, 240, 220),
)


def track_color(track_id: int) -> tuple[int, int, int]:
    """Return a deterministic display colour; colours do not mean teams yet."""
    return TRACK_COLORS_BGR[(track_id - 1) % len(TRACK_COLORS_BGR)]


class TrailStore:
    def __init__(self, length: int = 30, stale_after_frames: int = 150) -> None:
        self.length = length
        self.stale_after_frames = stale_after_frames
        self._points: dict[int, deque[tuple[int, int]]] = {}
        self._last_seen: dict[int, int] = {}
        self._frame_index = -1

    def update(self, detections: sv.Detections) -> None:
        self._frame_index += 1
        if detections.tracker_id is not None:
            for box, track_id_value in zip(
                detections.xyxy,
                detections.tracker_id,
                strict=True,
            ):
                track_id = int(track_id_value)
                foot = (round((box[0] + box[2]) / 2), round(box[3]))
                self._points.setdefault(track_id, deque(maxlen=self.length)).append(foot)
                self._last_seen[track_id] = self._frame_index

        stale_ids = [
            track_id
            for track_id, last_seen in self._last_seen.items()
            if self._frame_index - last_seen > self.stale_after_frames
        ]
        for track_id in stale_ids:
            self._points.pop(track_id, None)
            self._last_seen.pop(track_id, None)

    def draw(self, frame: np.ndarray) -> None:
        for track_id, points in self._points.items():
            if len(points) < 2:
                continue
            cv2.polylines(
                frame,
                [np.asarray(points, dtype=np.int32)],
                isClosed=False,
                color=track_color(track_id),
                thickness=2,
                lineType=cv2.LINE_AA,
            )

    def reset(self) -> None:
        self._points.clear()
        self._last_seen.clear()


def annotate_frame(
    frame: np.ndarray,
    detections: sv.Detections,
    *,
    class_name: Callable[[int], str],
    trails: TrailStore,
    segment: int,
    camera_cut: bool,
    scene_score: float,
) -> np.ndarray:
    """Draw trails, foot markers, bounding boxes, and non-identity track labels."""
    annotated = frame.copy()
    trails.update(detections)
    trails.draw(annotated)

    tracker_ids = detections.tracker_id
    if tracker_ids is not None:
        confidences = detections.confidence
        class_ids = detections.class_id
        for index, (box, track_id_value) in enumerate(
            zip(detections.xyxy, tracker_ids, strict=True)
        ):
            track_id = int(track_id_value)
            x1, y1, x2, y2 = (round(value) for value in box)
            color = track_color(track_id)
            foot = (round((x1 + x2) / 2), y2)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            cv2.ellipse(
                annotated,
                foot,
                (max(6, (x2 - x1) // 2), 5),
                0,
                0,
                360,
                color,
                2,
                cv2.LINE_AA,
            )
            object_name = class_name(int(class_ids[index])) if class_ids is not None else "person"
            confidence = float(confidences[index]) if confidences is not None else 0.0
            label = f"T{track_id} {object_name} {confidence:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            label_top = max(0, y1 - text_height - baseline - 6)
            cv2.rectangle(
                annotated,
                (x1, label_top),
                (x1 + text_width + 8, label_top + text_height + baseline + 6),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 4, label_top + text_height + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )

    status = f"segment {segment} | visible tracks {len(detections)}"
    cv2.putText(
        annotated,
        status,
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if camera_cut:
        cv2.putText(
            annotated,
            f"camera cut: tracker reset ({scene_score:.2f})",
            (14, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (80, 220, 255),
            2,
            cv2.LINE_AA,
        )
    return annotated

