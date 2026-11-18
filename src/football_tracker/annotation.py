"""OpenCV annotations and short image-space movement trails."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable

import cv2
import numpy as np
import supervision as sv

from football_tracker.classification import EntityAssignment

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
TEAM_COLORS_BGR = ((230, 125, 55), (70, 80, 245))
UNKNOWN_COLOR_BGR = (150, 150, 150)
REFEREE_COLOR_BGR = (40, 220, 245)
GOALKEEPER_COLOR_BGR = (210, 85, 210)


def track_color(track_id: int) -> tuple[int, int, int]:
    """Return a deterministic fallback colour when team assignments are disabled."""
    return TRACK_COLORS_BGR[(track_id - 1) % len(TRACK_COLORS_BGR)]


class TrailStore:
    def __init__(self, length: int = 30, stale_after_frames: int = 150) -> None:
        self.length = length
        self.stale_after_frames = stale_after_frames
        self._points: dict[int, deque[tuple[int, int]]] = {}
        self._last_seen: dict[int, int] = {}
        self._colors: dict[int, tuple[int, int, int]] = {}
        self._frame_index = -1

    def update(
        self,
        detections: sv.Detections,
        colors: list[tuple[int, int, int]] | None = None,
    ) -> None:
        self._frame_index += 1
        if detections.tracker_id is not None:
            for index, (box, track_id_value) in enumerate(
                zip(detections.xyxy, detections.tracker_id, strict=True)
            ):
                track_id = int(track_id_value)
                foot = (round((box[0] + box[2]) / 2), round(box[3]))
                self._points.setdefault(track_id, deque(maxlen=self.length)).append(foot)
                self._last_seen[track_id] = self._frame_index
                if colors is not None:
                    self._colors[track_id] = colors[index]

        stale_ids = [
            track_id
            for track_id, last_seen in self._last_seen.items()
            if self._frame_index - last_seen > self.stale_after_frames
        ]
        for track_id in stale_ids:
            self._points.pop(track_id, None)
            self._last_seen.pop(track_id, None)
            self._colors.pop(track_id, None)

    def draw(self, frame: np.ndarray) -> None:
        for track_id, points in self._points.items():
            if len(points) < 2:
                continue
            cv2.polylines(
                frame,
                [np.asarray(points, dtype=np.int32)],
                isClosed=False,
                color=self._colors.get(track_id, track_color(track_id)),
                thickness=2,
                lineType=cv2.LINE_AA,
            )

    def reset(self) -> None:
        self._points.clear()
        self._last_seen.clear()
        self._colors.clear()


def assignment_color(assignment: EntityAssignment) -> tuple[int, int, int]:
    if assignment.role == "referee":
        return REFEREE_COLOR_BGR
    if assignment.role == "goalkeeper":
        return GOALKEEPER_COLOR_BGR
    if assignment.team in (0, 1):
        return TEAM_COLORS_BGR[assignment.team]
    return UNKNOWN_COLOR_BGR


Rect = tuple[int, int, int, int]


def _intersection_area(first: Rect, second: Rect) -> int:
    width = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    return width * height


def place_label(
    box: Rect,
    label_size: tuple[int, int],
    frame_size: tuple[int, int],
    occupied: list[Rect],
) -> Rect:
    """Choose a nearby label rectangle with the least existing overlap."""
    x1, y1, x2, y2 = box
    label_width, label_height = label_size
    frame_width, frame_height = frame_size
    raw_candidates = (
        (x1, y1 - label_height - 4),
        (x2 + 4, y1),
        (x1 - label_width - 4, y1),
        (x1, y2 + 4),
        (x2 - label_width, y1 - label_height - 4),
    )
    candidates: list[Rect] = []
    for candidate_x, candidate_y in raw_candidates:
        left = int(np.clip(candidate_x, 0, max(0, frame_width - label_width)))
        top = int(np.clip(candidate_y, 0, max(0, frame_height - label_height)))
        candidates.append((left, top, left + label_width, top + label_height))

    def score(candidate: Rect) -> tuple[int, int]:
        overlap = sum(_intersection_area(candidate, used) for used in occupied)
        distance = abs(candidate[0] - x1) + abs(candidate[1] - y1)
        return overlap, distance

    return min(candidates, key=score)


def annotate_frame(
    frame: np.ndarray,
    detections: sv.Detections,
    *,
    class_name: Callable[[int], str],
    trails: TrailStore,
    assignments: list[EntityAssignment] | None,
    segment: int,
    camera_cut: bool,
    scene_score: float,
) -> np.ndarray:
    """Draw trails, foot markers, bounding boxes, and non-identity track labels."""
    annotated = frame.copy()
    colors = (
        [assignment_color(assignment) for assignment in assignments]
        if assignments is not None
        else None
    )
    trails.update(detections, colors)
    trails.draw(annotated)

    tracker_ids = detections.tracker_id
    if tracker_ids is not None:
        class_ids = detections.class_id
        occupied: list[Rect] = [(0, 0, min(520, annotated.shape[1]), 64)]
        for index, (box, track_id_value) in enumerate(
            zip(detections.xyxy, tracker_ids, strict=True)
        ):
            track_id = int(track_id_value)
            x1, y1, x2, y2 = (round(value) for value in box)
            color = colors[index] if colors is not None else track_color(track_id)
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
            if assignments is not None:
                object_name = assignments[index].label
            else:
                object_name = (
                    class_name(int(class_ids[index])) if class_ids is not None else "person"
                )
            label = f"T{track_id} {object_name}"
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1
            )
            label_width = text_width + 8
            label_height = text_height + baseline + 6
            label_left, label_top, label_right, label_bottom = place_label(
                (x1, y1, x2, y2),
                (label_width, label_height),
                (annotated.shape[1], annotated.shape[0]),
                occupied,
            )
            occupied.append((label_left, label_top, label_right, label_bottom))
            cv2.rectangle(
                annotated,
                (label_left, label_top),
                (label_right, label_bottom),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (label_left + 4, label_top + text_height + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
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

