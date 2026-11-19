"""Render a synchronized, uncertainty-aware tactical pitch panel."""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np
import supervision as sv

from football_tracker.annotation import (
    GOALKEEPER_COLOR_BGR,
    REFEREE_COLOR_BGR,
    UNKNOWN_COLOR_BGR,
    assignment_color,
    track_color,
)
from football_tracker.classification import EntityAssignment, role_from_class_name
from football_tracker.pitch import PitchCalibration, SoccerPitch

PANEL_BACKGROUND_BGR = (24, 31, 27)
PITCH_BACKGROUND_BGR = (45, 122, 54)
PITCH_LINE_BGR = (235, 239, 235)
DOT_EDGE_BGR = (18, 18, 18)


def resolve_panel_height(source_height: int, requested: int | None) -> int:
    """Choose an even tactical-panel height suitable for video codecs."""
    height = requested if requested is not None else max(240, round(source_height * 0.38))
    return height if height % 2 == 0 else height + 1


def _world_to_panel(
    point: tuple[float, float],
    *,
    origin: tuple[int, int],
    scale: float,
) -> tuple[int, int]:
    return (
        round(origin[0] + point[0] * scale),
        round(origin[1] + point[1] * scale),
    )


def _draw_pitch(
    panel: np.ndarray,
    pitch: SoccerPitch,
) -> tuple[tuple[int, int], float]:
    panel_height, panel_width = panel.shape[:2]
    horizontal_margin = max(12, round(panel_width * 0.025))
    top_margin = 42
    bottom_margin = 18
    available_width = max(1, panel_width - 2 * horizontal_margin)
    available_height = max(1, panel_height - top_margin - bottom_margin)
    scale = min(available_width / pitch.length, available_height / pitch.width)
    pitch_width = round(pitch.length * scale)
    pitch_height = round(pitch.width * scale)
    origin = ((panel_width - pitch_width) // 2, top_margin)
    bottom_right = (origin[0] + pitch_width, origin[1] + pitch_height)

    cv2.rectangle(panel, origin, bottom_right, PITCH_BACKGROUND_BGR, -1)
    line_thickness = max(1, round(scale * 45))
    cv2.rectangle(panel, origin, bottom_right, PITCH_LINE_BGR, line_thickness, cv2.LINE_AA)

    def point(x: float, y: float) -> tuple[int, int]:
        return _world_to_panel((x, y), origin=origin, scale=scale)

    centre_x = pitch.length / 2
    cv2.line(
        panel,
        point(centre_x, 0),
        point(centre_x, pitch.width),
        PITCH_LINE_BGR,
        line_thickness,
        cv2.LINE_AA,
    )
    cv2.circle(
        panel,
        point(centre_x, pitch.width / 2),
        max(2, round(pitch.centre_circle_radius * scale)),
        PITCH_LINE_BGR,
        line_thickness,
        cv2.LINE_AA,
    )
    cv2.circle(panel, point(centre_x, pitch.width / 2), max(1, line_thickness), PITCH_LINE_BGR, -1)

    for right_side in (False, True):
        if right_side:
            goal_line = pitch.length
            penalty_x = pitch.length - pitch.penalty_box_length
            goal_x = pitch.length - pitch.goal_box_length
            penalty_spot_x = pitch.length - pitch.penalty_spot_distance
        else:
            goal_line = 0.0
            penalty_x = pitch.penalty_box_length
            goal_x = pitch.goal_box_length
            penalty_spot_x = pitch.penalty_spot_distance

        penalty_y1 = (pitch.width - pitch.penalty_box_width) / 2
        penalty_y2 = (pitch.width + pitch.penalty_box_width) / 2
        goal_y1 = (pitch.width - pitch.goal_box_width) / 2
        goal_y2 = (pitch.width + pitch.goal_box_width) / 2
        cv2.rectangle(
            panel,
            point(min(goal_line, penalty_x), penalty_y1),
            point(max(goal_line, penalty_x), penalty_y2),
            PITCH_LINE_BGR,
            line_thickness,
            cv2.LINE_AA,
        )
        cv2.rectangle(
            panel,
            point(min(goal_line, goal_x), goal_y1),
            point(max(goal_line, goal_x), goal_y2),
            PITCH_LINE_BGR,
            line_thickness,
            cv2.LINE_AA,
        )
        cv2.circle(
            panel,
            point(penalty_spot_x, pitch.width / 2),
            max(1, line_thickness),
            PITCH_LINE_BGR,
            -1,
        )
    return origin, scale


def _fallback_color_and_role(
    detections: sv.Detections,
    index: int,
    track_id: int,
    class_name: Callable[[int], str],
) -> tuple[tuple[int, int, int], str]:
    if detections.class_id is None:
        return track_color(track_id), "unknown"
    role = role_from_class_name(class_name(int(detections.class_id[index])))
    if role == "referee":
        return REFEREE_COLOR_BGR, role
    if role == "goalkeeper":
        return GOALKEEPER_COLOR_BGR, role
    return UNKNOWN_COLOR_BGR, role


def _draw_entity_marker(
    panel: np.ndarray,
    center: tuple[int, int],
    *,
    color: tuple[int, int, int],
    role: str,
    track_id: int,
    radius: int,
) -> None:
    if role == "referee":
        cv2.rectangle(
            panel,
            (center[0] - radius, center[1] - radius),
            (center[0] + radius, center[1] + radius),
            color,
            -1,
            cv2.LINE_AA,
        )
        cv2.rectangle(
            panel,
            (center[0] - radius, center[1] - radius),
            (center[0] + radius, center[1] + radius),
            DOT_EDGE_BGR,
            2,
            cv2.LINE_AA,
        )
    elif role == "goalkeeper":
        diamond = np.asarray(
            [
                (center[0], center[1] - radius - 1),
                (center[0] + radius + 1, center[1]),
                (center[0], center[1] + radius + 1),
                (center[0] - radius - 1, center[1]),
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(panel, diamond, color, cv2.LINE_AA)
        cv2.polylines(panel, [diamond], True, DOT_EDGE_BGR, 2, cv2.LINE_AA)
    else:
        cv2.circle(panel, center, radius, color, -1, cv2.LINE_AA)
        cv2.circle(panel, center, radius, DOT_EDGE_BGR, 2, cv2.LINE_AA)

    cv2.putText(
        panel,
        f"T{track_id}",
        (center[0] + radius + 3, center[1] - radius - 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )


def render_tactical_panel(
    *,
    width: int,
    height: int,
    pitch: SoccerPitch,
    calibration: PitchCalibration,
    detections: sv.Detections,
    assignments: list[EntityAssignment] | None,
    class_name: Callable[[int], str],
) -> tuple[np.ndarray, int]:
    """Draw current-frame detections only; return panel and projected-dot count."""
    panel = np.full((height, width, 3), PANEL_BACKGROUND_BGR, dtype=np.uint8)
    origin, scale = _draw_pitch(panel, pitch)
    title = "TACTICAL PITCH"
    cv2.putText(
        panel,
        title,
        (16, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )

    if not calibration.trusted:
        message = f"CALIBRATION UNAVAILABLE — {calibration.reason}"
        (text_width, _), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(
            panel,
            message,
            (max(12, (width - text_width) // 2), height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (120, 205, 255),
            1,
            cv2.LINE_AA,
        )
        return panel, 0

    quality = (
        f"landmarks {calibration.visible_keypoints} | inliers {calibration.inlier_count} | "
        f"error {calibration.median_error_px:.1f}px"
    )
    cv2.putText(
        panel,
        quality,
        (width - min(width - 16, 430), 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (205, 225, 210),
        1,
        cv2.LINE_AA,
    )

    if detections.tracker_id is None or len(detections) == 0:
        return panel, 0
    foot_points = np.column_stack(
        (
            (detections.xyxy[:, 0] + detections.xyxy[:, 2]) / 2,
            detections.xyxy[:, 3],
        )
    ).astype(np.float32)
    projected = calibration.project(foot_points)
    finite = np.all(np.isfinite(projected), axis=1)
    on_pitch = (
        finite
        & (projected[:, 0] >= 0)
        & (projected[:, 0] <= pitch.length)
        & (projected[:, 1] >= 0)
        & (projected[:, 1] <= pitch.width)
    )

    radius = max(5, round(height * 0.018))
    projected_count = 0
    for index in np.flatnonzero(on_pitch):
        track_id = int(detections.tracker_id[index])
        if assignments is not None:
            assignment = assignments[index]
            color = assignment_color(assignment)
            role = assignment.role
        else:
            color, role = _fallback_color_and_role(
                detections,
                int(index),
                track_id,
                class_name,
            )
        center = _world_to_panel(
            (float(projected[index, 0]), float(projected[index, 1])),
            origin=origin,
            scale=scale,
        )
        _draw_entity_marker(
            panel,
            center,
            color=color,
            role=role,
            track_id=track_id,
            radius=radius,
        )
        projected_count += 1
    return panel, projected_count


def stack_replay(annotated_frame: np.ndarray, tactical_panel: np.ndarray) -> np.ndarray:
    if annotated_frame.shape[1] != tactical_panel.shape[1]:
        raise ValueError("Replay and tactical panel widths must match.")
    return np.vstack((annotated_frame, tactical_panel))
