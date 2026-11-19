"""Pitch landmark detection and uncertainty-gated camera calibration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from football_tracker.detection import resolve_device


def _pitch_vertices() -> np.ndarray:
    """Return Roboflow Sports' 32 soccer landmarks in pitch centimetres."""
    width = 7000.0
    length = 12000.0
    penalty_box_width = 4100.0
    penalty_box_length = 2015.0
    goal_box_width = 1832.0
    goal_box_length = 550.0
    centre_circle_radius = 915.0
    penalty_spot_distance = 1100.0
    return np.asarray(
        [
            (0, 0),
            (0, (width - penalty_box_width) / 2),
            (0, (width - goal_box_width) / 2),
            (0, (width + goal_box_width) / 2),
            (0, (width + penalty_box_width) / 2),
            (0, width),
            (goal_box_length, (width - goal_box_width) / 2),
            (goal_box_length, (width + goal_box_width) / 2),
            (penalty_spot_distance, width / 2),
            (penalty_box_length, (width - penalty_box_width) / 2),
            (penalty_box_length, (width - goal_box_width) / 2),
            (penalty_box_length, (width + goal_box_width) / 2),
            (penalty_box_length, (width + penalty_box_width) / 2),
            (length / 2, 0),
            (length / 2, width / 2 - centre_circle_radius),
            (length / 2, width / 2 + centre_circle_radius),
            (length / 2, width),
            (length - penalty_box_length, (width - penalty_box_width) / 2),
            (length - penalty_box_length, (width - goal_box_width) / 2),
            (length - penalty_box_length, (width + goal_box_width) / 2),
            (length - penalty_box_length, (width + penalty_box_width) / 2),
            (length - penalty_spot_distance, width / 2),
            (length - goal_box_length, (width - goal_box_width) / 2),
            (length - goal_box_length, (width + goal_box_width) / 2),
            (length, 0),
            (length, (width - penalty_box_width) / 2),
            (length, (width - goal_box_width) / 2),
            (length, (width + goal_box_width) / 2),
            (length, (width + penalty_box_width) / 2),
            (length, width),
            (length / 2 - centre_circle_radius, width / 2),
            (length / 2 + centre_circle_radius, width / 2),
        ],
        dtype=np.float32,
    )


@dataclass(frozen=True, slots=True)
class SoccerPitch:
    """Metric pitch geometry matching the published landmark checkpoint."""

    length: float = 12000.0
    width: float = 7000.0
    penalty_box_width: float = 4100.0
    penalty_box_length: float = 2015.0
    goal_box_width: float = 1832.0
    goal_box_length: float = 550.0
    centre_circle_radius: float = 915.0
    penalty_spot_distance: float = 1100.0
    vertices: np.ndarray = field(default_factory=_pitch_vertices)


@dataclass(frozen=True, slots=True)
class PitchKeypoints:
    """One indexed pitch-landmark observation in image coordinates."""

    xy: np.ndarray
    confidence: np.ndarray

    @classmethod
    def empty(cls, landmark_count: int = 32) -> PitchKeypoints:
        return cls(
            np.zeros((landmark_count, 2), dtype=np.float32),
            np.zeros(landmark_count, dtype=np.float32),
        )


@dataclass(frozen=True, slots=True)
class PitchCalibration:
    """A per-frame image-to-pitch transform and its measured trust signals."""

    trusted: bool
    image_to_pitch: np.ndarray | None
    visible_keypoints: int
    inlier_count: int
    inlier_ratio: float
    median_error_px: float
    image_coverage: float
    pitch_coverage: float
    reason: str

    @classmethod
    def unavailable(
        cls,
        reason: str,
        *,
        visible_keypoints: int = 0,
        inlier_count: int = 0,
        inlier_ratio: float = 0.0,
        median_error_px: float = float("inf"),
        image_coverage: float = 0.0,
        pitch_coverage: float = 0.0,
    ) -> PitchCalibration:
        return cls(
            False,
            None,
            visible_keypoints,
            inlier_count,
            inlier_ratio,
            median_error_px,
            image_coverage,
            pitch_coverage,
            reason,
        )

    def project(self, image_points: np.ndarray) -> np.ndarray:
        """Project image points to pitch centimetres when this frame is trusted."""
        points = np.asarray(image_points, dtype=np.float32)
        if points.size == 0:
            return points.reshape(-1, 2)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Image points must have shape (N, 2).")
        if not self.trusted or self.image_to_pitch is None:
            raise ValueError("Pitch calibration is unavailable for this frame.")
        projected = cv2.perspectiveTransform(points.reshape(-1, 1, 2), self.image_to_pitch)
        return projected.reshape(-1, 2).astype(np.float32)


class UltralyticsPitchDetector:
    """Load the published YOLO pose checkpoint and return its 32 landmarks."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str,
        image_size: int,
        confidence: float = 0.25,
        model: Any | None = None,
    ) -> None:
        if model is None:
            try:
                from ultralytics import YOLO

                model = YOLO(model_name)
            except Exception as exc:
                raise RuntimeError(f"Could not load pitch model {model_name!r}: {exc}") from exc
        self.model = model
        self.device = resolve_device(device)
        self.image_size = image_size
        self.confidence = confidence

    @staticmethod
    def _numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        return np.asarray(value)

    def detect(self, frame: np.ndarray) -> PitchKeypoints:
        try:
            result = self.model.predict(
                source=frame,
                conf=self.confidence,
                imgsz=self.image_size,
                device=self.device,
                verbose=False,
            )[0]
        except Exception as exc:
            raise RuntimeError(
                f"Pitch model inference failed on device {self.device!r}: {exc}"
            ) from exc

        keypoints = result.keypoints
        if keypoints is None or keypoints.xy is None or len(keypoints.xy) == 0:
            return PitchKeypoints.empty()

        box_confidence = self._numpy(result.boxes.conf).reshape(-1)
        candidate = int(np.argmax(box_confidence)) if len(box_confidence) else 0
        xy = self._numpy(keypoints.xy[candidate]).astype(np.float32)
        if keypoints.conf is None:
            confidence = np.ones(len(xy), dtype=np.float32)
        else:
            confidence = self._numpy(keypoints.conf[candidate]).astype(np.float32)
        return PitchKeypoints(xy, confidence)


def _convex_coverage(points: np.ndarray, denominator: float) -> float:
    if len(points) < 3 or denominator <= 0:
        return 0.0
    hull = cv2.convexHull(points.astype(np.float32))
    return float(abs(cv2.contourArea(hull)) / denominator)


class PitchCalibrator:
    """Fit a fresh homography for every frame and reject weak geometry."""

    def __init__(
        self,
        pitch: SoccerPitch | None = None,
        *,
        keypoint_confidence: float = 0.5,
        minimum_keypoints: int = 6,
        minimum_inlier_ratio: float = 0.65,
        maximum_median_error_px: float = 10.0,
        ransac_threshold_px: float = 12.0,
        minimum_image_coverage: float = 0.005,
        minimum_pitch_coverage: float = 0.08,
    ) -> None:
        self.pitch = pitch or SoccerPitch()
        self.keypoint_confidence = keypoint_confidence
        self.minimum_keypoints = minimum_keypoints
        self.minimum_inlier_ratio = minimum_inlier_ratio
        self.maximum_median_error_px = maximum_median_error_px
        self.ransac_threshold_px = ransac_threshold_px
        self.minimum_image_coverage = minimum_image_coverage
        self.minimum_pitch_coverage = minimum_pitch_coverage
        self.current = PitchCalibration.unavailable("not calibrated")

    def invalidate(self, reason: str = "camera cut") -> PitchCalibration:
        """Immediately discard the previous shot's geometry."""
        self.current = PitchCalibration.unavailable(reason)
        return self.current

    def update(
        self,
        keypoints: PitchKeypoints,
        *,
        frame_size: tuple[int, int],
    ) -> PitchCalibration:
        frame_width, frame_height = frame_size
        xy = np.asarray(keypoints.xy, dtype=np.float32)
        confidence = np.asarray(keypoints.confidence, dtype=np.float32).reshape(-1)
        if xy.shape != self.pitch.vertices.shape or len(confidence) != len(xy):
            self.current = PitchCalibration.unavailable(
                f"expected {len(self.pitch.vertices)} indexed pitch landmarks"
            )
            return self.current

        valid = (
            np.all(np.isfinite(xy), axis=1)
            & np.isfinite(confidence)
            & (confidence >= self.keypoint_confidence)
            & (xy[:, 0] > 1)
            & (xy[:, 1] > 1)
            & (xy[:, 0] < frame_width - 1)
            & (xy[:, 1] < frame_height - 1)
        )
        image_points = xy[valid]
        pitch_points = self.pitch.vertices[valid]
        visible_count = len(image_points)
        if visible_count < self.minimum_keypoints:
            self.current = PitchCalibration.unavailable(
                f"only {visible_count} reliable pitch landmarks",
                visible_keypoints=visible_count,
            )
            return self.current

        image_coverage = _convex_coverage(
            image_points,
            float(frame_width * frame_height),
        )
        pitch_coverage = _convex_coverage(
            pitch_points,
            float(self.pitch.length * self.pitch.width),
        )
        if image_coverage < self.minimum_image_coverage:
            self.current = PitchCalibration.unavailable(
                "pitch landmarks cover too little of the image",
                visible_keypoints=visible_count,
                image_coverage=image_coverage,
                pitch_coverage=pitch_coverage,
            )
            return self.current
        if pitch_coverage < self.minimum_pitch_coverage:
            self.current = PitchCalibration.unavailable(
                "visible landmarks cover too little of the pitch",
                visible_keypoints=visible_count,
                image_coverage=image_coverage,
                pitch_coverage=pitch_coverage,
            )
            return self.current

        pitch_to_image, inlier_mask = cv2.findHomography(
            pitch_points,
            image_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold_px,
        )
        if pitch_to_image is None or inlier_mask is None:
            self.current = PitchCalibration.unavailable(
                "homography could not be estimated",
                visible_keypoints=visible_count,
                image_coverage=image_coverage,
                pitch_coverage=pitch_coverage,
            )
            return self.current

        inliers = inlier_mask.reshape(-1).astype(bool)
        inlier_count = int(np.sum(inliers))
        inlier_ratio = inlier_count / visible_count
        projected = cv2.perspectiveTransform(
            pitch_points.reshape(-1, 1, 2),
            pitch_to_image,
        ).reshape(-1, 2)
        errors = np.linalg.norm(projected - image_points, axis=1)
        median_error = float(np.median(errors[inliers])) if inlier_count else float("inf")

        if inlier_count < self.minimum_keypoints or inlier_ratio < self.minimum_inlier_ratio:
            self.current = PitchCalibration.unavailable(
                "too few pitch landmarks agree with one camera view",
                visible_keypoints=visible_count,
                inlier_count=inlier_count,
                inlier_ratio=inlier_ratio,
                median_error_px=median_error,
                image_coverage=image_coverage,
                pitch_coverage=pitch_coverage,
            )
            return self.current
        if median_error > self.maximum_median_error_px:
            self.current = PitchCalibration.unavailable(
                "pitch landmark reprojection error is too high",
                visible_keypoints=visible_count,
                inlier_count=inlier_count,
                inlier_ratio=inlier_ratio,
                median_error_px=median_error,
                image_coverage=image_coverage,
                pitch_coverage=pitch_coverage,
            )
            return self.current

        try:
            image_to_pitch = np.linalg.inv(pitch_to_image)
        except np.linalg.LinAlgError:
            self.current = PitchCalibration.unavailable(
                "homography is singular",
                visible_keypoints=visible_count,
                inlier_count=inlier_count,
                inlier_ratio=inlier_ratio,
                median_error_px=median_error,
                image_coverage=image_coverage,
                pitch_coverage=pitch_coverage,
            )
            return self.current
        if not np.all(np.isfinite(image_to_pitch)):
            self.current = PitchCalibration.unavailable("homography is not finite")
            return self.current

        self.current = PitchCalibration(
            True,
            image_to_pitch,
            visible_count,
            inlier_count,
            inlier_ratio,
            median_error,
            image_coverage,
            pitch_coverage,
            "trusted",
        )
        return self.current
