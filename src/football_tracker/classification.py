"""Uncertainty-aware shirt appearance classification for visible football people."""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
import supervision as sv

Role = Literal["player", "goalkeeper", "referee", "unknown"]

PLAYER_CLASS_NAMES = {"person", "player", "football player", "soccer player"}
GOALKEEPER_CLASS_NAMES = {"goalkeeper", "goalie", "keeper"}
REFEREE_CLASS_NAMES = {"referee", "official", "match official"}


def role_from_class_name(class_name: str) -> Role:
    normalized = class_name.strip().lower()
    if normalized in PLAYER_CLASS_NAMES:
        return "player"
    if normalized in GOALKEEPER_CLASS_NAMES:
        return "goalkeeper"
    if normalized in REFEREE_CLASS_NAMES:
        return "referee"
    return "unknown"


def extract_shirt_feature(frame: np.ndarray, box: np.ndarray) -> np.ndarray | None:
    """Describe the central upper torso with robust Lab colour quantiles."""
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = (float(value) for value in box)
    x1 = int(np.clip(round(x1), 0, frame_width - 1))
    x2 = int(np.clip(round(x2), 0, frame_width))
    y1 = int(np.clip(round(y1), 0, frame_height - 1))
    y2 = int(np.clip(round(y2), 0, frame_height))
    width = x2 - x1
    height = y2 - y1
    if width < 6 or height < 12:
        return None

    shirt_x1 = x1 + round(width * 0.18)
    shirt_x2 = x1 + round(width * 0.82)
    shirt_y1 = y1 + round(height * 0.15)
    shirt_y2 = y1 + round(height * 0.55)
    patch = frame[shirt_y1:shirt_y2, shirt_x1:shirt_x2]
    if patch.size == 0:
        return None

    pixels = patch.reshape(-1, 3)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    likely_grass = (
        (hsv[:, 0] >= 30)
        & (hsv[:, 0] <= 95)
        & (hsv[:, 1] >= 45)
        & (hsv[:, 2] >= 25)
    )
    non_grass = pixels[~likely_grass]
    minimum_pixels = max(12, round(len(pixels) * 0.15))
    if len(non_grass) >= minimum_pixels:
        pixels = non_grass
    if len(pixels) < 12:
        return None

    lab = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    quantiles = np.quantile(lab.astype(np.float32) / 255.0, [0.25, 0.5, 0.75], axis=0)
    return quantiles.reshape(-1).astype(np.float32)


@dataclass(frozen=True, slots=True)
class TeamPrediction:
    team: int | None
    confidence: float
    reason: str

    @property
    def label(self) -> str:
        return "unknown" if self.team is None else f"team-{chr(ord('a') + self.team)}"


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    ready: bool
    sample_count: int
    cluster_sizes: tuple[int, int]
    separation: float
    reason: str


class ShirtTeamClassifier:
    """Cluster two outfield shirt appearances and reject weak assignments."""

    def __init__(
        self,
        *,
        minimum_samples: int = 20,
        minimum_separation: float = 0.12,
        uncertainty_threshold: float = 0.35,
        maximum_iterations: int = 60,
    ) -> None:
        self.minimum_samples = minimum_samples
        self.minimum_separation = minimum_separation
        self.uncertainty_threshold = uncertainty_threshold
        self.maximum_iterations = maximum_iterations
        self.centers: np.ndarray | None = None
        self.acceptance_radii: np.ndarray | None = None
        self.report = CalibrationReport(False, 0, (0, 0), 0.0, "not calibrated")

    @property
    def ready(self) -> bool:
        return self.report.ready

    def fit(self, features: list[np.ndarray]) -> CalibrationReport:
        if len(features) < self.minimum_samples:
            self.centers = None
            self.acceptance_radii = None
            self.report = CalibrationReport(
                False,
                len(features),
                (0, 0),
                0.0,
                f"need at least {self.minimum_samples} usable shirt samples",
            )
            return self.report

        data = np.asarray(features, dtype=np.float32)
        if data.ndim != 2 or not np.all(np.isfinite(data)):
            raise ValueError("Shirt features must be a finite two-dimensional array.")

        global_center = np.mean(data, axis=0)
        first = data[np.argmax(np.linalg.norm(data - global_center, axis=1))]
        second = data[np.argmax(np.linalg.norm(data - first, axis=1))]
        centers = np.stack([first, second]).astype(np.float32)

        labels = np.zeros(len(data), dtype=int)
        for _ in range(self.maximum_iterations):
            distances = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2)
            next_labels = np.argmin(distances, axis=1)
            if len(np.unique(next_labels)) < 2:
                self.report = CalibrationReport(
                    False,
                    len(features),
                    (len(features), 0),
                    0.0,
                    "shirt samples collapsed into one appearance cluster",
                )
                return self.report
            next_centers = np.stack(
                [np.mean(data[next_labels == index], axis=0) for index in range(2)]
            )
            converged = np.array_equal(labels, next_labels) and np.allclose(
                centers, next_centers, atol=1e-5
            )
            labels = next_labels
            centers = next_centers.astype(np.float32)
            if converged:
                break

        # K-means labels are arbitrary. Ordering chroma then luminance makes runs repeatable.
        order = sorted(range(2), key=lambda index: tuple(centers[index, [1, 2, 0]]))
        centers = centers[order]
        remap = {old_index: new_index for new_index, old_index in enumerate(order)}
        labels = np.asarray([remap[int(label)] for label in labels], dtype=int)
        cluster_sizes = tuple(int(np.sum(labels == index)) for index in range(2))
        minimum_cluster_size = max(3, self.minimum_samples // 8)
        separation = float(np.linalg.norm(centers[0] - centers[1]))

        radii = []
        for index in range(2):
            cluster_distances = np.linalg.norm(data[labels == index] - centers[index], axis=1)
            empirical_radius = float(np.quantile(cluster_distances, 0.9))
            radii.append(max(empirical_radius * 1.75, separation * 0.3, 0.06))

        if min(cluster_sizes) < minimum_cluster_size:
            reason = f"one shirt cluster has fewer than {minimum_cluster_size} samples"
            ready = False
        elif separation < self.minimum_separation:
            reason = "shirt clusters overlap too strongly"
            ready = False
        else:
            reason = "ready"
            ready = True

        self.centers = centers
        self.acceptance_radii = np.asarray(radii, dtype=np.float32)
        self.report = CalibrationReport(
            ready,
            len(features),
            cluster_sizes,
            separation,
            reason,
        )
        return self.report

    def predict_feature(self, feature: np.ndarray | None) -> TeamPrediction:
        if feature is None:
            return TeamPrediction(None, 0.0, "shirt crop unavailable")
        if not self.ready or self.centers is None or self.acceptance_radii is None:
            return TeamPrediction(None, 0.0, self.report.reason)

        distances = np.linalg.norm(self.centers - feature[None, :], axis=1)
        order = np.argsort(distances)
        best, second = int(order[0]), int(order[1])
        best_distance = float(distances[best])
        second_distance = float(distances[second])
        margin = (second_distance - best_distance) / max(second_distance, 1e-6)
        radius = float(self.acceptance_radii[best])
        distance_quality = float(np.clip(1.0 - (best_distance / radius), 0.0, 1.0))
        confidence = float(np.clip((0.75 * margin) + (0.25 * distance_quality), 0.0, 1.0))

        if best_distance > radius:
            return TeamPrediction(None, confidence, "shirt appearance is outside both teams")
        if confidence < self.uncertainty_threshold:
            return TeamPrediction(None, confidence, "shirt appearance is ambiguous")
        return TeamPrediction(best, confidence, "classified")

    def predict(self, frame: np.ndarray, box: np.ndarray) -> TeamPrediction:
        return self.predict_feature(extract_shirt_feature(frame, box))


class TrackTeamSmoother:
    """Require repeated compatible shirt decisions before labelling a track."""

    def __init__(self, history_length: int = 15, minimum_votes: int = 3) -> None:
        self.history_length = history_length
        self.minimum_votes = minimum_votes
        self._history: dict[int, deque[TeamPrediction]] = {}

    def update(self, track_id: int, prediction: TeamPrediction) -> TeamPrediction:
        history = self._history.setdefault(track_id, deque(maxlen=self.history_length))
        history.append(prediction)
        known = [item for item in history if item.team is not None]
        if len(known) < self.minimum_votes:
            return TeamPrediction(None, prediction.confidence, "waiting for stable shirt votes")

        weights = np.zeros(2, dtype=np.float32)
        for item in known:
            weights[int(item.team)] += max(item.confidence, 0.05)
        winner = int(np.argmax(weights))
        total = float(np.sum(weights))
        confidence = float(weights[winner] / total) if total else 0.0
        if confidence < 0.65:
            return TeamPrediction(None, confidence, "track shirt votes conflict")
        return TeamPrediction(winner, confidence, "temporally smoothed")

    def clear(self) -> None:
        self._history.clear()


@dataclass(frozen=True, slots=True)
class EntityAssignment:
    role: Role
    team: int | None
    label: str
    confidence: float
    reason: str


class TeamAppearanceAssigner:
    """Keep outfield team decisions separate from goalkeeper and referee roles."""

    def __init__(
        self,
        classifier: ShirtTeamClassifier,
        *,
        history_length: int = 15,
        minimum_votes: int = 3,
    ) -> None:
        self.classifier = classifier
        self.smoother = TrackTeamSmoother(history_length, minimum_votes)
        self._latest_by_track: dict[int, EntityAssignment] = {}

    def assign(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        class_name: Callable[[int], str],
    ) -> list[EntityAssignment]:
        assignments: list[EntityAssignment] = []
        class_ids = detections.class_id
        tracker_ids = detections.tracker_id
        for index, box in enumerate(detections.xyxy):
            object_name = class_name(int(class_ids[index])) if class_ids is not None else "person"
            role = role_from_class_name(object_name)
            if role == "player":
                prediction = self.classifier.predict(frame, box)
                if tracker_ids is not None:
                    prediction = self.smoother.update(int(tracker_ids[index]), prediction)
                assignment = EntityAssignment(
                    role,
                    prediction.team,
                    prediction.label,
                    prediction.confidence,
                    prediction.reason,
                )
            elif role == "referee":
                assignment = EntityAssignment(role, None, "referee", 1.0, "detector role")
            elif role == "goalkeeper":
                assignment = EntityAssignment(
                    role,
                    None,
                    "goalkeeper/unknown",
                    0.0,
                    "goalkeeper team is not inferred from its different shirt",
                )
            else:
                assignment = EntityAssignment(role, None, "unknown", 0.0, "unrecognized role")
            assignments.append(assignment)
            if tracker_ids is not None:
                self._latest_by_track[int(tracker_ids[index])] = assignment
        return assignments

    def reset_tracks(self) -> None:
        self.smoother.clear()

    def summary_counts(self) -> tuple[tuple[str, int], ...]:
        counts = Counter(item.label for item in self._latest_by_track.values())
        return tuple(sorted(counts.items()))
