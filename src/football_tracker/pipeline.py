"""End-to-end football video detection, tracking, and annotation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import supervision as sv
from tqdm import tqdm

from football_tracker.annotation import TrailStore, annotate_frame
from football_tracker.classification import (
    CalibrationReport,
    ShirtTeamClassifier,
    TeamAppearanceAssigner,
    extract_shirt_feature,
    role_from_class_name,
)
from football_tracker.config import PipelineConfig
from football_tracker.detection import UltralyticsDetector
from football_tracker.scene import SceneChangeDetector
from football_tracker.tracking import TrackManager
from football_tracker.video import VideoReader, VideoWriter


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> sv.Detections: ...

    def class_name(self, class_id: int) -> str: ...


class Tracks(Protocol):
    segment: int
    reset_count: int

    def update(self, detections: sv.Detections) -> sv.Detections: ...

    def reset_for_camera_cut(self) -> None: ...


@dataclass(frozen=True, slots=True)
class PipelineSummary:
    output_path: str
    frames_written: int
    fps: float
    camera_cut_resets: int
    unique_tracks: int | None
    team_calibration: CalibrationReport | None
    classification_counts: tuple[tuple[str, int], ...]

    def __str__(self) -> str:
        tracks = "unknown" if self.unique_tracks is None else str(self.unique_tracks)
        result = (
            f"Wrote {self.frames_written} frames at {self.fps:.3f} FPS to {self.output_path} "
            f"({tracks} tracks, {self.camera_cut_resets} camera-cut resets)."
        )
        if self.team_calibration is not None:
            state = "ready" if self.team_calibration.ready else "unavailable"
            counts = ", ".join(f"{label}={count}" for label, count in self.classification_counts)
            result += f" Team classification {state}"
            result += f" ({counts})." if counts else "."
        return result


def collect_team_features(
    input_path: Path,
    detector: Detector,
    *,
    frame_rate: float,
    sample_seconds: float,
    show_progress: bool,
    maximum_samples: int = 600,
) -> list[np.ndarray]:
    """Sample sparse untracked player crops for a global two-team calibration."""
    features: list[np.ndarray] = []
    stride = max(1, round(frame_rate * sample_seconds))
    with VideoReader(input_path) as reader:
        total_frames = reader.metadata.reported_frame_count if reader.metadata else 0
        total_samples = ((total_frames - 1) // stride) + 1 if total_frames else None
        progress = tqdm(
            total=total_samples,
            desc="calibrating teams",
            unit="sample",
            disable=not show_progress,
        )
        with progress:
            for frame_index, frame in reader:
                if frame_index % stride != 0:
                    continue
                detections = detector.detect(frame)
                if detections.class_id is not None:
                    for box, class_id in zip(
                        detections.xyxy,
                        detections.class_id,
                        strict=True,
                    ):
                        if role_from_class_name(detector.class_name(int(class_id))) != "player":
                            continue
                        feature = extract_shirt_feature(frame, box)
                        if feature is not None:
                            features.append(feature)
                        if len(features) >= maximum_samples:
                            break
                progress.update(1)
                if len(features) >= maximum_samples:
                    break
    return features


def run_pipeline(
    config: PipelineConfig,
    *,
    detector: Detector | None = None,
    tracks: Tracks | None = None,
    team_classifier: ShirtTeamClassifier | None = None,
    show_progress: bool = True,
) -> PipelineSummary:
    """Process every decoded source frame once and write it at the source FPS."""
    config.validate()
    with VideoReader(config.input_path) as metadata_reader:
        if metadata_reader.metadata is None:  # defensive; __enter__ always sets it
            raise RuntimeError("Input video metadata was not initialized.")
        metadata = metadata_reader.metadata
        if detector is None:
            detector = UltralyticsDetector(
                config.model,
                device=config.device,
                confidence=config.confidence,
                image_size=config.image_size,
                class_ids=config.class_ids,
            )

    appearance_assigner = None
    calibration_report = None
    if config.classify_teams:
        if team_classifier is None:
            team_classifier = ShirtTeamClassifier(
                minimum_samples=config.team_minimum_samples,
                uncertainty_threshold=config.team_uncertainty_threshold,
            )
            features = collect_team_features(
                config.input_path,
                detector,
                frame_rate=metadata.fps,
                sample_seconds=config.team_sample_seconds,
                show_progress=show_progress,
            )
            team_classifier.fit(features)
        calibration_report = team_classifier.report
        appearance_assigner = TeamAppearanceAssigner(team_classifier)

    if tracks is None:
        tracks = TrackManager(
            frame_rate=metadata.fps,
            track_activation_threshold=config.confidence,
            lost_track_buffer=config.track_buffer,
            minimum_consecutive_frames=config.minimum_consecutive_frames,
        )

    with VideoReader(config.input_path) as reader:
        scene_detector = SceneChangeDetector(threshold=config.cut_threshold)
        trails = TrailStore()
        total = metadata.reported_frame_count or None
        progress = tqdm(total=total, desc="tracking", unit="frame", disable=not show_progress)
        with progress, VideoWriter(config.output_path, metadata, config.codec) as writer:
            for _, frame in reader:
                scene_change = scene_detector.update(frame)
                reset_for_cut = config.reset_on_cut and scene_change.is_cut
                if reset_for_cut:
                    tracks.reset_for_camera_cut()
                    trails.reset()
                    if appearance_assigner is not None:
                        appearance_assigner.reset_tracks()

                detections = detector.detect(frame)
                tracked = tracks.update(detections)
                assignments = (
                    appearance_assigner.assign(frame, tracked, detector.class_name)
                    if appearance_assigner is not None
                    else None
                )
                annotated = annotate_frame(
                    frame,
                    tracked,
                    class_name=detector.class_name,
                    trails=trails,
                    assignments=assignments,
                    segment=tracks.segment,
                    camera_cut=reset_for_cut,
                    scene_score=scene_change.score,
                )
                writer.write(annotated)
                progress.update(1)

            frames_written = writer.frames_written

    unique_tracks = getattr(tracks, "display_track_count", None)
    return PipelineSummary(
        output_path=str(config.output_path),
        frames_written=frames_written,
        fps=metadata.fps,
        camera_cut_resets=tracks.reset_count,
        unique_tracks=unique_tracks,
        team_calibration=calibration_report,
        classification_counts=(
            appearance_assigner.summary_counts() if appearance_assigner is not None else ()
        ),
    )
