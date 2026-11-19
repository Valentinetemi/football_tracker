"""End-to-end football video detection, tracking, and annotation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import supervision as sv
from tqdm import tqdm

from football_tracker.annotation import TrailStore, annotate_frame
from football_tracker.ball import (
    BallTracker,
    BallTrackingSummary,
    split_ball_detections,
)
from football_tracker.classification import (
    CalibrationReport,
    ShirtTeamClassifier,
    TeamAppearanceAssigner,
    extract_shirt_feature,
    role_from_class_name,
)
from football_tracker.config import PipelineConfig
from football_tracker.detection import UltralyticsDetector
from football_tracker.identity import (
    EasyOCRJerseyNumberOCR,
    JerseyNumberOCR,
    JerseySuggestionEngine,
    JerseySuggestionSummary,
    Roster,
)
from football_tracker.pitch import (
    PitchCalibration,
    PitchCalibrator,
    PitchKeypoints,
    SoccerPitch,
    UltralyticsPitchDetector,
)
from football_tracker.scene import SceneChangeDetector
from football_tracker.tactical import render_tactical_panel, resolve_panel_height, stack_replay
from football_tracker.tracking import TrackManager
from football_tracker.video import VideoMetadata, VideoReader, VideoWriter


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> sv.Detections: ...

    def class_name(self, class_id: int) -> str: ...


class Tracks(Protocol):
    segment: int
    reset_count: int

    def update(self, detections: sv.Detections) -> sv.Detections: ...

    def reset_for_camera_cut(self) -> None: ...


class PitchDetector(Protocol):
    def detect(self, frame: np.ndarray) -> PitchKeypoints: ...


class Calibrator(Protocol):
    pitch: SoccerPitch

    def update(
        self,
        keypoints: PitchKeypoints,
        *,
        frame_size: tuple[int, int],
    ) -> PitchCalibration: ...

    def invalidate(self, reason: str = "camera cut") -> PitchCalibration: ...


@dataclass(frozen=True, slots=True)
class PipelineSummary:
    output_path: str
    frames_written: int
    fps: float
    camera_cut_resets: int
    unique_tracks: int | None
    team_calibration: CalibrationReport | None
    classification_counts: tuple[tuple[str, int], ...]
    pitch_trusted_frames: int
    pitch_unavailable_frames: int
    projected_positions: int
    pitch_dropout_reasons: tuple[tuple[str, int], ...]
    jersey_suggestions: JerseySuggestionSummary | None
    ball_tracking: BallTrackingSummary | None

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
        pitch_frames = self.pitch_trusted_frames + self.pitch_unavailable_frames
        if pitch_frames:
            result += (
                f" Pitch calibration trusted on {self.pitch_trusted_frames}/{pitch_frames} "
                f"frames with {self.projected_positions} visible positions projected."
            )
        if self.jersey_suggestions is not None:
            result += (
                f" Jersey suggestions: {self.jersey_suggestions.suggested_tracks} tracks "
                f"from {self.jersey_suggestions.accepted_reads} accepted reads across "
                f"{self.jersey_suggestions.clear_crops_sampled} clear crops; "
                f"{self.jersey_suggestions.unresolved_tracks} unresolved."
            )
        if self.ball_tracking is not None:
            result += (
                f" Ball visible on {self.ball_tracking.reliable_frames} frames; "
                f"missed={self.ball_tracking.missed_frames}, "
                f"ambiguous={self.ball_tracking.ambiguous_frames}, "
                f"recoveries={self.ball_tracking.recoveries}."
            )
        return result


def collect_team_features(
    input_path: Path,
    detector: Detector,
    *,
    frame_rate: float,
    sample_seconds: float,
    minimum_confidence: float,
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
                    confidence = (
                        detections.confidence
                        if detections.confidence is not None
                        else np.ones(len(detections), dtype=np.float32)
                    )
                    for box, class_id, score in zip(
                        detections.xyxy,
                        detections.class_id,
                        confidence,
                        strict=True,
                    ):
                        if float(score) < minimum_confidence:
                            continue
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
    pitch_detector: PitchDetector | None = None,
    pitch_calibrator: Calibrator | None = None,
    jersey_ocr: JerseyNumberOCR | None = None,
    ball_tracker: BallTracker | None = None,
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
                include_ball=config.track_ball,
                ball_confidence=config.ball_confidence,
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
                minimum_confidence=config.confidence,
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

    pitch_enabled = config.pitch_model is not None or pitch_detector is not None
    panel_height = 0
    output_metadata = metadata
    if pitch_enabled:
        if pitch_detector is None:
            if config.pitch_model is None:  # defensive; covered by pitch_enabled
                raise RuntimeError("Pitch model path was not configured.")
            pitch_detector = UltralyticsPitchDetector(
                config.pitch_model,
                device=config.device,
                image_size=config.pitch_image_size,
            )
        if pitch_calibrator is None:
            pitch_calibrator = PitchCalibrator(
                keypoint_confidence=config.pitch_keypoint_confidence,
                minimum_keypoints=config.pitch_minimum_keypoints,
                maximum_median_error_px=config.pitch_maximum_error,
            )
        panel_height = resolve_panel_height(metadata.height, config.pitch_panel_height)
        output_metadata = VideoMetadata(
            metadata.width,
            metadata.height + panel_height,
            metadata.fps,
            metadata.reported_frame_count,
        )

    pitch_trusted_frames = 0
    pitch_unavailable_frames = 0
    projected_positions = 0
    pitch_dropout_counts: dict[str, int] = {}

    jersey_engine = None
    if config.roster_path is not None:
        roster = Roster.from_csv(config.roster_path)
        if jersey_ocr is None:
            jersey_ocr = EasyOCRJerseyNumberOCR(
                device=config.device,
                model_directory=config.jersey_model_directory,
            )
        jersey_engine = JerseySuggestionEngine(
            roster,
            jersey_ocr,
            sample_interval=config.jersey_sample_interval,
            minimum_votes=config.jersey_minimum_votes,
            minimum_ocr_confidence=config.jersey_minimum_confidence,
            minimum_player_height=config.jersey_minimum_player_height,
            minimum_sharpness=config.jersey_minimum_sharpness,
        )

    if config.track_ball and ball_tracker is None:
        ball_tracker = BallTracker(
            minimum_hits=config.ball_minimum_hits,
            maximum_gap=config.ball_maximum_gap,
            maximum_distance_ratio=config.ball_maximum_distance_ratio,
        )

    with VideoReader(config.input_path) as reader:
        scene_detector = SceneChangeDetector(threshold=config.cut_threshold)
        trails = TrailStore()
        total = metadata.reported_frame_count or None
        progress = tqdm(total=total, desc="tracking", unit="frame", disable=not show_progress)
        with progress, VideoWriter(config.output_path, output_metadata, config.codec) as writer:
            for frame_index, frame in reader:
                scene_change = scene_detector.update(frame)
                reset_for_cut = config.reset_on_cut and scene_change.is_cut
                if reset_for_cut:
                    tracks.reset_for_camera_cut()
                    trails.reset()
                    if appearance_assigner is not None:
                        appearance_assigner.reset_tracks()
                    if jersey_engine is not None:
                        jersey_engine.reset_tracks()
                    if ball_tracker is not None:
                        ball_tracker.reset_for_camera_cut()

                detections = detector.detect(frame)
                if ball_tracker is not None:
                    people, ball_candidates = split_ball_detections(
                        detections,
                        class_name=detector.class_name,
                        people_confidence=config.confidence,
                        ball_confidence=config.ball_confidence,
                    )
                    _ball_observation = ball_tracker.update(
                        frame_index=frame_index,
                        candidates=ball_candidates,
                        frame_size=(metadata.width, metadata.height),
                    )
                else:
                    people = detections
                    _ball_observation = None
                tracked = tracks.update(people)
                assignments = (
                    appearance_assigner.assign(frame, tracked, detector.class_name)
                    if appearance_assigner is not None
                    else None
                )
                suggestions = (
                    jersey_engine.update(
                        frame_index=frame_index,
                        frame=frame,
                        detections=tracked,
                        assignments=assignments,
                    )
                    if jersey_engine is not None
                    else None
                )
                annotated = annotate_frame(
                    frame,
                    tracked,
                    class_name=detector.class_name,
                    trails=trails,
                    assignments=assignments,
                    suggestions=suggestions,
                    segment=tracks.segment,
                    camera_cut=reset_for_cut,
                    scene_score=scene_change.score,
                )
                output_frame = annotated
                if pitch_enabled:
                    if pitch_detector is None or pitch_calibrator is None:
                        raise RuntimeError("Pitch calibration components were not initialized.")
                    if reset_for_cut:
                        pitch_calibration = pitch_calibrator.invalidate("camera cut")
                    else:
                        pitch_keypoints = pitch_detector.detect(frame)
                        pitch_calibration = pitch_calibrator.update(
                            pitch_keypoints,
                            frame_size=(metadata.width, metadata.height),
                        )
                    tactical_panel, frame_positions = render_tactical_panel(
                        width=metadata.width,
                        height=panel_height,
                        pitch=pitch_calibrator.pitch,
                        calibration=pitch_calibration,
                        detections=tracked,
                        assignments=assignments,
                        class_name=detector.class_name,
                        suggestions=suggestions,
                    )
                    if pitch_calibration.trusted:
                        pitch_trusted_frames += 1
                        projected_positions += frame_positions
                    else:
                        pitch_unavailable_frames += 1
                        pitch_dropout_counts[pitch_calibration.reason] = (
                            pitch_dropout_counts.get(pitch_calibration.reason, 0) + 1
                        )
                    output_frame = stack_replay(annotated, tactical_panel)
                writer.write(output_frame)
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
        pitch_trusted_frames=pitch_trusted_frames,
        pitch_unavailable_frames=pitch_unavailable_frames,
        projected_positions=projected_positions,
        pitch_dropout_reasons=tuple(sorted(pitch_dropout_counts.items())),
        jersey_suggestions=jersey_engine.summary() if jersey_engine is not None else None,
        ball_tracking=ball_tracker.summary() if ball_tracker is not None else None,
    )
