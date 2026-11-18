"""End-to-end football video detection, tracking, and annotation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import supervision as sv
from tqdm import tqdm

from football_tracker.annotation import TrailStore, annotate_frame
from football_tracker.config import PipelineConfig
from football_tracker.detection import UltralyticsDetector
from football_tracker.scene import SceneChangeDetector
from football_tracker.tracking import TrackManager
from football_tracker.video import VideoReader, VideoWriter


class Detector(Protocol):
    def detect(self, frame):  # type: ignore[no-untyped-def]
        ...

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

    def __str__(self) -> str:
        tracks = "unknown" if self.unique_tracks is None else str(self.unique_tracks)
        return (
            f"Wrote {self.frames_written} frames at {self.fps:.3f} FPS to {self.output_path} "
            f"({tracks} tracks, {self.camera_cut_resets} camera-cut resets)."
        )


def run_pipeline(
    config: PipelineConfig,
    *,
    detector: Detector | None = None,
    tracks: Tracks | None = None,
    show_progress: bool = True,
) -> PipelineSummary:
    """Process every decoded source frame once and write it at the source FPS."""
    config.validate()
    with VideoReader(config.input_path) as reader:
        if reader.metadata is None:  # defensive; __enter__ always sets it
            raise RuntimeError("Input video metadata was not initialized.")
        metadata = reader.metadata
        if detector is None:
            detector = UltralyticsDetector(
                config.model,
                device=config.device,
                confidence=config.confidence,
                image_size=config.image_size,
                class_ids=config.class_ids,
            )
        if tracks is None:
            tracks = TrackManager(
                frame_rate=metadata.fps,
                track_activation_threshold=config.confidence,
                lost_track_buffer=config.track_buffer,
                minimum_consecutive_frames=config.minimum_consecutive_frames,
            )

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

                detections = detector.detect(frame)
                tracked = tracks.update(detections)
                annotated = annotate_frame(
                    frame,
                    tracked,
                    class_name=detector.class_name,
                    trails=trails,
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
    )
