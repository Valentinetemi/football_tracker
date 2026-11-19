"""Configuration shared by the CLI and processing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """User-configurable settings for one video processing run."""

    input_path: Path
    output_path: Path
    model: str = "yolo11n.pt"
    device: str = "auto"
    confidence: float = 0.25
    image_size: int = 1280
    class_ids: tuple[int, ...] | None = None
    cut_threshold: float = 0.25
    reset_on_cut: bool = True
    track_buffer: int = 30
    minimum_consecutive_frames: int = 2
    classify_teams: bool = True
    team_sample_seconds: float = 1.0
    team_minimum_samples: int = 20
    team_uncertainty_threshold: float = 0.35
    pitch_model: str | None = None
    pitch_image_size: int = 640
    pitch_keypoint_confidence: float = 0.5
    pitch_minimum_keypoints: int = 6
    pitch_maximum_error: float = 10.0
    codec: str = "mp4v"

    def validate(self) -> None:
        """Raise a helpful error before loading any large ML dependencies."""
        if not self.input_path.exists():
            raise ValueError(f"Input video does not exist: {self.input_path}")
        if not self.input_path.is_file():
            raise ValueError(f"Input path is not a file: {self.input_path}")
        if self.input_path.resolve() == self.output_path.resolve():
            raise ValueError("Input and output paths must be different.")
        if self.output_path.suffix.lower() != ".mp4":
            raise ValueError("Output path must end in .mp4")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("Confidence must be between 0 and 1.")
        if self.image_size <= 0:
            raise ValueError("Image size must be positive.")
        if not 0.0 <= self.cut_threshold <= 1.0:
            raise ValueError("Cut threshold must be between 0 and 1.")
        if self.track_buffer < 1:
            raise ValueError("Track buffer must be at least 1 frame.")
        if self.minimum_consecutive_frames < 1:
            raise ValueError("Minimum consecutive frames must be at least 1.")
        if self.team_sample_seconds <= 0:
            raise ValueError("Team sample interval must be positive.")
        if self.team_minimum_samples < 4:
            raise ValueError("Team classification needs at least 4 samples.")
        if not 0.0 < self.team_uncertainty_threshold < 1.0:
            raise ValueError("Team uncertainty threshold must be between 0 and 1.")
        if self.pitch_model is not None and not Path(self.pitch_model).expanduser().is_file():
            raise ValueError(f"Pitch model does not exist: {self.pitch_model}")
        if self.pitch_image_size <= 0:
            raise ValueError("Pitch image size must be positive.")
        if not 0.0 < self.pitch_keypoint_confidence < 1.0:
            raise ValueError("Pitch keypoint confidence must be between 0 and 1.")
        if self.pitch_minimum_keypoints < 4:
            raise ValueError("Pitch calibration needs at least 4 keypoints.")
        if self.pitch_maximum_error <= 0:
            raise ValueError("Pitch maximum reprojection error must be positive.")
        if len(self.codec) != 4:
            raise ValueError("Codec must be a four-character OpenCV FourCC value.")

