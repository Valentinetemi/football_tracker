"""Command-line entry point."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

from football_tracker.config import PipelineConfig


def _class_ids(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("classes must be comma-separated integers") from exc
    if not parsed or any(class_id < 0 for class_id in parsed):
        raise argparse.ArgumentTypeError("classes must contain non-negative integers")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="football-tracker",
        description="Detect and track visible people in a football video.",
    )
    parser.add_argument("input", type=Path, help="path to the source football video")
    parser.add_argument("-o", "--output", type=Path, required=True, help="annotated .mp4 path")
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Ultralytics model name or local checkpoint path (default: yolo11n.pt)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="inference device: auto, mps, cpu, or CUDA device (default: auto)",
    )
    parser.add_argument("--confidence", type=float, default=0.25, help="detection threshold")
    parser.add_argument("--image-size", type=int, default=1280, help="YOLO inference size")
    parser.add_argument(
        "--classes",
        type=_class_ids,
        default=None,
        metavar="0,1,...",
        help="model class IDs to track; by default football/person classes are inferred",
    )
    parser.add_argument(
        "--cut-threshold",
        type=float,
        default=0.25,
        help="camera-cut sensitivity from 0 to 1 (default: 0.25)",
    )
    parser.add_argument(
        "--no-cut-reset",
        action="store_true",
        help="do not reset tracker IDs after detected camera cuts",
    )
    parser.add_argument("--track-buffer", type=int, default=30, help="lost-track buffer in frames")
    parser.add_argument(
        "--minimum-consecutive-frames",
        type=int,
        default=2,
        help="detections required before a track is displayed",
    )
    parser.add_argument(
        "--no-team-classification",
        action="store_true",
        help="disable shirt-appearance team classification",
    )
    parser.add_argument(
        "--team-sample-seconds",
        type=float,
        default=1.0,
        help="seconds between calibration frames (default: 1.0)",
    )
    parser.add_argument(
        "--team-confidence",
        type=float,
        default=0.35,
        help="minimum shirt-classification confidence (default: 0.35)",
    )
    parser.add_argument(
        "--pitch-model",
        help="local Ultralytics pitch-landmark checkpoint; enables tactical pitch output",
    )
    parser.add_argument(
        "--pitch-image-size",
        type=int,
        default=640,
        help="pitch landmark inference size (default: 640)",
    )
    parser.add_argument(
        "--pitch-keypoint-confidence",
        type=float,
        default=0.5,
        help="minimum pitch landmark confidence (default: 0.5)",
    )
    parser.add_argument(
        "--pitch-minimum-keypoints",
        type=int,
        default=6,
        help="minimum landmarks required for calibration (default: 6)",
    )
    parser.add_argument(
        "--pitch-maximum-error",
        type=float,
        default=10.0,
        help="maximum median landmark reprojection error in pixels (default: 10)",
    )
    parser.add_argument(
        "--pitch-panel-height",
        type=int,
        help="tactical panel height in pixels (default: 38%% of source height, minimum 240)",
    )
    parser.add_argument(
        "--roster",
        type=Path,
        help="optional CSV with team,number,name; team must be team-a or team-b",
    )
    parser.add_argument(
        "--jersey-ocr-backend",
        choices=("easyocr",),
        default="easyocr",
        help="replaceable jersey OCR implementation (default: easyocr)",
    )
    parser.add_argument(
        "--jersey-model-directory",
        type=Path,
        default=Path("models/easyocr"),
        help="Git-ignored EasyOCR weights/cache directory",
    )
    parser.add_argument(
        "--jersey-sample-interval",
        type=int,
        default=5,
        help="frames between clear-crop OCR attempts per track (default: 5)",
    )
    parser.add_argument(
        "--jersey-minimum-votes",
        type=int,
        default=3,
        help="compatible OCR reads required for a suggestion (default: 3)",
    )
    parser.add_argument(
        "--jersey-minimum-confidence",
        type=float,
        default=0.35,
        help="minimum per-crop OCR confidence (default: 0.35)",
    )
    parser.add_argument(
        "--jersey-minimum-player-height",
        type=int,
        default=120,
        help="minimum detected player height for OCR (default: 120)",
    )
    parser.add_argument(
        "--track-ball",
        action="store_true",
        help="detect and conservatively track the checkpoint's ball class",
    )
    parser.add_argument(
        "--ball-confidence",
        type=float,
        default=0.35,
        help="minimum confidence for ball candidates (default: 0.35)",
    )
    parser.add_argument(
        "--ball-minimum-hits",
        type=int,
        default=2,
        help="consecutive detections required before showing the ball (default: 2)",
    )
    parser.add_argument(
        "--ball-maximum-gap",
        type=int,
        default=8,
        help="frames retaining hidden motion state after a missed ball (default: 8)",
    )
    parser.add_argument(
        "--replay-data",
        type=Path,
        help="optional per-frame .json export for the interactive replay dashboard",
    )
    parser.add_argument(
        "--codec",
        default="mp4v",
        help=(
            "four-character OpenCV output codec (default: mp4v). "
            "mp4v often will not play in browsers; use avc1 for H.264 when this "
            "OpenCV build supports it"
        ),
    )
    return parser


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        device=args.device,
        confidence=args.confidence,
        image_size=args.image_size,
        class_ids=args.classes,
        cut_threshold=args.cut_threshold,
        reset_on_cut=not args.no_cut_reset,
        track_buffer=args.track_buffer,
        minimum_consecutive_frames=args.minimum_consecutive_frames,
        classify_teams=not args.no_team_classification,
        team_sample_seconds=args.team_sample_seconds,
        team_uncertainty_threshold=args.team_confidence,
        pitch_model=args.pitch_model,
        pitch_image_size=args.pitch_image_size,
        pitch_keypoint_confidence=args.pitch_keypoint_confidence,
        pitch_minimum_keypoints=args.pitch_minimum_keypoints,
        pitch_maximum_error=args.pitch_maximum_error,
        pitch_panel_height=args.pitch_panel_height,
        roster_path=args.roster,
        jersey_ocr_backend=args.jersey_ocr_backend,
        jersey_model_directory=args.jersey_model_directory,
        jersey_sample_interval=args.jersey_sample_interval,
        jersey_minimum_votes=args.jersey_minimum_votes,
        jersey_minimum_confidence=args.jersey_minimum_confidence,
        jersey_minimum_player_height=args.jersey_minimum_player_height,
        track_ball=args.track_ball,
        ball_confidence=args.ball_confidence,
        ball_minimum_hits=args.ball_minimum_hits,
        ball_maximum_gap=args.ball_maximum_gap,
        replay_data_path=args.replay_data,
        codec=args.codec,
    )


def main(
    argv: Sequence[str] | None = None,
    runner: Callable[[PipelineConfig], object] | None = None,
) -> int:
    parser = build_parser()
    config = config_from_args(parser.parse_args(argv))
    try:
        config.validate()
        if runner is None:
            from football_tracker.pipeline import run_pipeline

            runner = run_pipeline
        result = runner(config)
        if result is not None:
            print(result)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
