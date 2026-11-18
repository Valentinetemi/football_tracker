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
        "--codec",
        default="mp4v",
        help="four-character OpenCV output codec (default: mp4v)",
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

