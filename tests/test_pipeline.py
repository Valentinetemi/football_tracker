from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from football_tracker.classification import CalibrationReport, TeamPrediction
from football_tracker.config import PipelineConfig
from football_tracker.identity import JerseyRead
from football_tracker.pipeline import run_pipeline
from football_tracker.pitch import PitchCalibration, PitchKeypoints, SoccerPitch


class FakeDetector:
    def detect(self, frame: np.ndarray) -> sv.Detections:
        height, width = frame.shape[:2]
        return sv.Detections(
            xyxy=np.asarray([[width * 0.35, height * 0.2, width * 0.55, height * 0.9]]),
            confidence=np.asarray([0.9]),
            class_id=np.asarray([0]),
        )

    def class_name(self, class_id: int) -> str:
        return "person"


class FakeTracks:
    segment = 1
    reset_count = 0
    display_track_count = 1

    def update(self, detections: sv.Detections) -> sv.Detections:
        detections.tracker_id = np.asarray([1])
        return detections

    def reset_for_camera_cut(self) -> None:
        self.segment += 1
        self.reset_count += 1


class FakePitchDetector:
    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame: np.ndarray) -> PitchKeypoints:
        self.calls += 1
        return PitchKeypoints.empty()


class FakePitchCalibrator:
    pitch = SoccerPitch()

    def __init__(self) -> None:
        self.invalidations = 0

    def update(
        self,
        keypoints: PitchKeypoints,
        *,
        frame_size: tuple[int, int],
    ) -> PitchCalibration:
        width, height = frame_size
        image_to_pitch = np.asarray(
            [[self.pitch.length / width, 0, 0], [0, self.pitch.width / height, 0], [0, 0, 1]],
            dtype=np.float64,
        )
        return PitchCalibration(
            True,
            image_to_pitch,
            8,
            8,
            1.0,
            1.0,
            0.2,
            0.3,
            "trusted",
        )

    def invalidate(self, reason: str = "camera cut") -> PitchCalibration:
        self.invalidations += 1
        return PitchCalibration.unavailable(reason)


class ConstantTeamClassifier:
    report = CalibrationReport(True, 20, (10, 10), 0.5, "ready")

    def predict(self, frame: np.ndarray, box: np.ndarray) -> TeamPrediction:
        return TeamPrediction(0, 0.9, "test")


class ConstantJerseyOCR:
    def __init__(self, number: int) -> None:
        self.number = number
        self.calls = 0

    def read(self, crop: np.ndarray) -> JerseyRead:
        self.calls += 1
        return JerseyRead(self.number, 0.9, str(self.number))


def make_source_video(path: Path, *, fps: float = 12.5, frames: int = 8) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (160, 96),
    )
    if not writer.isOpened():
        raise RuntimeError("Test environment cannot create an mp4v video.")
    for index in range(frames):
        frame = np.full((96, 160, 3), (20, 110 + index, 20), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def make_cut_source_video(path: Path, *, fps: float = 12.5, frames: int = 8) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (160, 96),
    )
    if not writer.isOpened():
        raise RuntimeError("Test environment cannot create an mp4v video.")
    for index in range(frames):
        value = 15 if index < frames // 2 else 240
        writer.write(np.full((96, 160, 3), value, dtype=np.uint8))
    writer.release()


class PipelineTests(unittest.TestCase):
    def test_writes_one_output_frame_per_input_frame_at_source_fps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            output = root / "tracked.mp4"
            make_source_video(source)

            summary = run_pipeline(
                PipelineConfig(input_path=source, output_path=output, reset_on_cut=False),
                detector=FakeDetector(),
                tracks=FakeTracks(),
                show_progress=False,
            )

            capture = cv2.VideoCapture(str(output))
            self.assertTrue(capture.isOpened())
            output_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            output_fps = capture.get(cv2.CAP_PROP_FPS)
            ok, first_frame = capture.read()
            capture.release()

            self.assertTrue(ok)
            self.assertEqual(first_frame.shape[:2], (96, 160))
            self.assertGreater(int(first_frame.max()), 200)
            self.assertEqual(output_frames, 8)
            self.assertAlmostEqual(output_fps, 12.5, places=2)
            self.assertEqual(summary.frames_written, 8)
            self.assertEqual(summary.fps, 12.5)
            self.assertEqual(summary.unique_tracks, 1)

    def test_appends_synchronized_tactical_panel_without_changing_timing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            output = root / "tactical.mp4"
            make_source_video(source)
            pitch_detector = FakePitchDetector()

            summary = run_pipeline(
                PipelineConfig(
                    input_path=source,
                    output_path=output,
                    reset_on_cut=False,
                    classify_teams=False,
                    pitch_panel_height=80,
                ),
                detector=FakeDetector(),
                tracks=FakeTracks(),
                pitch_detector=pitch_detector,
                pitch_calibrator=FakePitchCalibrator(),
                show_progress=False,
            )

            capture = cv2.VideoCapture(str(output))
            output_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            output_fps = capture.get(cv2.CAP_PROP_FPS)
            ok, first_frame = capture.read()
            capture.release()

        self.assertTrue(ok)
        self.assertEqual(first_frame.shape[:2], (176, 160))
        self.assertEqual(output_frames, 8)
        self.assertAlmostEqual(output_fps, 12.5, places=2)
        self.assertEqual(pitch_detector.calls, 8)
        self.assertEqual(summary.pitch_trusted_frames, 8)
        self.assertEqual(summary.pitch_unavailable_frames, 0)
        self.assertEqual(summary.projected_positions, 8)

    def test_camera_cut_invalidates_pitch_for_the_cut_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            output = root / "tactical.mp4"
            make_cut_source_video(source)
            pitch_detector = FakePitchDetector()
            pitch_calibrator = FakePitchCalibrator()

            summary = run_pipeline(
                PipelineConfig(
                    input_path=source,
                    output_path=output,
                    classify_teams=False,
                    pitch_panel_height=80,
                ),
                detector=FakeDetector(),
                tracks=FakeTracks(),
                pitch_detector=pitch_detector,
                pitch_calibrator=pitch_calibrator,
                show_progress=False,
            )

        self.assertEqual(pitch_calibrator.invalidations, 1)
        self.assertEqual(pitch_detector.calls, 7)
        self.assertEqual(summary.pitch_trusted_frames, 7)
        self.assertEqual(summary.pitch_unavailable_frames, 1)
        self.assertEqual(summary.pitch_dropout_reasons, (("camera cut", 1),))

    def test_roster_suggestion_is_opt_in_and_does_not_change_timing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            output = root / "identity.mp4"
            roster = root / "roster.csv"
            make_source_video(source)
            roster.write_text(
                "team,number,name\nteam-a,10,Ada\nteam-b,10,Bea\n",
                encoding="utf-8",
            )
            ocr = ConstantJerseyOCR(10)

            summary = run_pipeline(
                PipelineConfig(
                    input_path=source,
                    output_path=output,
                    reset_on_cut=False,
                    roster_path=roster,
                    jersey_sample_interval=1,
                    jersey_minimum_votes=2,
                    jersey_minimum_player_height=24,
                    jersey_minimum_sharpness=0,
                ),
                detector=FakeDetector(),
                tracks=FakeTracks(),
                team_classifier=ConstantTeamClassifier(),  # type: ignore[arg-type]
                jersey_ocr=ocr,
                show_progress=False,
            )

            capture = cv2.VideoCapture(str(output))
            output_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            output_fps = capture.get(cv2.CAP_PROP_FPS)
            capture.release()

        self.assertEqual(output_frames, 8)
        self.assertAlmostEqual(output_fps, 12.5, places=2)
        self.assertEqual(ocr.calls, 2)
        self.assertEqual(summary.jersey_suggestions.suggested_tracks, 1)  # type: ignore[union-attr]
        self.assertEqual(summary.jersey_suggestions.unresolved_tracks, 0)  # type: ignore[union-attr]
        self.assertIn("2 accepted reads across 2 clear crops", str(summary))


if __name__ == "__main__":
    unittest.main()
