from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from football_tracker.config import PipelineConfig
from football_tracker.pipeline import run_pipeline


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


if __name__ == "__main__":
    unittest.main()
