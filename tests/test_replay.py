from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import supervision as sv

from football_tracker.classification import EntityAssignment
from football_tracker.pitch import PitchCalibration, SoccerPitch
from football_tracker.replay import ReplayDataExporter


def trusted_calibration() -> PitchCalibration:
    image_to_pitch = np.asarray(
        [[12000 / 1000, 0, 0], [0, 7000 / 600, 0], [0, 0, 1]],
        dtype=np.float64,
    )
    return PitchCalibration(True, image_to_pitch, 8, 8, 1.0, 1.0, 0.2, 0.3, "trusted")


class ReplayDataExporterTests(unittest.TestCase):
    def test_exports_positions_labels_and_explicit_unavailable_frames(self) -> None:
        detections = sv.Detections(
            xyxy=np.asarray([[100, 100, 140, 240]], dtype=np.float32),
            class_id=np.asarray([2]),
            tracker_id=np.asarray([7]),
        )
        assignments = [EntityAssignment("player", 0, "team-a", 0.9, "test")]

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "replay.json"
            with ReplayDataExporter(
                output,
                fps=25.0,
                frame_count=2,
                pitch=SoccerPitch(),
                source_video="clip.mp4",
            ) as exporter:
                exporter.write_frame(
                    frame_index=0,
                    segment=1,
                    camera_cut=False,
                    calibration=trusted_calibration(),
                    detections=detections,
                    assignments=assignments,
                    suggestions=None,
                    class_name=lambda _: "player",
                )
                exporter.write_frame(
                    frame_index=1,
                    segment=2,
                    camera_cut=True,
                    calibration=PitchCalibration.unavailable("camera cut"),
                    detections=detections,
                    assignments=assignments,
                    suggestions=None,
                    class_name=lambda _: "player",
                )
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["video"]["fps"], 25.0)
        self.assertEqual(payload["pitch"], {"length_cm": 12000.0, "width_cm": 7000.0})
        self.assertEqual(payload["frames"][0]["players"][0]["track_id"], 7)
        self.assertEqual(payload["frames"][0]["players"][0]["team"], "team-a")
        self.assertAlmostEqual(payload["frames"][0]["players"][0]["x_cm"], 1440.0)
        self.assertAlmostEqual(payload["frames"][0]["players"][0]["y_cm"], 2800.0)
        self.assertTrue(payload["frames"][1]["camera_cut"])
        self.assertFalse(payload["frames"][1]["calibration_trusted"])
        self.assertIsNone(payload["frames"][1]["players"][0]["x_cm"])
        self.assertIsNone(payload["frames"][1]["players"][0]["y_cm"])

    def test_rejects_out_of_order_frames_without_publishing_partial_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "replay.json"
            with self.assertRaises(ValueError), ReplayDataExporter(
                output,
                fps=25.0,
                frame_count=2,
                pitch=SoccerPitch(),
                source_video="clip.mp4",
            ) as exporter:
                exporter.write_frame(
                    frame_index=1,
                    segment=1,
                    camera_cut=False,
                    calibration=PitchCalibration.unavailable("test"),
                    detections=sv.Detections.empty(),
                    assignments=None,
                    suggestions=None,
                    class_name=lambda _: "player",
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
