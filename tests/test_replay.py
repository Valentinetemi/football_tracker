from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import supervision as sv

from football_tracker.ball import BallObservation
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
        hidden_ball = {"visible": False, "reliable": False, "x_cm": None, "y_cm": None}
        self.assertEqual(payload["frames"][0]["ball"], hidden_ball)
        self.assertEqual(payload["frames"][1]["ball"], hidden_ball)

    def test_ball_is_exported_only_when_measured_and_calibration_is_trusted(self) -> None:
        on_pitch = BallObservation((100.0, 100.0), (96.0, 96.0, 104.0, 104.0), 0.8)
        off_pitch = BallObservation((2000.0, 100.0), (1996.0, 96.0, 2004.0, 104.0), 0.8)
        detections = sv.Detections.empty()

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "replay.json"
            with ReplayDataExporter(
                output,
                fps=25.0,
                frame_count=4,
                pitch=SoccerPitch(),
                source_video="clip.mp4",
            ) as exporter:
                exporter.write_frame(
                    frame_index=0,
                    segment=1,
                    camera_cut=False,
                    calibration=trusted_calibration(),
                    detections=detections,
                    assignments=None,
                    suggestions=None,
                    class_name=lambda _: "player",
                    ball=on_pitch,
                )
                exporter.write_frame(
                    frame_index=1,
                    segment=1,
                    camera_cut=False,
                    calibration=PitchCalibration.unavailable("too few landmarks"),
                    detections=detections,
                    assignments=None,
                    suggestions=None,
                    class_name=lambda _: "player",
                    ball=on_pitch,
                )
                exporter.write_frame(
                    frame_index=2,
                    segment=1,
                    camera_cut=False,
                    calibration=trusted_calibration(),
                    detections=detections,
                    assignments=None,
                    suggestions=None,
                    class_name=lambda _: "player",
                    ball=off_pitch,
                )
                exporter.write_frame(
                    frame_index=3,
                    segment=1,
                    camera_cut=False,
                    calibration=trusted_calibration(),
                    detections=detections,
                    assignments=None,
                    suggestions=None,
                    class_name=lambda _: "player",
                    ball=None,
                )
            payload = json.loads(output.read_text(encoding="utf-8"))

        visible = payload["frames"][0]["ball"]
        self.assertEqual(visible["visible"], True)
        self.assertEqual(visible["reliable"], True)
        self.assertAlmostEqual(visible["x_cm"], 1200.0)
        self.assertAlmostEqual(visible["y_cm"], 1166.67)
        untrusted = payload["frames"][1]["ball"]
        self.assertEqual(untrusted["reliable"], True)
        self.assertEqual(untrusted["visible"], False)
        self.assertIsNone(untrusted["x_cm"])
        self.assertIsNone(untrusted["y_cm"])
        outside = payload["frames"][2]["ball"]
        self.assertEqual(outside["reliable"], True)
        self.assertEqual(outside["visible"], False)
        self.assertIsNone(outside["x_cm"])
        self.assertIsNone(outside["y_cm"])
        absent = payload["frames"][3]["ball"]
        self.assertEqual(absent, {"visible": False, "reliable": False, "x_cm": None, "y_cm": None})

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
