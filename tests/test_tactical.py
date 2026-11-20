from __future__ import annotations

import unittest

import numpy as np
import supervision as sv

from football_tracker.ball import BallObservation
from football_tracker.classification import EntityAssignment
from football_tracker.identity import PlayerSuggestion
from football_tracker.pitch import PitchCalibration, SoccerPitch
from football_tracker.tactical import project_measured_ball, render_tactical_panel, stack_replay


def trusted_calibration() -> PitchCalibration:
    image_to_pitch = np.asarray(
        [[12000 / 1000, 0, 0], [0, 7000 / 600, 0], [0, 0, 1]],
        dtype=np.float64,
    )
    return PitchCalibration(True, image_to_pitch, 8, 8, 1.0, 1.2, 0.3, 0.4, "trusted")


class TacticalRenderingTests(unittest.TestCase):
    def test_renders_only_current_visible_detections(self) -> None:
        detections = sv.Detections(
            xyxy=np.asarray(
                [
                    [100, 100, 140, 240],
                    [500, 120, 540, 260],
                    [800, 110, 840, 250],
                ],
                dtype=np.float32,
            ),
            class_id=np.asarray([2, 2, 3]),
            tracker_id=np.asarray([7, 8, 9]),
        )
        assignments = [
            EntityAssignment("player", 0, "team-a", 0.9, "test"),
            EntityAssignment("player", 1, "team-b", 0.9, "test"),
            EntityAssignment("referee", None, "referee", 1.0, "test"),
        ]

        panel, count = render_tactical_panel(
            width=1000,
            height=360,
            pitch=SoccerPitch(),
            calibration=trusted_calibration(),
            detections=detections,
            assignments=assignments,
            class_name=lambda _: "player",
        )

        self.assertEqual(panel.shape, (360, 1000, 3))
        self.assertEqual(count, 3)

    def test_untrusted_calibration_renders_no_dots(self) -> None:
        detections = sv.Detections(
            xyxy=np.asarray([[100, 100, 140, 240]], dtype=np.float32),
            class_id=np.asarray([2]),
            tracker_id=np.asarray([7]),
        )

        _, count = render_tactical_panel(
            width=1000,
            height=360,
            pitch=SoccerPitch(),
            calibration=PitchCalibration.unavailable("test dropout"),
            detections=detections,
            assignments=None,
            class_name=lambda _: "player",
        )

        self.assertEqual(count, 0)

    def test_accepts_roster_suggestion_for_tactical_marker(self) -> None:
        detections = sv.Detections(
            xyxy=np.asarray([[100, 100, 140, 240]], dtype=np.float32),
            class_id=np.asarray([2]),
            tracker_id=np.asarray([7]),
        )
        assignments = [EntityAssignment("player", 0, "team-a", 0.9, "test")]
        suggestions = [PlayerSuggestion(7, "team-a", 10, "Ada", 0.8, 3, 3)]

        panel, count = render_tactical_panel(
            width=1000,
            height=360,
            pitch=SoccerPitch(),
            calibration=trusted_calibration(),
            detections=detections,
            assignments=assignments,
            class_name=lambda _: "player",
            suggestions=suggestions,
        )

        self.assertEqual(panel.shape, (360, 1000, 3))
        self.assertEqual(count, 1)

    def test_ball_requires_trusted_calibration_on_tactical_pitch(self) -> None:
        ball = BallObservation((500.0, 300.0), (496.0, 296.0, 504.0, 304.0), 0.8)
        arguments = {
            "width": 1000,
            "height": 360,
            "pitch": SoccerPitch(),
            "detections": sv.Detections.empty(),
            "assignments": None,
            "class_name": lambda _: "player",
        }

        without_ball, _ = render_tactical_panel(
            calibration=trusted_calibration(),
            **arguments,  # type: ignore[arg-type]
        )
        with_ball, count = render_tactical_panel(
            calibration=trusted_calibration(),
            ball=ball,
            **arguments,  # type: ignore[arg-type]
        )
        untrusted_without, _ = render_tactical_panel(
            calibration=PitchCalibration.unavailable("test dropout"),
            **arguments,  # type: ignore[arg-type]
        )
        untrusted_with, _ = render_tactical_panel(
            calibration=PitchCalibration.unavailable("test dropout"),
            ball=ball,
            **arguments,  # type: ignore[arg-type]
        )

        self.assertEqual(count, 0)
        self.assertTrue(np.any(with_ball != without_ball))
        np.testing.assert_array_equal(untrusted_with, untrusted_without)
        off_pitch = BallObservation((5000.0, 300.0), (4996.0, 296.0, 5004.0, 304.0), 0.8)
        self.assertIsNone(
            project_measured_ball(
                pitch=SoccerPitch(),
                calibration=trusted_calibration(),
                ball=off_pitch,
            )
        )
        off_pitch_panel, _ = render_tactical_panel(
            calibration=trusted_calibration(),
            ball=off_pitch,
            **arguments,  # type: ignore[arg-type]
        )
        np.testing.assert_array_equal(off_pitch_panel, without_ball)

    def test_stacks_pitch_beneath_replay(self) -> None:
        replay = np.zeros((96, 160, 3), dtype=np.uint8)
        panel = np.zeros((80, 160, 3), dtype=np.uint8)

        combined = stack_replay(replay, panel)

        self.assertEqual(combined.shape, (176, 160, 3))


if __name__ == "__main__":
    unittest.main()
