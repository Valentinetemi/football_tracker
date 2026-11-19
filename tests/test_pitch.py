from __future__ import annotations

import unittest

import cv2
import numpy as np

from football_tracker.pitch import PitchCalibrator, PitchKeypoints, SoccerPitch


def synthetic_observation() -> tuple[PitchKeypoints, np.ndarray]:
    pitch = SoccerPitch()
    pitch_corners = np.asarray(
        [[0, 0], [pitch.length, 0], [pitch.length, pitch.width], [0, pitch.width]],
        dtype=np.float32,
    )
    image_corners = np.asarray(
        [[140, 100], [910, 80], [850, 620], [180, 650]],
        dtype=np.float32,
    )
    pitch_to_image = cv2.getPerspectiveTransform(pitch_corners, image_corners)
    image_points = cv2.perspectiveTransform(
        pitch.vertices.reshape(-1, 1, 2),
        pitch_to_image,
    ).reshape(-1, 2)
    confidence = np.full(len(image_points), 0.95, dtype=np.float32)
    return PitchKeypoints(image_points, confidence), pitch_to_image


class PitchCalibrationTests(unittest.TestCase):
    def test_recovers_synthetic_pitch_projection(self) -> None:
        observation, pitch_to_image = synthetic_observation()
        calibrator = PitchCalibrator()

        calibration = calibrator.update(observation, frame_size=(1000, 720))

        world_point = np.asarray([[7350.0, 2800.0]], dtype=np.float32)
        image_point = cv2.perspectiveTransform(
            world_point.reshape(-1, 1, 2), pitch_to_image
        ).reshape(-1, 2)
        self.assertTrue(calibration.trusted, calibration.reason)
        np.testing.assert_allclose(calibration.project(image_point), world_point, atol=1.0)
        self.assertEqual(calibration.inlier_count, 32)

    def test_rejects_too_few_reliable_landmarks(self) -> None:
        observation, _ = synthetic_observation()
        confidence = np.zeros_like(observation.confidence)
        confidence[:5] = 0.95

        calibration = PitchCalibrator().update(
            PitchKeypoints(observation.xy, confidence),
            frame_size=(1000, 720),
        )

        self.assertFalse(calibration.trusted)
        self.assertIn("only 5", calibration.reason)

    def test_rejects_landmarks_without_spatial_coverage(self) -> None:
        observation, _ = synthetic_observation()
        collapsed = observation.xy.copy()
        collapsed[:, 0] = 500 + np.linspace(-2, 2, len(collapsed))
        collapsed[:, 1] = 300 + np.linspace(-1, 1, len(collapsed))

        calibration = PitchCalibrator().update(
            PitchKeypoints(collapsed, observation.confidence),
            frame_size=(1000, 720),
        )

        self.assertFalse(calibration.trusted)
        self.assertIn("too little of the image", calibration.reason)

    def test_cut_invalidation_discards_a_trusted_transform(self) -> None:
        observation, _ = synthetic_observation()
        calibrator = PitchCalibrator()
        self.assertTrue(
            calibrator.update(observation, frame_size=(1000, 720)).trusted
        )

        invalid = calibrator.invalidate("camera cut")

        self.assertFalse(invalid.trusted)
        self.assertIsNone(invalid.image_to_pitch)
        self.assertEqual(invalid.reason, "camera cut")


if __name__ == "__main__":
    unittest.main()
