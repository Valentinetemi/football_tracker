from __future__ import annotations

import unittest

import numpy as np
import supervision as sv

from football_tracker.ball import BallTracker, split_ball_detections


def candidates(
    boxes: list[list[float]],
    confidence: list[float] | None = None,
) -> sv.Detections:
    return sv.Detections(
        xyxy=np.asarray(boxes, dtype=np.float32).reshape((-1, 4)),
        confidence=np.asarray(confidence, dtype=np.float32) if confidence is not None else None,
        class_id=np.zeros(len(boxes), dtype=int),
    )


class BallTrackingTests(unittest.TestCase):
    def test_splits_ball_before_people_tracking_and_applies_thresholds(self) -> None:
        detections = sv.Detections(
            xyxy=np.asarray(
                [[10, 10, 20, 30], [40, 40, 45, 45], [50, 50, 55, 55], [70, 10, 80, 30]],
                dtype=np.float32,
            ),
            confidence=np.asarray([0.9, 0.8, 0.2, 0.7], dtype=np.float32),
            class_id=np.asarray([2, 0, 0, 3]),
        )
        names = {0: "ball", 2: "player", 3: "referee"}

        people, balls = split_ball_detections(
            detections,
            class_name=names.__getitem__,
            people_confidence=0.25,
            ball_confidence=0.35,
        )

        self.assertEqual(people.class_id.tolist(), [2, 3])  # type: ignore[union-attr]
        self.assertEqual(len(balls), 1)
        self.assertAlmostEqual(float(balls.confidence[0]), 0.8, places=5)  # type: ignore[index]

    def test_hides_first_hit_and_miss_then_reports_recovery(self) -> None:
        tracker = BallTracker(minimum_hits=2, maximum_gap=4, maximum_distance_ratio=0.2)

        first = tracker.update(
            frame_index=0,
            candidates=candidates([[10, 10, 14, 14]], [0.8]),
            frame_size=(200, 100),
        )
        reliable = tracker.update(
            frame_index=1,
            candidates=candidates([[14, 10, 18, 14]], [0.82]),
            frame_size=(200, 100),
        )
        missed = tracker.update(
            frame_index=2,
            candidates=candidates([]),
            frame_size=(200, 100),
        )
        recovery_first_hit = tracker.update(
            frame_index=3,
            candidates=candidates([[22, 10, 26, 14]], [0.84]),
            frame_size=(200, 100),
        )
        recovered = tracker.update(
            frame_index=4,
            candidates=candidates([[26, 10, 30, 14]], [0.85]),
            frame_size=(200, 100),
        )

        self.assertIsNone(first)
        self.assertIsNotNone(reliable)
        self.assertIsNone(missed)
        self.assertIsNone(recovery_first_hit)
        self.assertIsNotNone(recovered)
        self.assertEqual(tracker.summary().recoveries, 1)
        self.assertEqual(tracker.summary().missed_frames, 1)

    def test_rejects_two_equally_plausible_candidates(self) -> None:
        tracker = BallTracker(minimum_hits=1, maximum_distance_ratio=0.2)
        tracker.update(
            frame_index=0,
            candidates=candidates([[40, 40, 44, 44]], [0.8]),
            frame_size=(200, 100),
        )

        ambiguous = tracker.update(
            frame_index=1,
            candidates=candidates(
                [[43, 40, 47, 44], [45, 40, 49, 44]],
                [0.78, 0.77],
            ),
            frame_size=(200, 100),
        )

        self.assertIsNone(ambiguous)
        self.assertEqual(tracker.summary().ambiguous_frames, 1)

    def test_camera_cut_discards_motion_and_reliability(self) -> None:
        tracker = BallTracker(minimum_hits=2, maximum_distance_ratio=0.2)
        for frame_index in range(2):
            tracker.update(
                frame_index=frame_index,
                candidates=candidates([[10 + frame_index, 10, 14 + frame_index, 14]], [0.8]),
                frame_size=(200, 100),
            )

        tracker.reset_for_camera_cut()
        after_cut = tracker.update(
            frame_index=2,
            candidates=candidates([[150, 70, 154, 74]], [0.9]),
            frame_size=(200, 100),
        )

        self.assertIsNone(after_cut)
        self.assertEqual(tracker.summary().camera_cut_resets, 1)
        self.assertEqual(tracker.summary().recoveries, 0)


if __name__ == "__main__":
    unittest.main()
