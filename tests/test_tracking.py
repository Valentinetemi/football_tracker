from __future__ import annotations

import unittest

import numpy as np
import supervision as sv

from football_tracker.tracking import TrackManager


class FakeTracker:
    def __init__(self) -> None:
        self.next_ids: list[int] = []
        self.reset_calls = 0

    def update_with_detections(self, detections: sv.Detections) -> sv.Detections:
        detections.tracker_id = np.asarray(self.next_ids, dtype=int)
        return detections

    def reset(self) -> None:
        self.reset_calls += 1


def detections(count: int) -> sv.Detections:
    return sv.Detections(
        xyxy=np.asarray([[0, 0, 10, 20]] * count, dtype=float).reshape(count, 4),
        confidence=np.full(count, 0.9, dtype=float),
        class_id=np.zeros(count, dtype=int),
    )


class TrackManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tracker = FakeTracker()
        self.manager = TrackManager(
            frame_rate=25.0,
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_consecutive_frames=2,
            tracker_factory=lambda: self.tracker,
        )

    def test_preserves_display_id_for_same_local_track(self) -> None:
        self.tracker.next_ids = [7]
        first = self.manager.update(detections(1))
        second = self.manager.update(detections(1))

        self.assertEqual(first.tracker_id.tolist(), [1])
        self.assertEqual(second.tracker_id.tolist(), [1])

    def test_reset_creates_new_segment_and_never_reuses_display_id(self) -> None:
        self.tracker.next_ids = [1]
        before_cut = self.manager.update(detections(1))
        self.manager.reset_for_camera_cut()
        after_cut = self.manager.update(detections(1))

        self.assertEqual(before_cut.tracker_id.tolist(), [1])
        self.assertEqual(after_cut.tracker_id.tolist(), [2])
        self.assertEqual(after_cut.data["track_segment"].tolist(), [2])
        self.assertEqual(self.tracker.reset_calls, 1)
        self.assertEqual(self.manager.reset_count, 1)


if __name__ == "__main__":
    unittest.main()
