from __future__ import annotations

import unittest

import numpy as np

from football_tracker.scene import SceneChangeDetector


class SceneChangeDetectorTests(unittest.TestCase):
    def test_first_and_identical_frames_are_not_cuts(self) -> None:
        detector = SceneChangeDetector(threshold=0.65)
        frame = np.full((180, 320, 3), (30, 120, 30), dtype=np.uint8)

        self.assertFalse(detector.update(frame).is_cut)
        self.assertFalse(detector.update(frame.copy()).is_cut)

    def test_large_colour_change_is_a_cut(self) -> None:
        detector = SceneChangeDetector(threshold=0.65)
        detector.update(np.zeros((180, 320, 3), dtype=np.uint8))

        change = detector.update(np.full((180, 320, 3), 255, dtype=np.uint8))

        self.assertTrue(change.is_cut)
        self.assertGreaterEqual(change.score, 0.65)

    def test_broadcast_composition_change_is_a_cut_at_default_threshold(self) -> None:
        detector = SceneChangeDetector()
        wide = np.full((180, 320, 3), (30, 130, 30), dtype=np.uint8)
        wide[:60] = (30, 30, 160)
        close = np.full((180, 320, 3), (30, 130, 30), dtype=np.uint8)
        detector.update(wide)

        change = detector.update(close)

        self.assertTrue(change.is_cut)
        self.assertGreaterEqual(change.score, 0.25)

    def test_cut_cooldown_avoids_immediate_second_reset(self) -> None:
        detector = SceneChangeDetector(threshold=0.65, minimum_gap_frames=3)
        detector.update(np.zeros((180, 320, 3), dtype=np.uint8))
        self.assertTrue(
            detector.update(np.full((180, 320, 3), 255, dtype=np.uint8)).is_cut
        )
        self.assertFalse(detector.update(np.zeros((180, 320, 3), dtype=np.uint8)).is_cut)


if __name__ == "__main__":
    unittest.main()

