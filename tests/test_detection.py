from __future__ import annotations

import unittest

from football_tracker.detection import (
    UltralyticsDetector,
    infer_ball_class_ids,
    infer_trackable_class_ids,
)


class FakeModel:
    names = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}


class DetectionTests(unittest.TestCase):
    def test_infers_football_people_but_not_ball(self) -> None:
        self.assertEqual(
            infer_trackable_class_ids(FakeModel.names),
            (1, 2, 3),
        )

    def test_infers_coco_person(self) -> None:
        self.assertEqual(infer_trackable_class_ids({0: "person", 32: "sports ball"}), (0,))

    def test_infers_ball_from_football_and_coco_checkpoints(self) -> None:
        self.assertEqual(infer_ball_class_ids(FakeModel.names), (0,))
        self.assertEqual(infer_ball_class_ids({0: "person", 32: "sports ball"}), (32,))

    def test_ball_tracking_reuses_the_selected_checkpoint(self) -> None:
        detector = UltralyticsDetector(
            "unused.pt",
            device="cpu",
            confidence=0.25,
            image_size=640,
            class_ids=None,
            include_ball=True,
            ball_confidence=0.15,
            model=FakeModel(),
        )

        self.assertEqual(detector.class_ids, (1, 2, 3, 0))
        self.assertEqual(detector.ball_class_ids, (0,))
        self.assertEqual(detector.confidence, 0.15)

    def test_explicit_classes_override_inference(self) -> None:
        detector = UltralyticsDetector(
            "unused.pt",
            device="cpu",
            confidence=0.25,
            image_size=640,
            class_ids=(2,),
            model=FakeModel(),
        )

        self.assertEqual(detector.class_ids, (2,))
        self.assertEqual(detector.class_name(2), "player")

    def test_unknown_explicit_class_fails_early(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown model class IDs"):
            UltralyticsDetector(
                "unused.pt",
                device="cpu",
                confidence=0.25,
                image_size=640,
                class_ids=(99,),
                model=FakeModel(),
            )


if __name__ == "__main__":
    unittest.main()

