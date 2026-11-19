from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from football_tracker.classification import EntityAssignment
from football_tracker.identity import (
    EasyOCRJerseyNumberOCR,
    JerseyRead,
    JerseySuggestionEngine,
    Roster,
    RosterEntry,
    extract_jersey_crop,
)


class SequenceOCR:
    def __init__(self, reads: list[JerseyRead]) -> None:
        self.reads = iter(reads)
        self.calls = 0

    def read(self, crop: np.ndarray) -> JerseyRead:
        self.calls += 1
        return next(self.reads)


class FakeEasyOCRReader:
    def readtext(self, image: np.ndarray, **kwargs: object) -> list[tuple[object, str, float]]:
        return [([], "7", 0.51), ([], "10", 0.87), ([], "100", 0.99)]


def detections(track_id: int = 7) -> sv.Detections:
    return sv.Detections(
        xyxy=np.asarray([[20, 10, 100, 190]], dtype=np.float32),
        class_id=np.asarray([2]),
        tracker_id=np.asarray([track_id]),
    )


def player_assignment(team: int | None) -> list[EntityAssignment]:
    label = "unknown" if team is None else f"team-{chr(ord('a') + team)}"
    return [EntityAssignment("player", team, label, 0.9, "test")]


class RosterTests(unittest.TestCase):
    def test_same_number_maps_independently_for_each_team(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "roster.csv"
            path.write_text(
                "team,number,name\nteam-a,10,Ada\nteam-b,10,Bea\n",
                encoding="utf-8",
            )

            roster = Roster.from_csv(path)

        self.assertEqual(roster.lookup("team-a", 10).name, "Ada")  # type: ignore[union-attr]
        self.assertEqual(roster.lookup("team-b", 10).name, "Bea")  # type: ignore[union-attr]

    def test_rejects_implicit_team_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "roster.csv"
            path.write_text(
                "team,number,name\nhome,7,Ada\nhome,7,Bea\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "team-a or team-b"):
                Roster.from_csv(path)

    def test_rejects_duplicate_number_within_one_team(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "roster.csv"
            path.write_text(
                "team,number,name\nteam-a,7,Ada\nteam-a,7,Bea\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Duplicate roster number"):
                Roster.from_csv(path)

    def test_easyocr_adapter_keeps_best_one_or_two_digit_result(self) -> None:
        reader = FakeEasyOCRReader()
        ocr = EasyOCRJerseyNumberOCR(
            device="cpu",
            model_directory=Path("unused"),
            reader=reader,
        )
        crop = np.zeros((80, 40, 3), dtype=np.uint8)

        result = ocr.read(crop)

        self.assertEqual(result.number, 10)
        self.assertAlmostEqual(result.confidence, 0.87)


class JerseySuggestionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = np.zeros((220, 140, 3), dtype=np.uint8)
        cv2.putText(
            self.frame,
            "10",
            (40, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        self.roster = Roster(
            [RosterEntry("team-a", 10, "Ada"), RosterEntry("team-b", 10, "Bea")]
        )

    def make_engine(self, reads: list[JerseyRead]) -> JerseySuggestionEngine:
        return JerseySuggestionEngine(
            self.roster,
            SequenceOCR(reads),
            sample_interval=1,
            minimum_votes=3,
            minimum_player_height=80,
            minimum_sharpness=0,
        )

    def test_requires_consistent_reads_before_roster_suggestion(self) -> None:
        engine = self.make_engine(
            [JerseyRead(10, 0.9), JerseyRead(10, 0.8), JerseyRead(10, 0.85)]
        )

        outputs = []
        for frame_index in range(3):
            outputs.append(
                engine.update(
                    frame_index=frame_index,
                    frame=self.frame,
                    detections=detections(),
                    assignments=player_assignment(0),
                )[0]
            )

        self.assertIsNone(outputs[0])
        self.assertIsNone(outputs[1])
        self.assertEqual(outputs[2].name, "Ada")  # type: ignore[union-attr]
        self.assertEqual(outputs[2].number, 10)  # type: ignore[union-attr]

    def test_conflicting_reads_remain_unresolved(self) -> None:
        engine = self.make_engine(
            [
                JerseyRead(10, 0.9),
                JerseyRead(11, 0.9),
                JerseyRead(10, 0.9),
                JerseyRead(11, 0.9),
            ]
        )

        output = None
        for frame_index in range(4):
            output = engine.update(
                frame_index=frame_index,
                frame=self.frame,
                detections=detections(),
                assignments=player_assignment(0),
            )[0]

        self.assertIsNone(output)

    def test_unknown_team_skips_ocr_and_cut_requires_fresh_votes(self) -> None:
        ocr = SequenceOCR(
            [
                JerseyRead(10, 0.9),
                JerseyRead(10, 0.9),
                JerseyRead(10, 0.9),
                JerseyRead(10, 0.9),
                JerseyRead(10, 0.9),
                JerseyRead(10, 0.9),
            ]
        )
        engine = JerseySuggestionEngine(
            self.roster,
            ocr,
            sample_interval=1,
            minimum_votes=3,
            minimum_player_height=80,
            minimum_sharpness=0,
        )
        unknown = engine.update(
            frame_index=0,
            frame=self.frame,
            detections=detections(),
            assignments=player_assignment(None),
        )[0]
        self.assertIsNone(unknown)
        self.assertEqual(ocr.calls, 0)

        suggestion = None
        for frame_index in range(1, 4):
            suggestion = engine.update(
                frame_index=frame_index,
                frame=self.frame,
                detections=detections(),
                assignments=player_assignment(0),
            )[0]
        self.assertIsNotNone(suggestion)

        engine.reset_tracks()
        after_cut = engine.update(
            frame_index=4,
            frame=self.frame,
            detections=detections(20),
            assignments=player_assignment(0),
        )[0]
        self.assertIsNone(after_cut)

    def test_clear_crop_gate_rejects_distant_player(self) -> None:
        crop, quality = extract_jersey_crop(
            self.frame,
            np.asarray([20, 10, 60, 70]),
            minimum_player_height=120,
            minimum_sharpness=0,
        )

        self.assertIsNone(crop)
        self.assertFalse(quality.clear)
        self.assertIn("too small", quality.reason)


if __name__ == "__main__":
    unittest.main()
