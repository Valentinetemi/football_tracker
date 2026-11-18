from __future__ import annotations

import unittest

import numpy as np

from football_tracker.classification import (
    ShirtTeamClassifier,
    TeamPrediction,
    TrackTeamSmoother,
    extract_shirt_feature,
    role_from_class_name,
)


class ShirtFeatureTests(unittest.TestCase):
    def test_extracts_non_grass_torso_colour(self) -> None:
        frame = np.full((120, 80, 3), (30, 150, 30), dtype=np.uint8)
        frame[30:65, 25:55] = (20, 20, 220)

        feature = extract_shirt_feature(frame, np.asarray([15, 10, 65, 110]))

        self.assertIsNotNone(feature)
        self.assertEqual(feature.shape, (9,))

    def test_rejects_tiny_box(self) -> None:
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        self.assertIsNone(extract_shirt_feature(frame, np.asarray([1, 1, 4, 8])))


class TeamClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        generator = np.random.default_rng(7)
        self.team_a = [
            np.full(9, 0.2, dtype=np.float32) + generator.normal(0, 0.01, 9)
            for _ in range(12)
        ]
        self.team_b = [
            np.full(9, 0.8, dtype=np.float32) + generator.normal(0, 0.01, 9)
            for _ in range(12)
        ]

    def test_clusters_two_teams_and_rejects_ambiguous_feature(self) -> None:
        classifier = ShirtTeamClassifier(minimum_samples=8, uncertainty_threshold=0.35)
        report = classifier.fit(self.team_a + self.team_b)

        first = classifier.predict_feature(np.full(9, 0.21, dtype=np.float32))
        second = classifier.predict_feature(np.full(9, 0.79, dtype=np.float32))
        ambiguous = classifier.predict_feature(np.full(9, 0.5, dtype=np.float32))

        self.assertTrue(report.ready)
        self.assertIn(first.team, (0, 1))
        self.assertIn(second.team, (0, 1))
        self.assertNotEqual(first.team, second.team)
        self.assertIsNone(ambiguous.team)
        self.assertEqual(ambiguous.label, "unknown")

    def test_insufficient_samples_stays_unknown(self) -> None:
        classifier = ShirtTeamClassifier(minimum_samples=8)
        report = classifier.fit(self.team_a[:3])

        prediction = classifier.predict_feature(self.team_a[0])

        self.assertFalse(report.ready)
        self.assertIsNone(prediction.team)


class TrackSmoothingTests(unittest.TestCase):
    def test_waits_for_repeated_votes(self) -> None:
        smoother = TrackTeamSmoother(minimum_votes=3)

        self.assertIsNone(smoother.update(4, TeamPrediction(0, 0.8, "raw")).team)
        self.assertIsNone(smoother.update(4, TeamPrediction(0, 0.9, "raw")).team)
        self.assertEqual(smoother.update(4, TeamPrediction(0, 0.7, "raw")).team, 0)

    def test_roles_are_not_collapsed_into_team_players(self) -> None:
        self.assertEqual(role_from_class_name("player"), "player")
        self.assertEqual(role_from_class_name("goalkeeper"), "goalkeeper")
        self.assertEqual(role_from_class_name("referee"), "referee")
        self.assertEqual(role_from_class_name("something-new"), "unknown")


if __name__ == "__main__":
    unittest.main()
