from __future__ import annotations

import unittest

import numpy as np

from football_tracker.annotation import (
    BALL_COLOR_BGR,
    build_track_label,
    draw_ball_marker,
    place_label,
)
from football_tracker.ball import BallObservation
from football_tracker.classification import EntityAssignment
from football_tracker.identity import PlayerSuggestion


def overlaps(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> bool:
    return not (
        first[2] <= second[0]
        or second[2] <= first[0]
        or first[3] <= second[1]
        or second[3] <= first[1]
    )


class LabelLayoutTests(unittest.TestCase):
    def test_chooses_an_alternate_position_for_nearby_labels(self) -> None:
        first = place_label((100, 100, 130, 180), (80, 20), (400, 240), [])
        second = place_label((110, 102, 140, 182), (80, 20), (400, 240), [first])

        self.assertFalse(overlaps(first, second))

    def test_keeps_label_inside_frame(self) -> None:
        label = place_label((2, 2, 15, 40), (100, 20), (120, 80), [])

        self.assertGreaterEqual(label[0], 0)
        self.assertGreaterEqual(label[1], 0)
        self.assertLessEqual(label[2], 120)
        self.assertLessEqual(label[3], 80)

    def test_ball_marker_is_present_only_for_a_reliable_observation(self) -> None:
        blank = np.zeros((100, 100, 3), dtype=np.uint8)
        hidden = blank.copy()
        visible = blank.copy()

        draw_ball_marker(hidden, None)
        draw_ball_marker(
            visible,
            BallObservation((50.0, 50.0), (47.0, 47.0, 53.0, 53.0), 0.8),
        )

        self.assertFalse(np.any(hidden))
        self.assertEqual(tuple(int(value) for value in visible[50, 50]), BALL_COLOR_BGR)


class IdentityLabelTests(unittest.TestCase):
    def test_existing_team_label_is_unchanged_without_roster(self) -> None:
        assignment = EntityAssignment("player", 0, "team-a", 0.9, "test")

        label = build_track_label(
            7,
            object_name="team-a",
            assignment=assignment,
            suggestion=None,
            roster_enabled=False,
        )

        self.assertEqual(label, "T7 team-a")

    def test_roster_mode_shows_only_id_until_suggestion_is_strong(self) -> None:
        assignment = EntityAssignment("player", 0, "team-a", 0.9, "test")

        unresolved = build_track_label(
            7,
            object_name="team-a",
            assignment=assignment,
            suggestion=None,
            roster_enabled=True,
        )
        resolved = build_track_label(
            7,
            object_name="team-a",
            assignment=assignment,
            suggestion=PlayerSuggestion(7, "team-a", 10, "Ada", 0.8, 3, 3),
            roster_enabled=True,
        )

        self.assertEqual(unresolved, "T7")
        self.assertEqual(resolved, "T7 Ada?")

    def test_official_role_remains_explicit_in_roster_mode(self) -> None:
        assignment = EntityAssignment("referee", None, "referee", 1.0, "test")

        label = build_track_label(
            9,
            object_name="referee",
            assignment=assignment,
            suggestion=None,
            roster_enabled=True,
        )

        self.assertEqual(label, "T9 referee")


if __name__ == "__main__":
    unittest.main()
