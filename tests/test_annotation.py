from __future__ import annotations

import unittest

from football_tracker.annotation import place_label


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


if __name__ == "__main__":
    unittest.main()
