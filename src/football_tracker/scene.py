"""Conservative camera-cut detection for tracker resets."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class SceneChange:
    is_cut: bool
    score: float


class SceneChangeDetector:
    """Compare compact colour histograms and pixels between adjacent frames."""

    def __init__(self, threshold: float = 0.65, minimum_gap_frames: int = 8) -> None:
        self.threshold = threshold
        self.minimum_gap_frames = minimum_gap_frames
        self._previous_histogram: np.ndarray | None = None
        self._previous_thumbnail: np.ndarray | None = None
        self._frame_index = -1
        self._last_cut_frame = -minimum_gap_frames

    @staticmethod
    def _features(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        thumbnail = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist(
            [hsv],
            channels=[0, 1, 2],
            mask=None,
            histSize=[16, 8, 8],
            ranges=[0, 180, 0, 256, 0, 256],
        )
        cv2.normalize(histogram, histogram, alpha=1.0, norm_type=cv2.NORM_L1)
        return histogram, thumbnail

    def update(self, frame: np.ndarray) -> SceneChange:
        """Return a cut decision and always retain this frame as the next baseline."""
        self._frame_index += 1
        histogram, thumbnail = self._features(frame)
        if self._previous_histogram is None or self._previous_thumbnail is None:
            self._previous_histogram = histogram
            self._previous_thumbnail = thumbnail
            return SceneChange(is_cut=False, score=0.0)

        histogram_distance = cv2.compareHist(
            self._previous_histogram,
            histogram,
            cv2.HISTCMP_BHATTACHARYYA,
        )
        pixel_distance = float(
            np.mean(
                cv2.absdiff(self._previous_thumbnail, thumbnail),
                dtype=np.float64,
            )
            / 255.0
        )
        score = float((0.8 * histogram_distance) + (0.2 * pixel_distance))
        gap_is_large_enough = self._frame_index - self._last_cut_frame >= self.minimum_gap_frames
        is_cut = score >= self.threshold and gap_is_large_enough
        if is_cut:
            self._last_cut_frame = self._frame_index

        self._previous_histogram = histogram
        self._previous_thumbnail = thumbnail
        return SceneChange(is_cut=is_cut, score=score)

    def reset(self) -> None:
        self._previous_histogram = None
        self._previous_thumbnail = None
        self._frame_index = -1
        self._last_cut_frame = -self.minimum_gap_frames

