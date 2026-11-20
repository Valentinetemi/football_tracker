"""Small, explicit OpenCV video reader and atomic MP4 writer."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    reported_frame_count: int


class VideoReader:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._capture: cv2.VideoCapture | None = None
        self.metadata: VideoMetadata | None = None

    def __enter__(self) -> VideoReader:
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"OpenCV could not open input video: {self.path}")

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0:
            capture.release()
            raise RuntimeError("Input video reports an invalid frame size.")
        if not np.isfinite(fps) or fps <= 0:
            capture.release()
            raise RuntimeError(
                "Input video does not report a usable frame rate; timing cannot be preserved."
            )

        self._capture = capture
        self.metadata = VideoMetadata(width, height, fps, max(0, frame_count))
        return self

    def __iter__(self):  # type: ignore[no-untyped-def]
        if self._capture is None:
            raise RuntimeError("VideoReader must be used as a context manager.")
        frame_index = 0
        while True:
            ok, frame = self._capture.read()
            if not ok:
                break
            yield frame_index, frame
            frame_index += 1

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._capture is not None:
            self._capture.release()


class VideoWriter:
    """Write to a temporary MP4 and replace the destination only on success."""

    def __init__(self, path: Path, metadata: VideoMetadata, codec: str) -> None:
        self.path = path
        self.metadata = metadata
        self.codec = codec
        self.frames_written = 0
        self._writer: cv2.VideoWriter | None = None
        self._temporary_path: Path | None = None

    def __enter__(self) -> VideoWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.stem}-",
            suffix=".partial.mp4",
            dir=self.path.parent,
        )
        os.close(file_descriptor)
        self._temporary_path = Path(temporary_name)
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        writer = cv2.VideoWriter(
            str(self._temporary_path),
            fourcc,
            self.metadata.fps,
            (self.metadata.width, self.metadata.height),
        )
        if not writer.isOpened():
            writer.release()
            self._temporary_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"OpenCV could not open an MP4 writer with codec {self.codec!r}. "
                "Try --codec avc1 or --codec mp4v. For a file the dashboard can play "
                "in a browser, prefer avc1 (H.264); mp4v often will not."
            )
        self._writer = writer
        return self

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            raise RuntimeError("VideoWriter must be used as a context manager.")
        height, width = frame.shape[:2]
        if (width, height) != (self.metadata.width, self.metadata.height):
            raise RuntimeError(
                f"Output frame size changed to {width}x{height}; "
                f"expected {self.metadata.width}x{self.metadata.height}."
            )
        self._writer.write(frame)
        self.frames_written += 1

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._writer is not None:
            self._writer.release()
        if self._temporary_path is None:
            return
        if exc_type is not None:
            self._temporary_path.unlink(missing_ok=True)
            return
        if self.frames_written == 0:
            self._temporary_path.unlink(missing_ok=True)
            raise RuntimeError("Input video decoded successfully but contained no frames.")
        os.replace(self._temporary_path, self.path)

