"""Pretrained Ultralytics detector adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import supervision as sv

from football_tracker.ball import BALL_CLASS_NAMES

TRACKABLE_CLASS_NAMES = {
    "person",
    "player",
    "football player",
    "soccer player",
    "goalkeeper",
    "goalie",
    "referee",
    "official",
}


def normalize_class_names(names: Mapping[int, str] | Sequence[str]) -> dict[int, str]:
    """Return model class names in the mapping form used by the pipeline."""
    if isinstance(names, Mapping):
        return {int(class_id): str(name) for class_id, name in names.items()}
    return {class_id: str(name) for class_id, name in enumerate(names)}


def infer_trackable_class_ids(names: Mapping[int, str] | Sequence[str]) -> tuple[int, ...]:
    """Infer football people classes while deliberately excluding the ball."""
    normalized = normalize_class_names(names)
    return tuple(
        class_id
        for class_id, name in normalized.items()
        if name.strip().lower() in TRACKABLE_CLASS_NAMES
    )


def infer_ball_class_ids(names: Mapping[int, str] | Sequence[str]) -> tuple[int, ...]:
    """Infer ball classes supported by the already-selected checkpoint."""
    normalized = normalize_class_names(names)
    return tuple(
        class_id
        for class_id, name in normalized.items()
        if name.strip().lower() in BALL_CLASS_NAMES
    )


def resolve_device(requested: str) -> str:
    """Resolve ``auto`` to MPS, CUDA, or CPU in that order."""
    if requested != "auto":
        return requested

    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class UltralyticsDetector:
    """Convert Ultralytics predictions into Supervision detections."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str,
        confidence: float,
        image_size: int,
        class_ids: tuple[int, ...] | None,
        include_ball: bool = False,
        ball_confidence: float | None = None,
        model: Any | None = None,
    ) -> None:
        if model is None:
            try:
                from ultralytics import YOLO

                model = YOLO(model_name)
            except Exception as exc:
                raise RuntimeError(f"Could not load detection model {model_name!r}: {exc}") from exc

        self.model = model
        self.device = resolve_device(device)
        self.people_confidence = confidence
        self.image_size = image_size
        self.class_names = normalize_class_names(self.model.names)
        people_class_ids = class_ids or infer_trackable_class_ids(self.class_names)
        if not people_class_ids:
            available = ", ".join(
                f"{class_id}:{name}" for class_id, name in self.class_names.items()
            )
            raise ValueError(
                "Could not infer player/person classes from the model. "
                f"Pass --classes explicitly. Available classes: {available}"
            )

        unknown = sorted(set(people_class_ids) - self.class_names.keys())
        if unknown:
            raise ValueError(f"Unknown model class IDs: {unknown}")

        ball_class_ids = infer_ball_class_ids(self.class_names) if include_ball else ()
        if include_ball and not ball_class_ids:
            available = ", ".join(
                f"{class_id}:{name}" for class_id, name in self.class_names.items()
            )
            raise ValueError(
                "Ball tracking was requested, but the selected checkpoint has no ball class. "
                f"Available classes: {available}"
            )
        self.ball_class_ids = ball_class_ids
        self.class_ids = tuple(dict.fromkeys((*people_class_ids, *ball_class_ids)))
        self.confidence = min(confidence, ball_confidence or confidence)

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """Detect trackable people in one BGR frame."""
        try:
            result = self.model.predict(
                source=frame,
                conf=self.confidence,
                imgsz=self.image_size,
                classes=list(self.class_ids),
                device=self.device,
                verbose=False,
            )[0]
        except Exception as exc:
            raise RuntimeError(f"Model inference failed on device {self.device!r}: {exc}") from exc
        return sv.Detections.from_ultralytics(result)

    def class_name(self, class_id: int) -> str:
        return self.class_names.get(int(class_id), f"class-{class_id}")

