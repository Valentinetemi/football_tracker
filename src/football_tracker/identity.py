"""Replaceable jersey OCR and conservative roster-based name suggestions."""

from __future__ import annotations

import csv
import re
from collections import Counter, deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
import supervision as sv

from football_tracker.classification import EntityAssignment
from football_tracker.detection import resolve_device

ANONYMOUS_TEAMS = {"team-a", "team-b"}


@dataclass(frozen=True, slots=True)
class RosterEntry:
    team: str
    number: int
    name: str


class Roster:
    """An explicit anonymous-team and shirt-number lookup."""

    def __init__(self, entries: Sequence[RosterEntry]) -> None:
        self.entries = tuple(entries)
        self._by_team_number: dict[tuple[str, int], RosterEntry] = {}
        for entry in self.entries:
            key = (entry.team, entry.number)
            if key in self._by_team_number:
                raise ValueError(
                    f"Duplicate roster number {entry.number} for {entry.team}."
                )
            self._by_team_number[key] = entry

    @classmethod
    def from_csv(cls, path: Path) -> Roster:
        """Load ``team,number,name`` rows; team must be team-a or team-b."""
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                headers = set(reader.fieldnames or ())
                required = {"team", "number", "name"}
                if not required.issubset(headers):
                    missing = ", ".join(sorted(required - headers))
                    raise ValueError(f"Roster CSV is missing columns: {missing}")

                entries: list[RosterEntry] = []
                for row_number, row in enumerate(reader, start=2):
                    team = (row.get("team") or "").strip().lower()
                    number_text = (row.get("number") or "").strip()
                    name = (row.get("name") or "").strip()
                    if team not in ANONYMOUS_TEAMS:
                        raise ValueError(
                            f"Roster row {row_number} team must be team-a or team-b."
                        )
                    if not number_text.isdigit() or not 0 <= int(number_text) <= 99:
                        raise ValueError(
                            f"Roster row {row_number} number must be an integer from 0 to 99."
                        )
                    if not name:
                        raise ValueError(f"Roster row {row_number} name is empty.")
                    entries.append(RosterEntry(team, int(number_text), name))
        except OSError as exc:
            raise ValueError(f"Could not read roster CSV {path}: {exc}") from exc
        if not entries:
            raise ValueError("Roster CSV contains no player rows.")
        return cls(entries)

    def lookup(self, team: str, number: int) -> RosterEntry | None:
        return self._by_team_number.get((team, number))


@dataclass(frozen=True, slots=True)
class JerseyRead:
    number: int | None
    confidence: float
    raw_text: str = ""


class JerseyNumberOCR(Protocol):
    """Swappable single-crop OCR boundary."""

    def read(self, crop: np.ndarray) -> JerseyRead: ...


class EasyOCRJerseyNumberOCR:
    """General scene-text baseline; a soccer-specific model can replace this class."""

    def __init__(
        self,
        *,
        device: str,
        model_directory: Path,
        reader: Any | None = None,
    ) -> None:
        if reader is None:
            try:
                import easyocr
            except ImportError as exc:
                raise RuntimeError(
                    "Jersey OCR requires the optional EasyOCR dependencies. "
                    "Install them with: python -m pip install -e '.[ocr]'"
                ) from exc
            resolved_device = resolve_device(device)
            model_directory.mkdir(parents=True, exist_ok=True)
            try:
                reader = easyocr.Reader(
                    ["en"],
                    gpu=resolved_device,
                    model_storage_directory=str(model_directory),
                    verbose=False,
                    quantize=resolved_device == "cpu",
                )
            except Exception as exc:
                raise RuntimeError(f"Could not initialize EasyOCR: {exc}") from exc
        self.reader = reader

    @staticmethod
    def _prepare(crop: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        target_height = 180
        if gray.shape[0] < target_height:
            scale = min(4.0, target_height / max(1, gray.shape[0]))
            gray = cv2.resize(
                gray,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
        return cv2.copyMakeBorder(gray, 12, 12, 12, 12, cv2.BORDER_REPLICATE)

    def read(self, crop: np.ndarray) -> JerseyRead:
        prepared = self._prepare(crop)
        try:
            results = self.reader.readtext(
                prepared,
                allowlist="0123456789",
                detail=1,
                paragraph=False,
                min_size=8,
                text_threshold=0.45,
                low_text=0.25,
                mag_ratio=1.5,
            )
        except Exception as exc:
            raise RuntimeError(f"Jersey OCR inference failed: {exc}") from exc

        candidates: list[JerseyRead] = []
        for result in results:
            if len(result) < 3:
                continue
            raw_text = str(result[1]).strip()
            digits = re.sub(r"\D", "", raw_text)
            confidence = float(result[2])
            if 1 <= len(digits) <= 2 and 0 <= int(digits) <= 99:
                candidates.append(JerseyRead(int(digits), confidence, raw_text))
        if not candidates:
            return JerseyRead(None, 0.0)
        return max(candidates, key=lambda candidate: candidate.confidence)


@dataclass(frozen=True, slots=True)
class CropQuality:
    clear: bool
    player_height: float
    crop_width: int
    crop_height: int
    sharpness: float
    reason: str


def extract_jersey_crop(
    frame: np.ndarray,
    box: np.ndarray,
    *,
    minimum_player_height: int,
    minimum_sharpness: float,
) -> tuple[np.ndarray | None, CropQuality]:
    """Crop the central torso and reject physically unreadable candidates."""
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = (float(value) for value in box)
    player_height = y2 - y1
    player_width = x2 - x1
    if player_height < minimum_player_height:
        return None, CropQuality(
            False,
            player_height,
            0,
            0,
            0.0,
            "player crop is too small",
        )

    crop_x1 = int(np.clip(round(x1 + player_width * 0.08), 0, frame_width - 1))
    crop_x2 = int(np.clip(round(x2 - player_width * 0.08), 0, frame_width))
    crop_y1 = int(np.clip(round(y1 + player_height * 0.08), 0, frame_height - 1))
    crop_y2 = int(np.clip(round(y1 + player_height * 0.64), 0, frame_height))
    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.shape[0] < 24 or crop.shape[1] < 12:
        return None, CropQuality(
            False,
            player_height,
            crop.shape[1],
            crop.shape[0],
            0.0,
            "torso crop is too small",
        )

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if sharpness < minimum_sharpness:
        return None, CropQuality(
            False,
            player_height,
            crop.shape[1],
            crop.shape[0],
            sharpness,
            "torso crop is too blurred",
        )
    return crop, CropQuality(
        True,
        player_height,
        crop.shape[1],
        crop.shape[0],
        sharpness,
        "clear",
    )


@dataclass(frozen=True, slots=True)
class PlayerSuggestion:
    track_id: int
    team: str
    number: int
    name: str
    confidence: float
    votes: int
    total_reads: int

    @property
    def display_text(self) -> str:
        return f"{self.name}?"


@dataclass(slots=True)
class _TrackEvidence:
    team: str
    reads: deque[JerseyRead]
    suggestion: PlayerSuggestion | None = None


@dataclass(frozen=True, slots=True)
class JerseySuggestionSummary:
    clear_crops_sampled: int
    accepted_reads: int
    suggested_tracks: int
    unresolved_tracks: int


class JerseySuggestionEngine:
    """Accumulate clear-frame OCR reads and emit only roster-backed consensus."""

    def __init__(
        self,
        roster: Roster,
        ocr: JerseyNumberOCR,
        *,
        sample_interval: int = 5,
        minimum_votes: int = 3,
        minimum_support: float = 0.75,
        minimum_ocr_confidence: float = 0.35,
        minimum_player_height: int = 120,
        minimum_sharpness: float = 20.0,
        history_length: int = 12,
    ) -> None:
        self.roster = roster
        self.ocr = ocr
        self.sample_interval = sample_interval
        self.minimum_votes = minimum_votes
        self.minimum_support = minimum_support
        self.minimum_ocr_confidence = minimum_ocr_confidence
        self.minimum_player_height = minimum_player_height
        self.minimum_sharpness = minimum_sharpness
        self.history_length = history_length
        self._evidence: dict[int, _TrackEvidence] = {}
        self._seen_player_tracks: set[int] = set()
        self._suggested_tracks: set[int] = set()
        self._clear_crops_sampled = 0
        self._accepted_reads = 0

    @staticmethod
    def _team_label(assignment: EntityAssignment) -> str | None:
        if assignment.role != "player" or assignment.team not in (0, 1):
            return None
        return f"team-{chr(ord('a') + int(assignment.team))}"

    def _consensus(
        self,
        track_id: int,
        evidence: _TrackEvidence,
    ) -> PlayerSuggestion | None:
        if len(evidence.reads) < self.minimum_votes:
            return None
        counts = Counter(
            read.number for read in evidence.reads if read.number is not None
        )
        if not counts:
            return None
        number, votes = counts.most_common(1)[0]
        support = votes / len(evidence.reads)
        if votes < self.minimum_votes or support < self.minimum_support:
            return None
        winner_reads = [read for read in evidence.reads if read.number == number]
        mean_confidence = float(np.mean([read.confidence for read in winner_reads]))
        roster_entry = self.roster.lookup(evidence.team, int(number))
        if roster_entry is None:
            return None
        return PlayerSuggestion(
            track_id,
            evidence.team,
            int(number),
            roster_entry.name,
            mean_confidence * support,
            votes,
            len(evidence.reads),
        )

    def update(
        self,
        *,
        frame_index: int,
        frame: np.ndarray,
        detections: sv.Detections,
        assignments: list[EntityAssignment] | None,
    ) -> list[PlayerSuggestion | None]:
        suggestions: list[PlayerSuggestion | None] = [None] * len(detections)
        if assignments is None or detections.tracker_id is None:
            return suggestions

        for index, assignment in enumerate(assignments):
            track_id = int(detections.tracker_id[index])
            team = self._team_label(assignment)
            if assignment.role == "player":
                self._seen_player_tracks.add(track_id)
            if team is None:
                continue

            evidence = self._evidence.get(track_id)
            if evidence is None or evidence.team != team:
                evidence = _TrackEvidence(team, deque(maxlen=self.history_length))
                self._evidence[track_id] = evidence
            if evidence.suggestion is not None:
                suggestions[index] = evidence.suggestion
                continue
            if (frame_index + track_id) % self.sample_interval != 0:
                continue

            crop, quality = extract_jersey_crop(
                frame,
                detections.xyxy[index],
                minimum_player_height=self.minimum_player_height,
                minimum_sharpness=self.minimum_sharpness,
            )
            if crop is None or not quality.clear:
                continue
            self._clear_crops_sampled += 1
            read = self.ocr.read(crop)
            if read.number is None or read.confidence < self.minimum_ocr_confidence:
                continue
            evidence.reads.append(read)
            self._accepted_reads += 1
            evidence.suggestion = self._consensus(track_id, evidence)
            if evidence.suggestion is not None:
                self._suggested_tracks.add(track_id)
                suggestions[index] = evidence.suggestion
        return suggestions

    def reset_tracks(self) -> None:
        """Require new track evidence after a camera cut."""
        self._evidence.clear()

    def summary(self) -> JerseySuggestionSummary:
        return JerseySuggestionSummary(
            self._clear_crops_sampled,
            self._accepted_reads,
            len(self._suggested_tracks),
            len(self._seen_player_tracks - self._suggested_tracks),
        )
