"""Prediction-independent OCR target realignment for GUI grounding.

The LLaDA-V GUI-grounding paper describes replacing inconsistent icon-level
target boxes with the OCR text region associated with the requested element,
but it does not publish the OCR engine or matching code.  This module provides
the deterministic matching part of an auditable approximation.  It deliberately
uses only the instruction target text, the source annotation, and OCR output;
model predictions never participate in target construction.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Any, Iterable, Mapping, Sequence


OCR_REALIGNMENT_VERSION = "mind2web-easyocr-linked-text-v1"
MINIMUM_OCR_CONFIDENCE = 0.20
MINIMUM_TEXT_SIMILARITY = 0.68
MAXIMUM_EDGE_DISTANCE = 0.22
OCR_MATCH_CONFIG = {
    "minimum_ocr_confidence": MINIMUM_OCR_CONFIDENCE,
    "minimum_text_similarity": MINIMUM_TEXT_SIMILARITY,
    "maximum_edge_distance_normalized": MAXIMUM_EDGE_DISTANCE,
    "near_exact_similarity": 0.92,
    "near_exact_distance_multiplier": 1.5,
    "score_weights": {
        "text_similarity": 0.72,
        "spatial_proximity": 0.18,
        "ocr_confidence": 0.07,
        "source_iou": 0.03,
    },
    "accepted_box_policy": "replace DOM box with matched OCR detection xyxy",
    "rejection_policy": "retain original DOM box",
    "prediction_independent": True,
}
_WORD_RE = re.compile(r"[a-z0-9]+")
_GUI_ACTION_RE = re.compile(
    r"^(lclick|hover|type_in)\s+\[\s*[-+0-9., ]+\s*\](.*)$", re.DOTALL
)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize_ocr_text(value: Any) -> str:
    """Normalize visible text without silently translating or stemming it."""

    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(_WORD_RE.findall(text))


def token_f1(left: str, right: str) -> float:
    left_tokens = left.split()
    right_tokens = right.split()
    if not left_tokens or not right_tokens:
        return 0.0
    left_counts: dict[str, int] = {}
    right_counts: dict[str, int] = {}
    for token in left_tokens:
        left_counts[token] = left_counts.get(token, 0) + 1
    for token in right_tokens:
        right_counts[token] = right_counts.get(token, 0) + 1
    overlap = sum(
        min(count, right_counts.get(token, 0)) for token, count in left_counts.items()
    )
    if overlap == 0:
        return 0.0
    precision = overlap / len(right_tokens)
    recall = overlap / len(left_tokens)
    return 2.0 * precision * recall / (precision + recall)


def text_similarity(target: Any, candidate: Any) -> float:
    """Return a conservative similarity score for target/OCR text."""

    target_text = normalize_ocr_text(target)
    candidate_text = normalize_ocr_text(candidate)
    if not target_text or not candidate_text:
        return 0.0
    if target_text == candidate_text:
        return 1.0

    sequence = SequenceMatcher(None, target_text, candidate_text, autojunk=False).ratio()
    tokens = token_f1(target_text, candidate_text)
    containment = 0.0
    shorter, longer = sorted((target_text, candidate_text), key=len)
    if len(shorter) >= 3 and shorter in longer:
        # A short but exact rendered fragment (for example, "Downloads" from a
        # longer accessibility label) is strong evidence, without treating a
        # one-character OCR fragment as a match.
        coverage = len(shorter) / len(longer)
        containment = 0.82 + 0.18 * coverage
    return min(1.0, max(sequence, tokens, containment))


def replace_action_bbox(answer: Any, bbox_1000: Iterable[Any]) -> str:
    """Replace only the coordinate field while preserving action/value text."""

    match = _GUI_ACTION_RE.fullmatch(str(answer or ""))
    if match is None:
        raise ValueError(f"invalid GUI action string: {answer!r}")
    values = [int(value) for value in bbox_1000]
    if len(values) != 4:
        raise ValueError("GUI action bbox must contain four coordinates")
    coords = ",".join(str(value) for value in values)
    return f"{match.group(1)} [{coords}]{match.group(2)}"


def valid_bbox(values: Sequence[Any]) -> tuple[float, float, float, float]:
    if len(values) != 4:
        raise ValueError("bbox must contain four coordinates")
    x1, y1, x2, y2 = (float(value) for value in values)
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        raise ValueError("bbox coordinates must be finite")
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox must have positive area")
    return x1, y1, x2, y2


def polygon_bbox(points: Sequence[Sequence[Any]]) -> tuple[float, float, float, float]:
    if len(points) < 3:
        raise ValueError("OCR polygon must contain at least three points")
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return valid_bbox((min(xs), min(ys), max(xs), max(ys)))


def bbox_edge_distance(
    left: Sequence[Any], right: Sequence[Any], image_width: float, image_height: float
) -> float:
    """Return the minimum box-to-box distance normalized by image diagonal."""

    left_x1, left_y1, left_x2, left_y2 = valid_bbox(left)
    right_x1, right_y1, right_x2, right_y2 = valid_bbox(right)
    dx = max(left_x1 - right_x2, right_x1 - left_x2, 0.0)
    dy = max(left_y1 - right_y2, right_y1 - left_y2, 0.0)
    diagonal = math.hypot(float(image_width), float(image_height))
    if diagonal <= 0:
        raise ValueError("image dimensions must be positive")
    return math.hypot(dx, dy) / diagonal


def bbox_iou(left: Sequence[Any], right: Sequence[Any]) -> float:
    left_x1, left_y1, left_x2, left_y2 = valid_bbox(left)
    right_x1, right_y1, right_x2, right_y2 = valid_bbox(right)
    intersection_width = max(0.0, min(left_x2, right_x2) - max(left_x1, right_x1))
    intersection_height = max(0.0, min(left_y2, right_y2) - max(left_y1, right_y1))
    intersection = intersection_width * intersection_height
    if intersection == 0:
        return 0.0
    left_area = (left_x2 - left_x1) * (left_y2 - left_y1)
    right_area = (right_x2 - right_x1) * (right_y2 - right_y1)
    return intersection / (left_area + right_area - intersection)


def scale_bbox(
    bbox: Sequence[Any], width: float, height: float, scale: int = 1000
) -> list[int]:
    x1, y1, x2, y2 = valid_bbox(bbox)
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    values = [
        round(scale * clamp(x1, 0.0, width) / width),
        round(scale * clamp(y1, 0.0, height) / height),
        round(scale * clamp(x2, 0.0, width) / width),
        round(scale * clamp(y2, 0.0, height) / height),
    ]
    if values[2] <= values[0]:
        values[2] = min(scale, values[0] + 1)
        values[0] = min(values[0], values[2] - 1)
    if values[3] <= values[1]:
        values[3] = min(scale, values[1] + 1)
        values[1] = min(values[1], values[3] - 1)
    return values


def unscale_bbox(
    bbox: Sequence[Any], width: float, height: float, scale: int = 1000
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = valid_bbox(bbox)
    return (
        width * x1 / scale,
        height * y1 / scale,
        width * x2 / scale,
        height * y2 / scale,
    )


@dataclass(frozen=True)
class OcrDetection:
    text: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]

    @classmethod
    def from_easyocr(cls, value: Sequence[Any]) -> "OcrDetection":
        if len(value) < 3:
            raise ValueError("EasyOCR detection must contain polygon, text, confidence")
        return cls(
            text=str(value[1]),
            confidence=float(value[2]),
            bbox_xyxy=polygon_bbox(value[0]),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "OcrDetection":
        return cls(
            text=str(value.get("text") or ""),
            confidence=float(value.get("confidence") or 0.0),
            bbox_xyxy=valid_bbox(value["bbox_xyxy"]),
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["bbox_xyxy"] = list(self.bbox_xyxy)
        return value


@dataclass(frozen=True)
class OcrTargetMatch:
    accepted: bool
    reason: str
    target_text: str
    matched_text: str
    text_similarity: float
    ocr_confidence: float
    edge_distance_normalized: float
    source_iou: float
    score: float
    bbox_xyxy: tuple[float, float, float, float] | None
    candidate_index: int | None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if self.bbox_xyxy is not None:
            value["bbox_xyxy"] = list(self.bbox_xyxy)
        return value


def _rejection(reason: str, target_text: Any) -> OcrTargetMatch:
    return OcrTargetMatch(
        accepted=False,
        reason=reason,
        target_text=str(target_text or ""),
        matched_text="",
        text_similarity=0.0,
        ocr_confidence=0.0,
        edge_distance_normalized=1.0,
        source_iou=0.0,
        score=0.0,
        bbox_xyxy=None,
        candidate_index=None,
    )


def match_ocr_target(
    *,
    target_text: Any,
    source_bbox_xyxy: Sequence[Any],
    detections: Iterable[OcrDetection | Mapping[str, Any]],
    image_width: float,
    image_height: float,
    minimum_ocr_confidence: float = MINIMUM_OCR_CONFIDENCE,
    minimum_text_similarity: float = MINIMUM_TEXT_SIMILARITY,
    maximum_edge_distance: float = MAXIMUM_EDGE_DISTANCE,
) -> OcrTargetMatch:
    """Link a target description to the best nearby OCR text region.

    Acceptance is decided from annotation-side evidence only.  Exact text is
    allowed slightly farther from an icon box; fuzzy matches must be closer.
    """

    normalized_target = normalize_ocr_text(target_text)
    if not normalized_target:
        return _rejection("empty_target_text", target_text)
    source_bbox = valid_bbox(source_bbox_xyxy)

    candidates: list[tuple[float, int, OcrDetection, float, float, float]] = []
    for index, raw_detection in enumerate(detections):
        detection = (
            raw_detection
            if isinstance(raw_detection, OcrDetection)
            else OcrDetection.from_mapping(raw_detection)
        )
        if detection.confidence < minimum_ocr_confidence:
            continue
        similarity = text_similarity(normalized_target, detection.text)
        if similarity < minimum_text_similarity:
            continue
        distance = bbox_edge_distance(
            source_bbox, detection.bbox_xyxy, image_width, image_height
        )
        # Fuzzy text must be local.  Near-exact text can link an icon to a
        # caption just outside its DOM rectangle, but never across the screen.
        allowed_distance = (
            maximum_edge_distance * 1.5 if similarity >= 0.92 else maximum_edge_distance
        )
        if distance > allowed_distance:
            continue
        overlap = bbox_iou(source_bbox, detection.bbox_xyxy)
        spatial = max(0.0, 1.0 - distance / max(allowed_distance, 1e-9))
        confidence = clamp(detection.confidence, 0.0, 1.0)
        score = 0.72 * similarity + 0.18 * spatial + 0.07 * confidence + 0.03 * min(
            1.0, overlap * 4.0
        )
        candidates.append((score, index, detection, similarity, distance, overlap))

    if not candidates:
        return _rejection("no_credible_nearby_text", target_text)

    # Stable tie-breaking keeps results independent of OCR list iteration
    # quirks after score/confidence: prefer closer, then the original index.
    score, index, detection, similarity, distance, overlap = max(
        candidates,
        key=lambda item: (
            item[0],
            item[2].confidence,
            -item[4],
            -item[1],
        ),
    )
    return OcrTargetMatch(
        accepted=True,
        reason="matched_nearby_ocr_text",
        target_text=str(target_text or ""),
        matched_text=detection.text,
        text_similarity=similarity,
        ocr_confidence=detection.confidence,
        edge_distance_normalized=distance,
        source_iou=overlap,
        score=score,
        bbox_xyxy=detection.bbox_xyxy,
        candidate_index=index,
    )


__all__ = [
    "OCR_REALIGNMENT_VERSION",
    "OCR_MATCH_CONFIG",
    "OcrDetection",
    "OcrTargetMatch",
    "bbox_edge_distance",
    "match_ocr_target",
    "normalize_ocr_text",
    "replace_action_bbox",
    "scale_bbox",
    "text_similarity",
    "unscale_bbox",
]
