"""Parsing and metrics for the GUI-grounding protocol in arXiv:2603.26211.

The paper reports action-type F1 and point-in-box Step Success Rate (SSR).
Several statements in the paper leave small protocol ambiguities, so this
module deliberately reports both interpretations instead of hiding them:

* point-only SSR, matching the metric definition in section 5.4/appendix D;
* joint step success, requiring both the action type and point to be correct;
* macro F1 over all three action types, plus micro F1/action accuracy.
"""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


ACTION_TYPES = ("lclick", "hover", "type_in")
_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
ACTION_RE = re.compile(
    rf"(?<![A-Za-z0-9_])(lclick|hover|type_in)\s*"
    rf"\[\s*({_NUMBER})\s*,\s*({_NUMBER})\s*,\s*({_NUMBER})\s*,\s*({_NUMBER})\s*\]"
    rf"(?:\s+([^\r\n<]*))?",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedAction:
    """Normalized model action parsed from a generated string."""

    action: str | None
    bbox_1000: tuple[float, float, float, float] | None
    value: str
    valid: bool
    error: str | None


def parse_action(text: Any) -> ParsedAction:
    """Parse the first supported action and validate its normalized box."""

    if not isinstance(text, str) or not text.strip():
        return ParsedAction(None, None, "", False, "empty_prediction")
    match = ACTION_RE.search(text)
    if match is None:
        return ParsedAction(None, None, "", False, "action_or_bbox_not_found")

    action = match.group(1).lower()
    coords = tuple(float(value) for value in match.groups()[1:5])
    value = (match.group(6) or "").strip()
    if not all(math.isfinite(value_) for value_ in coords):
        return ParsedAction(action, None, value, False, "non_finite_bbox")
    if not all(0.0 <= value_ <= 1000.0 for value_ in coords):
        return ParsedAction(action, coords, value, False, "bbox_out_of_range")
    x1, y1, x2, y2 = coords
    if x2 <= x1 or y2 <= y1:
        return ParsedAction(action, coords, value, False, "degenerate_bbox")
    return ParsedAction(action, coords, value, True, None)


def point_in_box(
    point: Sequence[float], bbox: Sequence[float], *, inclusive: bool = True
) -> bool:
    """Return whether a point lies inside an ``xyxy`` box."""

    if len(point) != 2 or len(bbox) != 4:
        return False
    x, y = (float(value) for value in point)
    x1, y1, x2, y2 = (float(value) for value in bbox)
    if inclusive:
        return x1 <= x <= x2 and y1 <= y <= y2
    return x1 < x < x2 and y1 < y < y2


def bbox_center(bbox: Sequence[float]) -> tuple[float, float]:
    if len(bbox) != 4:
        raise ValueError("bbox must contain four coordinates")
    x1, y1, x2, y2 = (float(value) for value in bbox)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _f1_per_label(
    targets: Sequence[str], predictions: Sequence[str | None], label: str
) -> float:
    tp = sum(t == label and p == label for t, p in zip(targets, predictions))
    fp = sum(t != label and p == label for t, p in zip(targets, predictions))
    fn = sum(t == label and p != label for t, p in zip(targets, predictions))
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else (2.0 * tp) / denominator


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _mean_or_none(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def score_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Score joined ground-truth/prediction records.

    Each record must include ``target_action`` and ``target_bbox_1000``.  A
    prediction can be supplied as generated ``prediction`` text or as the
    already parsed ``predicted_action``/``predicted_bbox_1000`` fields.
    """

    rows = list(records)
    targets: list[str] = []
    predictions: list[str | None] = []
    point_hits: list[bool] = []
    joint_hits: list[bool] = []
    parse_errors: Counter[str] = Counter()
    latencies: list[float] = []
    convergence_steps: list[float] = []

    for row in rows:
        target_action = str(row["target_action"]).lower()
        if target_action not in ACTION_TYPES:
            raise ValueError(f"unsupported target action: {target_action}")
        target_bbox = tuple(float(value) for value in row["target_bbox_1000"])
        if len(target_bbox) != 4:
            raise ValueError("target_bbox_1000 must contain four coordinates")

        if "predicted_action" in row or "predicted_bbox_1000" in row:
            predicted_action = row.get("predicted_action")
            predicted_action = (
                str(predicted_action).lower() if predicted_action is not None else None
            )
            raw_bbox = row.get("predicted_bbox_1000")
            predicted_bbox = (
                tuple(float(value) for value in raw_bbox)
                if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4
                else None
            )
            valid = (
                predicted_action in ACTION_TYPES
                and predicted_bbox is not None
                and all(0.0 <= value <= 1000.0 for value in predicted_bbox)
                and predicted_bbox[2] > predicted_bbox[0]
                and predicted_bbox[3] > predicted_bbox[1]
            )
            if not valid:
                parse_errors[str(row.get("parse_error") or "invalid_parsed_prediction")] += 1
        else:
            parsed = parse_action(row.get("prediction"))
            predicted_action = parsed.action
            predicted_bbox = parsed.bbox_1000
            valid = parsed.valid
            if not valid:
                parse_errors[str(parsed.error)] += 1

        action_hit = predicted_action == target_action
        point_hit = bool(
            valid
            and predicted_bbox is not None
            and point_in_box(bbox_center(predicted_bbox), target_bbox)
        )
        targets.append(target_action)
        predictions.append(predicted_action)
        point_hits.append(point_hit)
        joint_hits.append(action_hit and point_hit)

        latency = row.get("latency_seconds")
        if isinstance(latency, (int, float)) and math.isfinite(float(latency)):
            latencies.append(float(latency))
        converged = row.get("convergence_steps")
        if isinstance(converged, (int, float)) and math.isfinite(float(converged)):
            convergence_steps.append(float(converged))

    count = len(rows)
    correct_actions = sum(target == prediction for target, prediction in zip(targets, predictions))
    per_label = {
        label: _f1_per_label(targets, predictions, label) for label in ACTION_TYPES
    }
    present_labels = [label for label in ACTION_TYPES if label in targets]
    macro_present = (
        statistics.fmean(per_label[label] for label in present_labels)
        if present_labels
        else 0.0
    )

    return {
        "num_samples": count,
        "num_parsed": count - sum(parse_errors.values()),
        "parse_rate": 0.0 if count == 0 else (count - sum(parse_errors.values())) / count,
        "parse_errors": dict(sorted(parse_errors.items())),
        "action_accuracy": 0.0 if count == 0 else correct_actions / count,
        "action_f1_micro": 0.0 if count == 0 else correct_actions / count,
        "action_f1_macro_all": statistics.fmean(per_label.values()),
        "action_f1_macro_present": macro_present,
        "action_f1_per_type": per_label,
        "action_support": dict(Counter(targets)),
        "ssr_point_only": 0.0 if count == 0 else sum(point_hits) / count,
        "joint_step_success": 0.0 if count == 0 else sum(joint_hits) / count,
        "latency_seconds": {
            "count": len(latencies),
            "mean": _mean_or_none(latencies),
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "min": min(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "convergence_steps": {
            "count": len(convergence_steps),
            "mean": _mean_or_none(convergence_steps),
            "p50": _percentile(convergence_steps, 0.50),
            "p95": _percentile(convergence_steps, 0.95),
        },
    }
