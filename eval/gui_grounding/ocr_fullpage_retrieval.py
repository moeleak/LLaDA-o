#!/usr/bin/env python3
"""Ground full-page GUI targets with prompt-only OCR retrieval and model fallback."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from eval.gui_grounding.score_benchmark import (
    load_predictions,
    load_targets,
)
from scripts.data.ocr_target_realignment import (
    OcrDetection,
    scale_bbox,
    text_similarity,
)


_FULL_PAGE_PREFIX_END = "Treat them as one complete page. "
_FULL_PAGE_SUFFIX = (
    " Return the action and bounding box with coordinates normalized to "
    "the complete original screenshot in [0,1000]."
)
_CLICK_RE = re.compile(r"^Click on (.+)\.$", re.DOTALL)
_HOVER_RE = re.compile(r"^Hover over (.+)\.$", re.DOTALL)
_TYPE_RE = re.compile(r'^Type "(.*)" into (.+)\.$', re.DOTALL)
_SELECT_RE = re.compile(r'^Select "(.*)" from (.+)\.$', re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmark", default="mind2web_fullpage")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--languages", default="en")
    parser.add_argument("--gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--minimum-confidence", type=float, default=0.20)
    parser.add_argument("--minimum-similarity", type=float, default=0.68)
    parser.add_argument("--model-proximity-weight", type=float, default=0.10)
    parser.add_argument("--label-control-offset", type=float, default=40.0)
    args = parser.parse_args()
    if args.limit <= 0 or args.limit > 100:
        parser.error("--limit must be in [1, 100]")
    if not 0.0 <= args.model_proximity_weight <= 1.0:
        parser.error("--model-proximity-weight must be in [0, 1]")
    if args.label_control_offset < 0:
        parser.error("--label-control-offset must be non-negative")
    return args


def native_instruction(prompt: str) -> str:
    if _FULL_PAGE_PREFIX_END in prompt:
        prompt = prompt.split(_FULL_PAGE_PREFIX_END, 1)[1]
    if prompt.endswith(_FULL_PAGE_SUFFIX):
        prompt = prompt[: -len(_FULL_PAGE_SUFFIX)]
    return prompt.strip()


def instruction_target(prompt: str) -> tuple[str, str, str]:
    """Parse action, visible target phrase, and optional typed value."""

    instruction = native_instruction(prompt)
    match = _CLICK_RE.fullmatch(instruction)
    if match:
        return "lclick", match.group(1), ""
    match = _HOVER_RE.fullmatch(instruction)
    if match:
        return "hover", match.group(1), ""
    match = _TYPE_RE.fullmatch(instruction)
    if match:
        return "type_in", match.group(2), match.group(1)
    match = _SELECT_RE.fullmatch(instruction)
    if match:
        return "type_in", match.group(2), match.group(1)
    raise ValueError(f"unsupported GUI instruction: {instruction!r}")


def globalize_detection(
    raw: Iterable[Any],
    tile_box: Iterable[Any],
) -> OcrDetection:
    local = OcrDetection.from_easyocr(list(raw))
    left, top, _, _ = (float(value) for value in tile_box)
    x1, y1, x2, y2 = local.bbox_xyxy
    return OcrDetection(
        text=local.text,
        confidence=local.confidence,
        bbox_xyxy=(x1 + left, y1 + top, x2 + left, y2 + top),
    )


def select_text_match(
    target: str,
    detections: Iterable[OcrDetection],
    *,
    minimum_confidence: float = 0.20,
    minimum_similarity: float = 0.68,
    reference_point: tuple[float, float] | None = None,
    image_size: tuple[int, int] | None = None,
    proximity_weight: float = 0.0,
) -> tuple[OcrDetection | None, float]:
    candidates: list[tuple[float, float, float, float, OcrDetection]] = []
    for detection in detections:
        if detection.confidence < minimum_confidence:
            continue
        similarity = text_similarity(target, detection.text)
        if similarity < minimum_similarity:
            continue
        score = 0.90 * similarity + 0.10 * min(
            1.0, max(0.0, detection.confidence)
        )
        x1, y1, x2, y2 = detection.bbox_xyxy
        if reference_point is not None and image_size is not None:
            diagonal = max(
                1.0,
                (image_size[0] ** 2 + image_size[1] ** 2) ** 0.5,
            )
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            distance = (
                (center_x - reference_point[0]) ** 2
                + (center_y - reference_point[1]) ** 2
            ) ** 0.5
            proximity = max(0.0, 1.0 - distance / diagonal)
            score = (
                (1.0 - proximity_weight) * score
                + proximity_weight * proximity
            )
        candidates.append((score, similarity, -y1, -x1, detection))
    if not candidates:
        return None, 0.0
    score, _, _, _, detection = max(candidates, key=lambda value: value[:4])
    return detection, score


def label_points_to_control(action: str, target_text: str) -> bool:
    normalized = target_text.strip().casefold()
    return bool(
        action == "type_in"
        or normalized.startswith("*")
        or normalized.startswith("select ")
        or normalized.startswith("search by ")
    )


def shift_detection(
    detection: OcrDetection,
    *,
    offset_y: float,
) -> OcrDetection:
    x1, y1, x2, y2 = detection.bbox_xyxy
    return OcrDetection(
        text=detection.text,
        confidence=detection.confidence,
        bbox_xyxy=(x1, y1 + offset_y, x2, y2 + offset_y),
    )


def build_reader(args: argparse.Namespace) -> Any:
    try:
        import easyocr
    except ImportError as exc:
        raise RuntimeError("easyocr==1.7.2 is required") from exc
    languages = [
        value.strip() for value in args.languages.split(",") if value.strip()
    ]
    return easyocr.Reader(
        languages,
        gpu=args.gpu,
        model_storage_directory=str(args.model_dir),
        download_enabled=False,
        verbose=True,
    )


def detect_tiles(reader: Any, image: Image.Image, sample: dict[str, Any]) -> list[OcrDetection]:
    import numpy as np

    detections: list[OcrDetection] = []
    for tile in sample["tile_layout"]:
        box = tuple(int(value) for value in tile["box_xyxy"])
        raw = reader.readtext(
            np.asarray(image.crop(box)),
            decoder="greedy",
            beamWidth=1,
            batch_size=32,
            workers=0,
            detail=1,
            paragraph=False,
            canvas_size=2560,
            mag_ratio=1.0,
            text_threshold=0.6,
            low_text=0.3,
            link_threshold=0.4,
        )
        detections.extend(globalize_detection(value, box) for value in raw)
    return detections


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    temporary.replace(path)
    return count


def main() -> None:
    args = parse_args()
    root = args.benchmark_root.expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    targets = load_targets(root, manifest, args.benchmark, args.limit)
    predictions = load_predictions(
        args.predictions_dir.expanduser().resolve(),
        args.benchmark,
    )
    unexpected = set(predictions) - set(targets)
    missing = set(targets) - set(predictions)
    if unexpected or missing:
        raise RuntimeError(
            f"prediction coverage mismatch: missing={len(missing)}, "
            f"unexpected={len(unexpected)}"
        )
    reader = build_reader(args)
    output_rows: list[dict[str, Any]] = []
    accepted = 0
    started = time.perf_counter()
    for index, (sample_id, sample) in enumerate(targets.items(), 1):
        row = dict(predictions[sample_id])
        action, target_text, _ = instruction_target(str(sample["prompt"]))
        with Image.open(root / sample["image"]) as source:
            image = source.convert("RGB")
            detections = detect_tiles(reader, image, sample)
        baseline_bbox = row.get("predicted_bbox_1000")
        reference_point = None
        if isinstance(baseline_bbox, (list, tuple)) and len(baseline_bbox) == 4:
            reference_point = (
                sample["image_width"]
                * (float(baseline_bbox[0]) + float(baseline_bbox[2]))
                / 2_000.0,
                sample["image_height"]
                * (float(baseline_bbox[1]) + float(baseline_bbox[3]))
                / 2_000.0,
            )
        match, score = select_text_match(
            target_text,
            detections,
            minimum_confidence=args.minimum_confidence,
            minimum_similarity=args.minimum_similarity,
            reference_point=reference_point,
            image_size=(sample["image_width"], sample["image_height"]),
            proximity_weight=args.model_proximity_weight,
        )
        if match is not None:
            raw_match = match
            if label_points_to_control(action, target_text):
                match = shift_detection(
                    match,
                    offset_y=args.label_control_offset,
                )
            row["predicted_bbox_1000"] = scale_bbox(
                match.bbox_xyxy,
                sample["image_width"],
                sample["image_height"],
            )
            accepted += 1
        row["ocr_retrieval"] = {
            "accepted": match is not None,
            "target_text": target_text,
            "matched_text": match.text if match else "",
            "text_score": score,
            "ocr_confidence": match.confidence if match else 0.0,
            "bbox_xyxy": list(match.bbox_xyxy) if match else None,
            "raw_bbox_xyxy": (
                list(raw_match.bbox_xyxy)
                if match is not None
                else None
            ),
            "model_reference_point_xy": (
                list(reference_point) if reference_point else None
            ),
            "model_proximity_weight": args.model_proximity_weight,
            "label_control_offset": (
                args.label_control_offset
                if match is not None
                and label_points_to_control(action, target_text)
                else 0.0
            ),
            "detections": len(detections),
            "uses_ground_truth_location": False,
        }
        output_rows.append(row)
        if index == 1 or index % 10 == 0:
            elapsed = time.perf_counter() - started
            print(
                f"OCR retrieval {index}/{len(targets)} accepted={accepted} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
    output = args.output_dir.expanduser().resolve()
    count = write_jsonl(
        output / args.benchmark / "part-00000.jsonl",
        output_rows,
    )
    (output / "ocr-retrieval-config.json").write_text(
        json.dumps(
            {
                "benchmark_root": str(root),
                "predictions_dir": str(
                    args.predictions_dir.expanduser().resolve()
                ),
                "samples": count,
                "accepted": accepted,
                "minimum_confidence": args.minimum_confidence,
                "minimum_similarity": args.minimum_similarity,
                "model_proximity_weight": args.model_proximity_weight,
                "label_control_offset": args.label_control_offset,
                "uses_prompt_only": True,
                "uses_ground_truth_location": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"wrote {count} predictions to {output}", flush=True)


if __name__ == "__main__":
    main()
