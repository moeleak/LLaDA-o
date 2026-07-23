#!/usr/bin/env python3
"""Prepare the GUI-grounding benchmarks used in arXiv:2603.26211.

The paper names Mind2Web test, ScreenSpot-Web-Text,
ScreenSpot-Web-Icon, and VisualWebArena, but does not release its evaluation
code, prompts, sample IDs, or the static VisualWebArena extraction.  This
script pins and converts the publicly identifiable datasets and records the
remaining protocol gap in ``manifest.json``.  A separately obtained VWA
static export can be imported with ``--visualwebarena-jsonl``.

Prepared samples are JSONL records with image paths and normalized ``[0,1000]``
``xyxy`` boxes, which makes the exact evaluated examples independently
auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from PIL import Image, ImageFile

try:
    from .gui_grounding_protocol import (
        TARGET_GROUNDING,
        TASK_HISTORY,
        mind2web_prompt,
        parse_target_action,
    )
except ImportError:  # Direct execution: python scripts/data/<this file>.py
    from gui_grounding_protocol import (
        TARGET_GROUNDING,
        TASK_HISTORY,
        mind2web_prompt,
        parse_target_action,
    )


Image.MAX_IMAGE_PIXELS = 250_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = True

PAPER_URL = "https://arxiv.org/abs/2603.26211"
MIND2WEB_REPO = "osunlp/Multimodal-Mind2Web"
MIND2WEB_REVISION = "1b4c6a8cf9f77b7a5e0d641959935c80c4a05889"
SCREENSPOT_REPO = "bevaya/ScreenSpot"
SCREENSPOT_REVISION = "0be08781e2e188582f6131625ae1598d443b4d5d"
MIND2WEB_TEST_ROWS = {
    "test_domain": 4_060,
    "test_task": 1_339,
    "test_website": 1_019,
}
SCREENSPOT_ROWS = 1_272
ACTION_TYPES = {"lclick", "hover", "type_in"}

_scratch = os.environ.get("SCRATCH")
DEFAULT_ROOT = Path(
    os.environ.get(
        "LLADAO_GUI_BENCHMARK_ROOT",
        f"{_scratch}/datasets/lladao_gui_benchmarks"
        if _scratch
        else "datasets/lladao_gui_benchmarks",
    )
)
EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
PASSWORD_RE = re.compile(r"(?i)(password\s*[:=]\s*)([^\s,;]+)")


def log(message: str) -> None:
    print(message, flush=True)


def compact_text(value: Any, limit: int = 2_000) -> str:
    text = " ".join(str(value or "").replace("\x00", " ").split())
    text = EMAIL_RE.sub("<EMAIL>", text)
    text = PASSWORD_RE.sub(r"\1<REDACTED>", text)
    return text[:limit]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize_bbox(
    bbox: Sequence[float], width: float, height: float, scale: int = 1000
) -> list[int]:
    if len(bbox) != 4 or width <= 0 or height <= 0:
        raise ValueError("invalid bbox or image dimensions")
    x1, y1, x2, y2 = (float(value) for value in bbox)
    x1, x2 = sorted((clamp(x1, 0, width), clamp(x2, 0, width)))
    y1, y2 = sorted((clamp(y1, 0, height), clamp(y2, 0, height)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("degenerate bbox")
    result = [
        round(scale * x1 / width),
        round(scale * y1 / height),
        round(scale * x2 / width),
        round(scale * y2 / height),
    ]
    result = [max(0, min(scale, value)) for value in result]
    if result[2] <= result[0]:
        result[2] = min(scale, result[0] + 1)
        result[0] = min(result[0], result[2] - 1)
    if result[3] <= result[1]:
        result[3] = min(scale, result[1] + 1)
        result[1] = min(result[1], result[3] - 1)
    return result


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_bytes(value: Any) -> bytes:
    if isinstance(value, dict):
        data = value.get("bytes")
    else:
        data = value
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ValueError("image column does not contain embedded bytes")
    return bytes(data)


def encode_jpeg(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.convert("RGB").save(output, format="JPEG", quality=90, optimize=True)
    return output.getvalue()


def write_image_once(root: Path, category: str, data: bytes) -> str:
    digest = sha256_bytes(data)
    relative = Path("images") / category / digest[:2] / f"{digest}.jpg"
    destination = root / relative
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".jpg.tmp")
        temporary.write_bytes(data)
        os.replace(temporary, destination)
    return relative.as_posix()


def write_source_image_once(root: Path, category: str, data: bytes) -> str:
    """Persist source screenshot bytes without resizing or re-encoding."""

    digest = sha256_bytes(data)
    with Image.open(io.BytesIO(data)) as image:
        image_format = str(image.format or "bin").lower()
    suffix = {
        "jpeg": "jpg",
        "jpg": "jpg",
        "png": "png",
        "webp": "webp",
    }.get(image_format, image_format)
    relative = Path("images") / category / digest[:2] / f"{digest}.{suffix}"
    destination = root / relative
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_bytes(data)
        os.replace(temporary, destination)
    return relative.as_posix()


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    os.replace(temporary, path)
    return count, sha256_file(path)


def parse_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("value is not a JSON object")


def choose_mind2web_candidate(raw_candidates: Sequence[Any]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for raw in raw_candidates or []:
        try:
            candidate = parse_json_object(raw)
            attributes = parse_json_object(candidate.get("attributes", {}))
            if attributes.get("bounding_box_rect"):
                prepared = dict(candidate)
                prepared["_attributes"] = attributes
                candidates.append(prepared)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    if not candidates:
        raise ValueError("no positive candidate with a bounding box")
    return next(
        (candidate for candidate in candidates if candidate.get("is_original_target")),
        candidates[0],
    )


def mind2web_metadata(row: dict[str, Any]) -> dict[str, Any]:
    operation = parse_json_object(row.get("operation", {}))
    candidate = choose_mind2web_candidate(row.get("pos_candidates") or [])
    attributes = candidate["_attributes"]
    raw_rect = attributes["bounding_box_rect"]
    x, y, width, height = [float(value) for value in str(raw_rect).split(",")]
    if width <= 0 or height <= 0:
        raise ValueError("invalid target rectangle")

    fallback_description = next(
        (
            attributes.get(key)
            for key in (
                "aria-label",
                "aria_label",
                "text",
                "title",
                "placeholder",
                "alt",
                "value",
            )
            if attributes.get(key)
        ),
        candidate.get("tag") or "",
    )
    target = parse_target_action(
        row.get("target_action_reprs"),
        operation,
        fallback_description=fallback_description,
    )
    prompt_arguments = {
        "confirmed_task": row.get("confirmed_task"),
        "action_reprs": row.get("action_reprs") or [],
        "target_action_index": row.get("target_action_index", 0),
    }

    return {
        "action_uid": str(row["action_uid"]),
        "annotation_id": str(row.get("annotation_id") or ""),
        "website": str(row.get("website") or ""),
        "bbox": [x, y, x + width, y + height],
        "action": target.action,
        "operation": target.operation,
        "value": target.value,
        "target_description": target.description,
        "target_role": target.role,
        "target_action_repr": target.raw_target_action_repr,
        "prompts": {
            protocol: mind2web_prompt(protocol, target, **prompt_arguments)
            for protocol in (TARGET_GROUNDING, TASK_HISTORY)
        },
    }


def target_center_crop(
    image: Image.Image, bbox: Sequence[float], crop_size: int
) -> tuple[Image.Image, list[float], list[int]]:
    image = image.convert("RGB")
    image_width, image_height = image.size
    x1, y1, x2, y2 = (float(value) for value in bbox)
    if x2 <= 0 or y2 <= 0 or x1 >= image_width or y1 >= image_height:
        raise ValueError("target bbox is outside the screenshot")
    x1, x2 = clamp(x1, 0, image_width), clamp(x2, 0, image_width)
    y1, y2 = clamp(y1, 0, image_height), clamp(y2, 0, image_height)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("target bbox is empty after clipping")

    crop_width = min(image_width, max(crop_size, math.ceil(x2 - x1) + 64))
    crop_height = min(image_height, max(crop_size, math.ceil(y2 - y1) + 64))
    x_low = max(0, math.ceil(x2 - crop_width))
    x_high = min(math.floor(x1), image_width - crop_width)
    y_low = max(0, math.ceil(y2 - crop_height))
    y_high = min(math.floor(y1), image_height - crop_height)
    if x_low > x_high or y_low > y_high:
        raise ValueError("unable to construct a target-preserving crop")
    crop_x = round((x_low + x_high) / 2)
    crop_y = round((y_low + y_high) / 2)
    crop_box = [crop_x, crop_y, crop_x + crop_width, crop_y + crop_height]
    shifted_bbox = [x1 - crop_x, y1 - crop_y, x2 - crop_x, y2 - crop_y]
    return image.crop(tuple(crop_box)), shifted_bbox, crop_box


def parquet_files(directory: Path, pattern: str) -> list[Path]:
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no files matching {pattern} below {directory}")
    return files


def iter_mind2web(
    root: Path, raw_root: Path, crop_size: int, counters: Counter[str]
) -> Iterator[dict[str, Any]]:
    columns = [
        "action_uid",
        "operation",
        "pos_candidates",
        "website",
        "annotation_id",
        "confirmed_task",
        "action_reprs",
        "target_action_index",
        "target_action_reprs",
        "screenshot",
    ]
    for split, expected_rows in MIND2WEB_TEST_ROWS.items():
        split_rows = 0
        for path in parquet_files(raw_root / "data", f"{split}-*.parquet"):
            parquet = pq.ParquetFile(path)
            for row_group in range(parquet.num_row_groups):
                rows = parquet.read_row_group(row_group, columns=columns).to_pylist()
                split_rows += len(rows)
                for row in rows:
                    counters["mind2web_raw"] += 1
                    try:
                        metadata = mind2web_metadata(row)
                        raw_image = image_bytes(row["screenshot"])
                        with Image.open(io.BytesIO(raw_image)) as source:
                            source.load()
                            original_size = list(source.size)
                            cropped, shifted_bbox, crop_box = target_center_crop(
                                source, metadata["bbox"], crop_size
                            )
                        target_bbox = normalize_bbox(shifted_bbox, *cropped.size)
                        prepared_image = encode_jpeg(cropped)
                        image_path = write_image_once(
                            root, f"mind2web/{split}", prepared_image
                        )
                        counters["mind2web_valid_source_rows"] += 1
                        common = {
                            "split": split,
                            "image": image_path,
                            "image_width": cropped.width,
                            "image_height": cropped.height,
                            "target_action": metadata["action"],
                            "target_bbox_1000": target_bbox,
                            "target_value": metadata["value"],
                        }
                        for benchmark, protocol in (
                            ("mind2web", TARGET_GROUNDING),
                            ("mind2web_task_history", TASK_HISTORY),
                        ):
                            counters[f"{benchmark}_valid"] += 1
                            yield {
                                "sample_id": (
                                    f"{benchmark}:{split}:{metadata['action_uid']}"
                                ),
                                "benchmark": benchmark,
                                "prompt": metadata["prompts"][protocol],
                                **common,
                                "provenance": {
                                    "repo": MIND2WEB_REPO,
                                    "revision": MIND2WEB_REVISION,
                                    "action_uid": metadata["action_uid"],
                                    "annotation_id": metadata["annotation_id"],
                                    "website": metadata["website"],
                                    "source_operation": metadata["operation"],
                                    "target_action_repr": metadata[
                                        "target_action_repr"
                                    ],
                                    "target_description": metadata[
                                        "target_description"
                                    ],
                                    "target_role": metadata["target_role"],
                                    "prompt_protocol": protocol,
                                    "source_bbox_xyxy": metadata["bbox"],
                                    "source_image_size": original_size,
                                    "crop_xyxy": crop_box,
                                    "preprocessing": (
                                        "deterministic centered target-preserving "
                                        "crop; DOM target box; JPEG quality 90"
                                    ),
                                },
                            }
                    except Exception as exc:
                        counters[f"mind2web_rejected:{type(exc).__name__}"] += 1
        if split_rows != expected_rows:
            raise RuntimeError(
                f"{split} contains {split_rows:,} rows; expected {expected_rows:,}"
            )


def full_page_tile_layout(
    width: int,
    height: int,
    *,
    tile_size: int,
    patch_size: int,
) -> list[dict[str, Any]]:
    if width <= 0 or height <= 0:
        raise ValueError("full-page image dimensions must be positive")
    if tile_size <= 0 or tile_size > 980:
        raise ValueError("tile_size must be in [1, 980]")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    layout: list[dict[str, Any]] = []
    index = 0
    for top in range(0, height, tile_size):
        for left in range(0, width, tile_size):
            right = min(left + tile_size, width)
            bottom = min(top + tile_size, height)
            tile_width = right - left
            tile_height = bottom - top
            grid_width = math.ceil(tile_width / patch_size)
            grid_height = math.ceil(tile_height / patch_size)
            layout.append(
                {
                    "index": index,
                    "box_xyxy": [left, top, right, bottom],
                    "grid_width": grid_width,
                    "grid_height": grid_height,
                    "patch_tokens": grid_width * grid_height,
                }
            )
            index += 1
    return layout


def full_page_prompt(
    instruction: str,
    *,
    width: int,
    height: int,
    tile_count: int,
) -> str:
    return (
        f"The following {tile_count} images are non-overlapping tiles from one "
        f"{width}x{height} webpage screenshot, ordered left-to-right and then "
        "top-to-bottom. Treat them as one complete page. "
        f"{instruction} Return the action and bounding box with coordinates "
        "normalized to the complete original screenshot in [0,1000]."
    )


def iter_mind2web_fullpage(
    root: Path,
    raw_root: Path,
    counters: Counter[str],
    *,
    tokenizer: Any,
    tile_size: int,
    patch_size: int,
    max_new_tokens: int,
    min_total_tokens: int,
    max_total_tokens: int,
) -> Iterator[dict[str, Any]]:
    columns = [
        "action_uid",
        "operation",
        "pos_candidates",
        "website",
        "annotation_id",
        "confirmed_task",
        "action_reprs",
        "target_action_index",
        "target_action_reprs",
        "screenshot",
    ]
    for split, expected_rows in MIND2WEB_TEST_ROWS.items():
        split_rows = 0
        for path in parquet_files(raw_root / "data", f"{split}-*.parquet"):
            parquet = pq.ParquetFile(path)
            for row_group in range(parquet.num_row_groups):
                rows = parquet.read_row_group(
                    row_group, columns=columns
                ).to_pylist()
                split_rows += len(rows)
                for row in rows:
                    counters["mind2web_fullpage_raw"] += 1
                    try:
                        metadata = mind2web_metadata(row)
                        raw_image = image_bytes(row["screenshot"])
                        with Image.open(io.BytesIO(raw_image)) as source:
                            source.load()
                            width, height = source.size
                        target_bbox = normalize_bbox(
                            metadata["bbox"], width, height
                        )
                        layout = full_page_tile_layout(
                            width,
                            height,
                            tile_size=tile_size,
                            patch_size=patch_size,
                        )
                        prompt = full_page_prompt(
                            metadata["prompts"][TARGET_GROUNDING],
                            width=width,
                            height=height,
                            tile_count=len(layout),
                        )
                        prompt_tokens = (
                            len(
                                tokenizer.encode(
                                    prompt, add_special_tokens=False
                                )
                            )
                            + 2
                        )
                        patch_tokens = sum(
                            int(tile["patch_tokens"]) for tile in layout
                        )
                        image_tokens = patch_tokens + 2 * len(layout)
                        total_tokens = (
                            image_tokens
                            + prompt_tokens
                            + max_new_tokens
                        )
                        if total_tokens <= min_total_tokens:
                            counters[
                                "mind2web_fullpage_filtered:at_or_below_min"
                            ] += 1
                            continue
                        if total_tokens > max_total_tokens:
                            counters[
                                "mind2web_fullpage_filtered:above_max"
                            ] += 1
                            continue
                        image_path = write_source_image_once(
                            root,
                            f"mind2web_fullpage/{split}",
                            raw_image,
                        )
                        counters["mind2web_fullpage_valid"] += 1
                        yield {
                            "sample_id": (
                                f"mind2web_fullpage:{split}:"
                                f"{metadata['action_uid']}"
                            ),
                            "benchmark": "mind2web_fullpage",
                            "split": split,
                            "image": image_path,
                            "image_width": width,
                            "image_height": height,
                            "input_protocol": "full_page_tiles",
                            "prompt": prompt,
                            "target_action": metadata["action"],
                            "target_bbox_1000": target_bbox,
                            "target_value": metadata["value"],
                            "sequence_tokens": {
                                "image": image_tokens,
                                "patches": patch_tokens,
                                "prompt": prompt_tokens,
                                "generation": max_new_tokens,
                                "total": total_tokens,
                            },
                            "tile_layout": layout,
                            "provenance": {
                                "repo": MIND2WEB_REPO,
                                "revision": MIND2WEB_REVISION,
                                "action_uid": metadata["action_uid"],
                                "annotation_id": metadata["annotation_id"],
                                "website": metadata["website"],
                                "source_operation": metadata["operation"],
                                "target_action_repr": metadata[
                                    "target_action_repr"
                                ],
                                "target_description": metadata[
                                    "target_description"
                                ],
                                "target_role": metadata["target_role"],
                                "source_bbox_xyxy": metadata["bbox"],
                                "source_image_sha256": sha256_bytes(raw_image),
                                "preprocessing": (
                                    "original screenshot bytes; deterministic "
                                    "non-overlapping row-major tiles; no crop, "
                                    "resize, OCR realignment, or re-encoding"
                                ),
                            },
                        }
                    except Exception as exc:
                        counters[
                            "mind2web_fullpage_rejected:"
                            f"{type(exc).__name__}"
                        ] += 1
        if split_rows != expected_rows:
            raise RuntimeError(
                f"{split} contains {split_rows:,} rows; "
                f"expected {expected_rows:,}"
            )


def is_screenspot_web(row: dict[str, Any]) -> bool:
    file_name = str(row.get("file_name") or "").lower()
    source = str(row.get("data_source") or "").lower()
    return file_name.startswith("web_") or source in {
        "web",
        "gitlab",
        "shop",
        "forum",
        "tool",
    }


def iter_screenspot(
    root: Path, raw_root: Path, counters: Counter[str]
) -> Iterator[dict[str, Any]]:
    raw_rows = 0
    for path in parquet_files(raw_root / "data", "test-*.parquet"):
        parquet = pq.ParquetFile(path)
        for row_group in range(parquet.num_row_groups):
            for row in parquet.read_row_group(row_group).to_pylist():
                raw_rows += 1
                counters["screenspot_raw"] += 1
                if not is_screenspot_web(row):
                    continue
                try:
                    data_type = str(row.get("data_type") or "").lower()
                    category = "text" if data_type == "text" else "icon"
                    bbox = [float(value) for value in row["bbox"]]
                    if len(bbox) != 4:
                        raise ValueError("bbox does not have four coordinates")
                    raw_image = image_bytes(row["image"])
                    with Image.open(io.BytesIO(raw_image)) as image:
                        width, height = image.size
                    if all(0.0 <= value <= 1.0 for value in bbox):
                        target_bbox = [round(value * 1000) for value in bbox]
                    else:
                        target_bbox = normalize_bbox(bbox, width, height)
                    if target_bbox[2] <= target_bbox[0] or target_bbox[3] <= target_bbox[1]:
                        raise ValueError("degenerate target box")
                    prepared_image = encode_jpeg(Image.open(io.BytesIO(raw_image)))
                    image_path = write_image_once(root, "screenspot_web", prepared_image)
                    file_name = str(row.get("file_name") or sha256_bytes(raw_image))
                    instruction = compact_text(row.get("instruction"), 1_000)
                    sample_hash = hashlib.sha256(
                        f"{file_name}\x1f{instruction}\x1f{bbox}".encode("utf-8")
                    ).hexdigest()[:20]
                    benchmark = f"screenspot_web_{category}"
                    counters[f"{benchmark}_valid"] += 1
                    yield {
                        "sample_id": f"screenspot:{category}:{sample_hash}",
                        "benchmark": benchmark,
                        "split": "test",
                        "image": image_path,
                        "image_width": width,
                        "image_height": height,
                        "prompt": (
                            "Locate and click the web UI element described as: "
                            f'"{instruction}".'
                        ),
                        "target_action": "lclick",
                        "target_bbox_1000": target_bbox,
                        "target_value": "",
                        "provenance": {
                            "repo": SCREENSPOT_REPO,
                            "revision": SCREENSPOT_REVISION,
                            "file_name": file_name,
                            "data_type": data_type,
                            "data_source": str(row.get("data_source") or ""),
                            "source_bbox": bbox,
                        },
                    }
                except Exception as exc:
                    counters[f"screenspot_rejected:{type(exc).__name__}"] += 1
    if raw_rows != SCREENSPOT_ROWS:
        raise RuntimeError(
            f"ScreenSpot contains {raw_rows:,} rows; expected {SCREENSPOT_ROWS:,}"
        )


def external_bbox_1000(row: dict[str, Any], width: int, height: int) -> list[int]:
    if "target_bbox_1000" in row:
        bbox = [float(value) for value in row["target_bbox_1000"]]
        if len(bbox) != 4:
            raise ValueError("target_bbox_1000 must contain four coordinates")
        return [round(value) for value in bbox]
    bbox = [float(value) for value in row["bbox"]]
    bbox_format = str(row.get("bbox_format") or "xyxy_pixels").lower()
    if bbox_format in {"xyxy_0_1", "normalized", "normalized_0_1"}:
        return [round(value * 1000) for value in bbox]
    if bbox_format in {"xyxy_1000", "normalized_1000"}:
        return [round(value) for value in bbox]
    if bbox_format == "xywh_pixels":
        x, y, box_width, box_height = bbox
        bbox = [x, y, x + box_width, y + box_height]
    return normalize_bbox(bbox, width, height)


def iter_external_visualwebarena(
    root: Path, source_path: Path, counters: Counter[str]
) -> Iterator[dict[str, Any]]:
    """Import a static VWA export without claiming it is the unpublished set."""

    with source_path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            counters["visualwebarena_raw"] += 1
            row = json.loads(line)
            source_image = Path(row["image"])
            if not source_image.is_absolute():
                source_image = source_path.parent / source_image
            raw_image = source_image.read_bytes()
            with Image.open(io.BytesIO(raw_image)) as image:
                width, height = image.size
                prepared_image = encode_jpeg(image)
            bbox = external_bbox_1000(row, width, height)
            action = str(row.get("target_action") or row.get("action") or "lclick").lower()
            if action not in ACTION_TYPES:
                raise ValueError(f"unsupported VWA action type: {action}")
            image_path = write_image_once(root, "visualwebarena", prepared_image)
            counters["visualwebarena_valid"] += 1
            yield {
                "sample_id": str(row.get("sample_id") or f"visualwebarena:{index:06d}"),
                "benchmark": "visualwebarena",
                "split": str(row.get("split") or "test"),
                "image": image_path,
                "image_width": width,
                "image_height": height,
                "prompt": str(row.get("prompt") or row.get("instruction") or ""),
                "target_action": action,
                "target_bbox_1000": bbox,
                "target_value": str(row.get("target_value") or ""),
                "provenance": {
                    "external_manifest": str(source_path.resolve()),
                    "source_sample_id": row.get("sample_id"),
                    "paper_static_subset_match": False,
                },
            }


def download_sources(root: Path) -> None:
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    log(f"Downloading pinned {MIND2WEB_REPO}@{MIND2WEB_REVISION}")
    snapshot_download(
        repo_id=MIND2WEB_REPO,
        repo_type="dataset",
        revision=MIND2WEB_REVISION,
        allow_patterns=["README.md", "data/test_*.parquet"],
        local_dir=raw / "mind2web",
    )
    log(f"Downloading pinned {SCREENSPOT_REPO}@{SCREENSPOT_REVISION}")
    snapshot_download(
        repo_id=SCREENSPOT_REPO,
        repo_type="dataset",
        revision=SCREENSPOT_REVISION,
        allow_patterns=["README.md", "data/*.parquet"],
        local_dir=raw / "screenspot",
    )


def prepare_output(root: Path, force: bool) -> None:
    for path in (root / "samples", root / "images"):
        if path.exists():
            if not force:
                raise FileExistsError(f"prepared output exists: {path}; pass --force")
            shutil.rmtree(path)
    for path in (root / "manifest.json", root / "validation.json"):
        if path.exists():
            if not force:
                raise FileExistsError(f"prepared output exists: {path}; pass --force")
            path.unlink()


def build_sources(
    root: Path,
    *,
    crop_size: int,
    visualwebarena_jsonl: Path | None,
    force: bool,
) -> dict[str, Any]:
    prepare_output(root, force)
    counters: Counter[str] = Counter()
    files: dict[str, dict[str, Any]] = {}

    mind2web_records = list(
        iter_mind2web(root, root / "raw" / "mind2web", crop_size, counters)
    )
    for benchmark, protocol in (
        ("mind2web", TARGET_GROUNDING),
        ("mind2web_task_history", TASK_HISTORY),
    ):
        path = root / "samples" / f"{benchmark}.jsonl"
        count, digest = write_jsonl(
            path,
            (row for row in mind2web_records if row["benchmark"] == benchmark),
        )
        files[benchmark] = {
            "path": path.relative_to(root).as_posix(),
            "rows": count,
            "sha256": digest,
            "prompt_protocol": protocol,
            "paper_comparison_eligible": protocol == TARGET_GROUNDING,
        }

    screenspot_records = list(
        iter_screenspot(root, root / "raw" / "screenspot", counters)
    )
    for benchmark in ("screenspot_web_text", "screenspot_web_icon"):
        path = root / "samples" / f"{benchmark}.jsonl"
        count, digest = write_jsonl(
            path,
            (row for row in screenspot_records if row["benchmark"] == benchmark),
        )
        files[benchmark] = {
            "path": path.relative_to(root).as_posix(),
            "rows": count,
            "sha256": digest,
        }

    visualwebarena_status: dict[str, Any]
    if visualwebarena_jsonl is not None:
        path = root / "samples" / "visualwebarena.jsonl"
        count, digest = write_jsonl(
            path,
            iter_external_visualwebarena(
                root, visualwebarena_jsonl.resolve(), counters
            ),
        )
        files["visualwebarena"] = {
            "path": path.relative_to(root).as_posix(),
            "rows": count,
            "sha256": digest,
        }
        visualwebarena_status = {
            "available": True,
            "paper_subset_match": False,
            "source": str(visualwebarena_jsonl.resolve()),
        }
    else:
        visualwebarena_status = {
            "available": False,
            "paper_subset_match": False,
            "reason": (
                "The paper does not publish the screenshots, action trajectories, "
                "sample IDs, or extraction code for its static single-step "
                "VisualWebArena subset. Official VisualWebArena is an online "
                "multi-step environment, not this static benchmark."
            ),
        }

    manifest = {
        "paper": PAPER_URL,
        "exact_paper_reproduction": False,
        "protocol_notes": [
            "The paper does not release evaluation code or prompt templates.",
            "Mind2Web includes all three official multimodal test splits because the paper only says 'test split'.",
            "The default mind2web benchmark is target-explicit single-step grounding derived from target_action_reprs, matching the direct imperative shown in the paper figure.",
            "mind2web_task_history is a diagnostic planning-plus-grounding A/B using the high-level task and previous actions; its score must not be compared to the paper's grounding score.",
            "Mind2Web uses the same 1280px target-preserving crop recipe as this repository's training data; the paper does not publish crop seeds or OCR realignment code.",
            "ScreenSpot web text/icon are the official web rows and are click-only.",
            "VisualWebArena cannot be matched exactly without the authors' unpublished static extraction.",
        ],
        "sources": {
            "mind2web": {
                "repo": MIND2WEB_REPO,
                "revision": MIND2WEB_REVISION,
                "license": "OpenRAIL",
            },
            "screenspot": {
                "repo": SCREENSPOT_REPO,
                "revision": SCREENSPOT_REVISION,
                "license": "Apache-2.0",
            },
        },
        "benchmarks": files,
        "visualwebarena": visualwebarena_status,
        "counters": dict(sorted(counters.items())),
        "mind2web_crop_size": crop_size,
        "mind2web_default_protocol": TARGET_GROUNDING,
        "mind2web_protocol_inference": (
            "Exact author prompt is unpublished; direct imperative inferred from "
            "the paper figure and instantiated from public target_action_reprs"
        ),
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_fullpage_mind2web(
    root: Path,
    *,
    raw_root: Path,
    tokenizer_path: Path,
    tile_size: int,
    patch_size: int,
    max_new_tokens: int,
    min_total_tokens: int,
    max_total_tokens: int,
    force: bool,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    if min_total_tokens < 0 or max_total_tokens <= min_total_tokens:
        raise ValueError("token bounds must satisfy 0 <= min < max")
    prepare_output(root, force)
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), use_fast=True, trust_remote_code=False
    )
    counters: Counter[str] = Counter()
    path = root / "samples" / "mind2web_fullpage.jsonl"
    count, digest = write_jsonl(
        path,
        iter_mind2web_fullpage(
            root,
            raw_root,
            counters,
            tokenizer=tokenizer,
            tile_size=tile_size,
            patch_size=patch_size,
            max_new_tokens=max_new_tokens,
            min_total_tokens=min_total_tokens,
            max_total_tokens=max_total_tokens,
        ),
    )
    manifest = {
        "paper": PAPER_URL,
        "exact_paper_reproduction": False,
        "protocol_notes": [
            "This is a long-context diagnostic, not the paper's cropped Mind2Web protocol.",
            "Every screenshot is stored from the pinned official source bytes without crop, resize, OCR realignment, or re-encoding.",
            "Only samples whose exact image+prompt+generation length is greater than the lower bound and at most the upper bound are included.",
        ],
        "sources": {
            "mind2web": {
                "repo": MIND2WEB_REPO,
                "revision": MIND2WEB_REVISION,
                "license": "OpenRAIL",
            }
        },
        "benchmarks": {
            "mind2web_fullpage": {
                "path": path.relative_to(root).as_posix(),
                "rows": count,
                "sha256": digest,
                "prompt_protocol": TARGET_GROUNDING,
                "input_protocol": "full_page_tiles",
                "paper_comparison_eligible": False,
            }
        },
        "counters": dict(sorted(counters.items())),
        "full_page": {
            "tile_size": tile_size,
            "patch_size": patch_size,
            "max_new_tokens": max_new_tokens,
            "min_total_tokens_exclusive": min_total_tokens,
            "max_total_tokens_inclusive": max_total_tokens,
            "tokenizer": str(tokenizer_path.resolve()),
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def validate(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    sample_ids: set[str] = set()
    image_paths: set[str] = set()
    counts: dict[str, int] = {}
    action_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()

    for benchmark, details in manifest["benchmarks"].items():
        path = root / details["path"]
        if sha256_file(path) != details["sha256"]:
            raise RuntimeError(f"checksum mismatch: {path}")
        count = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                sample_id = str(row["sample_id"])
                if sample_id in sample_ids:
                    raise RuntimeError(f"duplicate sample_id: {sample_id}")
                sample_ids.add(sample_id)
                if row["benchmark"] != benchmark:
                    raise RuntimeError(f"benchmark mismatch in {sample_id}")
                action = str(row["target_action"])
                if action not in ACTION_TYPES:
                    raise RuntimeError(f"invalid action in {sample_id}: {action}")
                bbox = [float(value) for value in row["target_bbox_1000"]]
                if (
                    len(bbox) != 4
                    or not all(0 <= value <= 1000 for value in bbox)
                    or bbox[2] <= bbox[0]
                    or bbox[3] <= bbox[1]
                ):
                    raise RuntimeError(f"invalid bbox in {sample_id}: {bbox}")
                image_path = root / row["image"]
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                image_paths.add(row["image"])
                action_counts[action] += 1
                split_counts[f"{benchmark}:{row['split']}"] += 1
                count += 1
        if count != int(details["rows"]):
            raise RuntimeError(
                f"row count mismatch for {benchmark}: {count} != {details['rows']}"
            )
        counts[benchmark] = count

    report = {
        "benchmarks": counts,
        "total_samples": sum(counts.values()),
        "unique_sample_ids": len(sample_ids),
        "unique_images": len(image_paths),
        "action_counts": dict(sorted(action_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "visualwebarena_available": bool(
            manifest.get("visualwebarena", {}).get("available")
        ),
    }
    (root / "validation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def add_common_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="download pinned source files")
    add_common_root(download)

    build = subparsers.add_parser("build", help="convert source data to benchmark JSONL")
    add_common_root(build)
    build.add_argument("--mind2web-crop-size", type=int, default=1280)
    build.add_argument("--visualwebarena-jsonl", type=Path)
    build.add_argument("--force", action="store_true")

    all_parser = subparsers.add_parser("all", help="download, build, and validate")
    add_common_root(all_parser)
    all_parser.add_argument("--mind2web-crop-size", type=int, default=1280)
    all_parser.add_argument("--visualwebarena-jsonl", type=Path)
    all_parser.add_argument("--force", action="store_true")

    fullpage = subparsers.add_parser(
        "build-fullpage",
        help="build the uncompressed 16K-64K Mind2Web diagnostic",
    )
    add_common_root(fullpage)
    fullpage.add_argument("--raw-root", type=Path)
    fullpage.add_argument("--tokenizer", type=Path, required=True)
    fullpage.add_argument("--tile-size", type=int, default=980)
    fullpage.add_argument("--patch-size", type=int, default=14)
    fullpage.add_argument("--max-new-tokens", type=int, default=64)
    fullpage.add_argument("--min-total-tokens", type=int, default=16_384)
    fullpage.add_argument("--max-total-tokens", type=int, default=65_536)
    fullpage.add_argument("--force", action="store_true")

    validate_parser = subparsers.add_parser("validate", help="validate prepared data")
    add_common_root(validate_parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if args.command in {"download", "all"}:
        download_sources(root)
    if args.command in {"build", "all"}:
        manifest = build_sources(
            root,
            crop_size=args.mind2web_crop_size,
            visualwebarena_jsonl=args.visualwebarena_jsonl,
            force=args.force,
        )
        log(json.dumps(manifest["benchmarks"], indent=2, sort_keys=True))
    if args.command == "build-fullpage":
        raw_root = (
            args.raw_root.expanduser().resolve()
            if args.raw_root
            else root / "raw" / "mind2web"
        )
        manifest = build_fullpage_mind2web(
            root,
            raw_root=raw_root,
            tokenizer_path=args.tokenizer.expanduser().resolve(),
            tile_size=args.tile_size,
            patch_size=args.patch_size,
            max_new_tokens=args.max_new_tokens,
            min_total_tokens=args.min_total_tokens,
            max_total_tokens=args.max_total_tokens,
            force=args.force,
        )
        log(json.dumps(manifest["benchmarks"], indent=2, sort_keys=True))
    if args.command in {"validate", "all"}:
        report = validate(root)
        log(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
