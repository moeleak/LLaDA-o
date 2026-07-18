#!/usr/bin/env python3
"""Prepare the GUI-grounding mixture described in arXiv:2603.26211.

The paper publishes aggregate source counts, but not sample IDs, crop seeds, or
the OCR realignment implementation.  This script therefore builds a pinned,
deterministic approximation and records every decision in the output manifest.
Mind2Web's usable public rows are repeated with deterministic random,
target-preserving crops to reach the paper's 20K allocation; every crop variant
is explicitly identified in provenance rather than presented as a new source
example.

The resulting Parquet files are directly consumable by LLaDA-o's
``vlm_parquet`` dataset loader.  Each row contains one image and one
human/assistant turn; only the assistant action string contributes to loss.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageFile

try:
    from .gui_grounding_protocol import (
        MIND2WEB_PROMPT_PROTOCOLS,
        TARGET_GROUNDING,
        mind2web_crop_plan,
        mind2web_prompt,
        parse_target_action,
    )
except ImportError:  # Direct execution: python scripts/data/<this file>.py
    from gui_grounding_protocol import (
        MIND2WEB_PROMPT_PROTOCOLS,
        TARGET_GROUNDING,
        mind2web_crop_plan,
        mind2web_prompt,
        parse_target_action,
    )


Image.MAX_IMAGE_PIXELS = 250_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = True

DEFAULT_ROOT = Path("/home/ubuntu/datasets/lladao_gui_120k")
DEFAULT_COUNT = 20_000
DEFAULT_SEED = 42
MIND2WEB_PUBLISHED_TRAIN_ROWS = 7_775

SOURCE_REVISIONS = {
    "mind2web": {
        "repo": "osunlp/Multimodal-Mind2Web",
        "revision": "1b4c6a8cf9f77b7a5e0d641959935c80c4a05889",
        "split": "train",
    },
    "weblinx_meta": {
        "repo": "McGill-NLP/WebLINX",
        "revision": "a30ff2cbecb75f2d04fac75a2b92721c6c9e3f13",
        "split": "train",
    },
    "weblinx_raw": {
        "repo": "McGill-NLP/WebLINX-full",
        "revision": "36ef9f79b43df50e25b7f3b68e5c9f6ccf4160e8",
        "split": "train",
    },
    "rico_widget_caption": {
        "repo": "bevaya/RICO-WidgetCaptioning",
        "revision": "6ec57b56bebd722b9c646c78d0f34e1199b6d7a9",
        "split": "train",
        "upstream": "google-research-datasets/widget-caption + RICO",
    },
    "os_atlas": {
        "repo": "OS-Copilot/OS-Atlas-data",
        "revision": "c129865bf1b4577fea978e14cb882e1cacb45c9a",
        "split": "train",
    },
    "os_atlas_web_annotations": {
        "repo": "andersonbcdefg/os-atlas-fineweb-annotations",
        "revision": "ffdfa0ed02fab17669754114c6502089905d1870",
        "upstream": "OS-Copilot/OS-Atlas-data web_domain/fineweb",
    },
    "os_atlas_web_images": {
        "repo": "andersonbcdefg/osatlas-fineweb-images-filtered-combined",
        "revision": "24dafb8ba88b136c0160f99f8a7a5f6a5b54d862",
        "upstream": "OS-Copilot/OS-Atlas-data web_domain/fineweb",
    },
    "os_atlas_desktop": {
        "repo": "maharshpatelx/osatlas-windows-combined",
        "revision": "f86c493c99ee266be1d6f4d20d0623e977304f10",
        "upstream": "OS-Copilot/OS-Atlas-data desktop_domain/windows",
    },
}

CONVERSATION_TYPE = pa.list_(
    pa.struct([pa.field("from", pa.string()), pa.field("value", pa.string())])
)
IMAGE_TYPE = pa.struct(
    [pa.field("bytes", pa.binary()), pa.field("path", pa.string())]
)
OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string()),
        pa.field("source", pa.string()),
        pa.field("image", IMAGE_TYPE),
        pa.field("conversations", CONVERSATION_TYPE),
        pa.field("metadata", pa.string()),
    ]
)

ACTION_RE = re.compile(
    r"^(lclick|hover|type_in) \[(\d+),(\d+),(\d+),(\d+)\](?: (.*))?$"
)
EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
PASSWORD_RE = re.compile(r"(?i)(password\s*[:=]\s*)([^\s,;]+)")


def log(message: str) -> None:
    print(message, flush=True)


def stable_int(*parts: Any) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def stable_sample(items: Sequence[Any], count: int, seed: int, key) -> list[Any]:
    if count > len(items):
        raise ValueError(f"Cannot select {count:,} rows from a pool of {len(items):,}")
    return sorted(items, key=lambda item: stable_int(seed, key(item)))[:count]


def redact_sensitive(text: str) -> str:
    text = EMAIL_RE.sub("<EMAIL>", text)
    return PASSWORD_RE.sub(r"\1<REDACTED>", text)


def compact_text(value: Any, limit: int = 2_000) -> str:
    text = " ".join(str(value or "").replace("\x00", " ").split())
    return redact_sensitive(text)[:limit]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def valid_scaled_bbox(values: Sequence[float], scale: int = 1000) -> list[int]:
    if len(values) != 4:
        raise ValueError("A bounding box must have four coordinates")
    x1, y1, x2, y2 = [int(clamp(round(value), 0, scale)) for value in values]
    if x2 <= x1:
        x1, x2 = (scale - 1, scale) if x1 >= scale else (x1, x1 + 1)
    if y2 <= y1:
        y1, y2 = (scale - 1, scale) if y1 >= scale else (y1, y1 + 1)
    return [x1, y1, x2, y2]


def scale_unit_bbox(bbox: Sequence[float], scale: int = 1000) -> list[int]:
    return valid_scaled_bbox([float(value) * scale for value in bbox], scale)


def normalize_bbox(
    bbox: Sequence[float], width: float, height: float, scale: int = 1000
) -> list[int]:
    if width <= 0 or height <= 0 or len(bbox) != 4:
        raise ValueError("Invalid image dimensions or bbox")
    x1, y1, x2, y2 = (float(v) for v in bbox)
    x1, x2 = sorted((clamp(x1, 0, width), clamp(x2, 0, width)))
    y1, y2 = sorted((clamp(y1, 0, height), clamp(y2, 0, height)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Degenerate bbox: {bbox}")
    return valid_scaled_bbox(
        [
            scale * x1 / width,
            scale * y1 / height,
            scale * x2 / width,
            scale * y2 / height,
        ],
        scale,
    )


def action_string(action: str, bbox_1000: Sequence[int], value: str = "") -> str:
    coords = ",".join(str(int(v)) for v in bbox_1000)
    output = f"{action} [{coords}]"
    if action == "type_in" and value:
        output += f" {compact_text(value, limit=512)}"
    return output


def make_record(
    *,
    sample_id: str,
    source: str,
    image_bytes: bytes,
    image_path: str,
    prompt: str,
    answer: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    if not image_bytes:
        raise ValueError("Empty image")
    if ACTION_RE.fullmatch(answer) is None:
        raise ValueError(f"Invalid action string: {answer}")
    return {
        "sample_id": sample_id,
        "source": source,
        "image": {"bytes": image_bytes, "path": image_path},
        "conversations": [
            {"from": "human", "value": f"<image>\n{prompt.strip()}"},
            {"from": "gpt", "value": answer},
        ],
        "metadata": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
    }


class ShardedWriter:
    def __init__(self, output_dir: Path, shard_size: int = 1_000):
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.buffer: list[dict[str, Any]] = []
        self.shard_index = 0
        self.count = 0
        output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        self.buffer.append(record)
        self.count += 1
        if len(self.buffer) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        final_path = self.output_dir / f"shard-{self.shard_index:05d}.parquet"
        temp_path = final_path.with_suffix(".parquet.tmp")
        table = pa.Table.from_pylist(self.buffer, schema=OUTPUT_SCHEMA)
        pq.write_table(
            table,
            temp_path,
            compression="zstd",
            compression_level=6,
            row_group_size=min(256, len(self.buffer)),
            use_dictionary=["source"],
        )
        os.replace(temp_path, final_path)
        self.buffer.clear()
        self.shard_index += 1

    def close(self) -> int:
        self.flush()
        return self.count


@dataclass(frozen=True)
class RowRef:
    path: str
    row_group: int
    row_index: int
    sample_id: str
    payload: dict[str, Any]


def parquet_files(path: Path, pattern: str = "*.parquet") -> list[Path]:
    files = sorted(path.rglob(pattern))
    if not files:
        raise FileNotFoundError(f"No Parquet files found below {path}")
    return files


def prepare_output_dir(path: Path, force: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not force:
            raise FileExistsError(f"Output already exists: {path}; pass --force to rebuild")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def choose_mind2web_candidate(raw_candidates: Sequence[str]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for raw in raw_candidates or []:
        try:
            candidate = json.loads(raw) if isinstance(raw, str) else raw
            attrs = candidate.get("attributes", {})
            attrs = json.loads(attrs) if isinstance(attrs, str) else attrs
            rect = attrs.get("bounding_box_rect")
            if rect:
                candidate = dict(candidate)
                candidate["_attrs"] = attrs
                candidates.append(candidate)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    if not candidates:
        raise ValueError("No positive candidate with bounding_box_rect")
    return next((x for x in candidates if x.get("is_original_target")), candidates[0])


def mind2web_metadata(
    row: dict[str, Any], prompt_protocol: str = TARGET_GROUNDING
) -> dict[str, Any]:
    operation = row.get("operation", {})
    operation = json.loads(operation) if isinstance(operation, str) else operation
    candidate = choose_mind2web_candidate(row.get("pos_candidates") or [])
    attributes = candidate["_attrs"]
    raw_rect = attributes["bounding_box_rect"]
    x, y, width, height = [float(v) for v in str(raw_rect).split(",")]
    if width <= 0 or height <= 0:
        raise ValueError("Invalid Mind2Web target rectangle")

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
    prompt = mind2web_prompt(
        prompt_protocol,
        target,
        confirmed_task=row.get("confirmed_task"),
        action_reprs=row.get("action_reprs") or [],
        target_action_index=row.get("target_action_index", 0),
    )

    return {
        "action_uid": str(row["action_uid"]),
        "annotation_id": str(row.get("annotation_id") or ""),
        "website": str(row.get("website") or ""),
        "bbox": [x, y, x + width, y + height],
        "action": target.action,
        "operation": target.operation,
        "value": target.value,
        "prompt": prompt,
        "prompt_protocol": prompt_protocol,
        "target_description": target.description,
        "target_role": target.role,
        "target_action_repr": target.raw_target_action_repr,
    }


def target_center_crop(
    image: Image.Image,
    bbox: Sequence[float],
    *,
    crop_size: int,
    seed: int,
    sample_id: str,
    variant: int,
    randomize: bool,
) -> tuple[Image.Image, list[float], list[int]]:
    image = image.convert("RGB")
    image_width, image_height = image.size
    x1, y1, x2, y2 = [float(v) for v in bbox]
    if x2 <= 0 or y2 <= 0 or x1 >= image_width or y1 >= image_height:
        raise ValueError(f"Target bbox outside image: {bbox} vs {image.size}")
    # DOM-derived rectangles occasionally overshoot a screenshot edge by a
    # few pixels. Preserve the visible intersection, which is also what the
    # paper's OCR realignment step would have to operate on.
    x1, x2 = clamp(x1, 0, image_width), clamp(x2, 0, image_width)
    y1, y2 = clamp(y1, 0, image_height), clamp(y2, 0, image_height)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Degenerate target bbox after clipping: {bbox} vs {image.size}")

    crop_width = min(image_width, max(crop_size, math.ceil(x2 - x1) + 64))
    crop_height = min(image_height, max(crop_size, math.ceil(y2 - y1) + 64))
    x_low = max(0, math.ceil(x2 - crop_width))
    x_high = min(math.floor(x1), image_width - crop_width)
    y_low = max(0, math.ceil(y2 - crop_height))
    y_high = min(math.floor(y1), image_height - crop_height)
    if x_low > x_high or y_low > y_high:
        raise ValueError("Unable to construct target-preserving crop")

    if not randomize:
        crop_x = round((x_low + x_high) / 2)
        crop_y = round((y_low + y_high) / 2)
    else:
        rng = random.Random(stable_int(seed, sample_id, "crop", variant))
        crop_x = rng.randint(x_low, x_high)
        crop_y = rng.randint(y_low, y_high)

    crop_box = [crop_x, crop_y, crop_x + crop_width, crop_y + crop_height]
    cropped = image.crop(tuple(crop_box))
    shifted_bbox = [x1 - crop_x, y1 - crop_y, x2 - crop_x, y2 - crop_y]
    return cropped, shifted_bbox, crop_box


def encode_jpeg(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90, optimize=True)
    return buffer.getvalue()


def build_mind2web(
    raw_dir: Path,
    output_dir: Path,
    *,
    seed: int,
    crop_size: int,
    shard_size: int,
    prompt_protocol: str = TARGET_GROUNDING,
    target_count: int = DEFAULT_COUNT,
    random_crop: bool = True,
) -> dict[str, Any]:
    if target_count <= 0:
        raise ValueError("Mind2Web target_count must be positive")
    files = parquet_files(raw_dir / "data", "train-*.parquet")
    refs: list[RowRef] = []
    metadata_columns = [
        "action_uid",
        "operation",
        "pos_candidates",
        "website",
        "annotation_id",
        "confirmed_task",
        "action_reprs",
        "target_action_index",
        "target_action_reprs",
    ]
    raw_rows = 0
    skipped = 0
    skipped_not_visible = 0
    rejections: list[dict[str, str]] = []
    for path in files:
        pf = pq.ParquetFile(path)
        for row_group in range(pf.num_row_groups):
            rows = pf.read_row_group(row_group, columns=metadata_columns).to_pylist()
            screenshots = (
                pf.read_row_group(row_group, columns=["screenshot"])
                .column(0)
                .to_pylist()
            )
            for row_index, row in enumerate(rows):
                raw_rows += 1
                try:
                    meta = mind2web_metadata(row, prompt_protocol)
                    with Image.open(io.BytesIO(screenshots[row_index]["bytes"])) as image:
                        image_width, image_height = image.size
                    x1, y1, x2, y2 = meta["bbox"]
                    if (
                        x2 <= 0
                        or y2 <= 0
                        or x1 >= image_width
                        or y1 >= image_height
                    ):
                        skipped_not_visible += 1
                        rejections.append(
                            {
                                "action_uid": str(row.get("action_uid") or ""),
                                "reason": "target_bbox_outside_screenshot",
                            }
                        )
                        continue
                    refs.append(
                        RowRef(str(path), row_group, row_index, meta["action_uid"], meta)
                    )
                except Exception as exc:
                    skipped += 1
                    rejections.append(
                        {
                            "action_uid": str(row.get("action_uid") or ""),
                            "reason": f"{type(exc).__name__}: {exc}",
                        }
                    )
    if raw_rows != MIND2WEB_PUBLISHED_TRAIN_ROWS:
        raise RuntimeError(
            f"Pinned Mind2Web train split has {raw_rows:,} rows, expected "
            f"{MIND2WEB_PUBLISHED_TRAIN_ROWS:,}"
        )
    if not refs:
        raise RuntimeError("Mind2Web yielded no eligible training rows")

    # The paper allocates 20K rows to Mind2Web although the public train split
    # has only 7,341 usable coordinate targets.  Retain every eligible target,
    # then repeat them as evenly as possible with deterministic random crop
    # variants.  When a smaller count is requested, select targets by a stable
    # seeded ordering rather than depending on Parquet file order.
    refs_by_id = {ref.sample_id: ref for ref in refs}
    selected = [
        (refs_by_id[sample_id], variant)
        for sample_id, variant in mind2web_crop_plan(
            list(refs_by_id), target_count, seed
        )
    ]

    variants_by_location: dict[tuple[str, int, int], list[tuple[RowRef, int]]] = defaultdict(list)
    for ref, variant in selected:
        variants_by_location[(ref.path, ref.row_group, ref.row_index)].append((ref, variant))

    writer = ShardedWriter(output_dir, shard_size)
    for path in files:
        pf = pq.ParquetFile(path)
        for row_group in range(pf.num_row_groups):
            wanted_indexes = {
                key[2]
                for key in variants_by_location
                if key[0] == str(path) and key[1] == row_group
            }
            if not wanted_indexes:
                continue
            images = pf.read_row_group(row_group, columns=["screenshot"]).column(0).to_pylist()
            for row_index in sorted(wanted_indexes):
                image_value = images[row_index]
                image = Image.open(io.BytesIO(image_value["bytes"]))
                for ref, variant in variants_by_location[(str(path), row_group, row_index)]:
                    meta = ref.payload
                    cropped, shifted_bbox, crop_box = target_center_crop(
                        image,
                        meta["bbox"],
                        crop_size=crop_size,
                        seed=seed,
                        sample_id=ref.sample_id,
                        variant=variant,
                        randomize=random_crop,
                    )
                    bbox_1000 = normalize_bbox(shifted_bbox, *cropped.size)
                    answer = action_string(meta["action"], bbox_1000, meta["value"])
                    sample_id = f"mind2web:{ref.sample_id}:crop{variant}"
                    provenance = {
                        **SOURCE_REVISIONS["mind2web"],
                        "original_id": ref.sample_id,
                        "annotation_id": meta["annotation_id"],
                        "website": meta["website"],
                        "source_operation": meta["operation"],
                        "target_action_repr": meta["target_action_repr"],
                        "target_description": meta["target_description"],
                        "target_role": meta["target_role"],
                        "prompt_protocol": meta["prompt_protocol"],
                        "source_bbox_xyxy": meta["bbox"],
                        "crop_xyxy": crop_box,
                        "bbox_1000": bbox_1000,
                        "crop_variant": variant,
                        "preprocessing": (
                            "deterministic seeded random target-preserving crop; source DOM bbox"
                            if random_crop
                            else "deterministic centered target-preserving crop; source DOM bbox"
                        ),
                        "paper_approximation": "Paper did not publish crop IDs/seed or OCR realignment code",
                    }
                    writer.write(
                        make_record(
                            sample_id=sample_id,
                            source="mind2web",
                            image_bytes=encode_jpeg(cropped),
                            image_path=f"{sample_id}.jpg",
                            prompt=meta["prompt"],
                            answer=answer,
                            metadata=provenance,
                        )
                    )
    written = writer.close()
    if written != len(selected):
        raise RuntimeError(f"Mind2Web wrote {written:,}, expected {len(selected):,}")
    rejection_path = output_dir / "rejections.json"
    rejection_path.write_text(
        json.dumps(rejections, indent=2, ensure_ascii=False) + "\n"
    )
    return {
        "rows": written,
        "published_train_rows": raw_rows,
        "eligible_source_rows": len(refs),
        "target_rows": target_count,
        "repeated_crop_variants": max(0, target_count - len(refs)),
        "random_target_preserving_crop": random_crop,
        "prompt_protocol": prompt_protocol,
        "skipped_invalid_or_missing_bbox": skipped,
        "skipped_target_not_visible": skipped_not_visible,
        "rejection_audit": str(rejection_path.relative_to(output_dir.parent)),
    }


def rico_metadata(row: dict[str, Any], identity: str, seed: int) -> dict[str, Any]:
    captions = [compact_text(x, 500) for x in (row.get("captions") or []) if compact_text(x)]
    bbox = [float(v) for v in (row.get("bbox") or [])]
    if not captions or len(bbox) != 4 or not (0 <= bbox[0] < bbox[2] <= 1 and 0 <= bbox[1] < bbox[3] <= 1):
        raise ValueError("Invalid RICO caption or bbox")
    caption = captions[stable_int(seed, identity, "caption") % len(captions)]
    return {
        "screen_id": str(row.get("screenId")),
        "caption": caption,
        "bbox": bbox,
        "prompt": f'Locate and click the mobile UI element described as: "{caption}".',
    }


def build_rico(
    raw_dir: Path,
    output_dir: Path,
    *,
    count: int,
    seed: int,
    shard_size: int,
) -> dict[str, Any]:
    files = parquet_files(raw_dir / "data", "train-*.parquet")
    refs: list[RowRef] = []
    skipped = 0
    for path in files:
        pf = pq.ParquetFile(path)
        for row_group in range(pf.num_row_groups):
            rows = pf.read_row_group(
                row_group, columns=["screenId", "captions", "bbox"]
            ).to_pylist()
            for row_index, row in enumerate(rows):
                identity = f"{path.name}:{row_group}:{row_index}:{row.get('screenId')}"
                try:
                    meta = rico_metadata(row, identity, seed)
                    refs.append(RowRef(str(path), row_group, row_index, identity, meta))
                except Exception:
                    skipped += 1
    selected = stable_sample(refs, count, seed, lambda x: x.sample_id)
    wanted = {(x.path, x.row_group, x.row_index): x for x in selected}
    writer = ShardedWriter(output_dir, shard_size)
    for path in files:
        pf = pq.ParquetFile(path)
        for row_group in range(pf.num_row_groups):
            indexes = sorted(
                key[2] for key in wanted if key[0] == str(path) and key[1] == row_group
            )
            if not indexes:
                continue
            images = pf.read_row_group(row_group, columns=["image"]).column(0).to_pylist()
            for row_index in indexes:
                ref = wanted[(str(path), row_group, row_index)]
                image_value = images[row_index]
                bbox_1000 = scale_unit_bbox(ref.payload["bbox"])
                answer = action_string("lclick", bbox_1000)
                sample_id = f"rico_widget_caption:{stable_int(ref.sample_id):016x}"
                provenance = {
                    **SOURCE_REVISIONS["rico_widget_caption"],
                    "original_id": ref.sample_id,
                    "screen_id": ref.payload["screen_id"],
                    "source_bbox_normalized": ref.payload["bbox"],
                    "bbox_1000": bbox_1000,
                    "preprocessing": "full image; source widget-caption bbox",
                    "paper_approximation": "Paper did not publish sample IDs or OCR realignment code",
                }
                writer.write(
                    make_record(
                        sample_id=sample_id,
                        source="rico_widget_caption",
                        image_bytes=image_value["bytes"],
                        image_path=image_value.get("path") or f"{sample_id}.jpg",
                        prompt=ref.payload["prompt"],
                        answer=answer,
                        metadata=provenance,
                    )
                )
    written = writer.close()
    if written != count:
        raise RuntimeError(f"RICO wrote {written:,}, expected {count:,}")
    return {"rows": written, "eligible_source_rows": len(refs), "skipped": skipped}


def parse_python_call(value: str) -> tuple[str, dict[str, Any]]:
    node = ast.parse(value, mode="eval").body
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise ValueError(f"Not a simple action call: {value}")
    kwargs = {item.arg: ast.literal_eval(item.value) for item in node.keywords if item.arg}
    return node.func.id.lower(), kwargs


def parse_weblinx_candidate_bbox(candidates: str, uid: str) -> list[float]:
    marker = re.compile(rf"\(uid\s*=\s*{re.escape(uid)}\)\s*(.*?)(?=\n\(uid\s*=|\Z)", re.S)
    match = marker.search(candidates or "")
    if not match:
        raise ValueError(f"Target uid {uid} missing from candidates")
    bbox = re.search(
        r"\[\[bbox\]\]\s*x=([-+0-9.eE]+)\s+y=([-+0-9.eE]+)\s+width=([-+0-9.eE]+)\s+height=([-+0-9.eE]+)",
        match.group(1),
    )
    if not bbox:
        raise ValueError("Target candidate has no bbox")
    x, y, width, height = [float(v) for v in bbox.groups()]
    return [x, y, x + width, y + height]


def weblinx_prompt(replay: Any, turn_index: int, history_depth: int) -> str:
    instructor_messages: list[str] = []
    previous_actions: list[str] = []
    for turn in replay[:turn_index]:
        if turn.type == "chat" and turn.get("speaker") == "instructor":
            utterance = compact_text(turn.get("utterance"), 1_000)
            if utterance:
                instructor_messages.append(utterance)
        elif turn.type == "browser" and history_depth > 0:
            previous_actions.append(compact_text(turn.format_text(), 300))
    instruction = instructor_messages[-1] if instructor_messages else "Continue the current web task."
    prompt = f"Follow the user's instruction and predict the next GUI action.\nInstruction: {instruction}"
    if history_depth and previous_actions:
        history = previous_actions[-history_depth:]
        prompt += "\nRecent actions:\n" + "\n".join(f"- {item}" for item in history)
    return prompt


def plan_weblinx(csv_path: Path, count: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    eligible: list[dict[str, Any]] = []
    reasons: dict[str, int] = defaultdict(int)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), start=2):
            try:
                intent, kwargs = parse_python_call(row["action"])
                action_map = {
                    "click": "lclick",
                    "submit": "lclick",
                    "text_input": "type_in",
                    "change": "type_in",
                }
                if intent not in action_map:
                    reasons[f"unsupported:{intent}"] += 1
                    continue
                uid = kwargs.get("uid")
                if not uid:
                    reasons["missing_uid"] += 1
                    continue
                viewport = re.fullmatch(r"\s*([0-9.]+)h\s*x\s*([0-9.]+)w\s*", row["viewport"])
                if not viewport:
                    reasons["invalid_viewport"] += 1
                    continue
                height, width = [float(v) for v in viewport.groups()]
                bbox = parse_weblinx_candidate_bbox(row["candidates"], str(uid))
                bbox_1000 = normalize_bbox(bbox, width, height)
                identity = f"{row['demo']}:{row['turn']}:{line_number}"
                eligible.append(
                    {
                        "identity": identity,
                        "demo": row["demo"],
                        "turn": int(row["turn"]),
                        "intent": intent,
                        "action": action_map[intent],
                        "value": str(kwargs.get("text") or kwargs.get("value") or ""),
                        "bbox": bbox,
                        "bbox_1000": bbox_1000,
                        "viewport": [width, height],
                        "prompt_variant": 0,
                    }
                )
            except Exception:
                reasons["parse_error"] += 1

    if not eligible:
        raise RuntimeError("WebLINX yielded no eligible rows")
    if len(eligible) >= count:
        selected = stable_sample(eligible, count, seed, lambda x: x["identity"])
    else:
        # The processed WebLINX train split has fewer than 20K element-targeted
        # click/type rows. Keep every original row, then add deterministic prompt
        # variants with progressively longer histories. The screenshot, target,
        # and action stay unchanged and the variant is recorded in provenance.
        selected = list(eligible)
        prompt_variant = 1
        while len(selected) < count:
            extras = sorted(
                eligible,
                key=lambda x: stable_int(
                    seed, x["identity"], "prompt_variant", prompt_variant
                ),
            )
            take = min(count - len(selected), len(extras))
            selected.extend(
                {**item, "prompt_variant": prompt_variant}
                for item in extras[:take]
            )
            prompt_variant += 1
    return selected, dict(reasons)


def build_weblinx(
    meta_dir: Path,
    raw_dir: Path,
    output_dir: Path,
    *,
    count: int,
    seed: int,
    shard_size: int,
) -> dict[str, Any]:
    try:
        import weblinx as wl
    except ImportError as exc:
        raise RuntimeError("Install the official parser with: pip install weblinx==0.3.2") from exc

    selected, reasons = plan_weblinx(meta_dir / "data" / "train.csv", count, seed)
    by_demo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in selected:
        by_demo[item["demo"]].append(item)

    base_dir = raw_dir / "demonstrations"
    writer = ShardedWriter(output_dir, shard_size)
    missing = 0
    for demo_name in sorted(by_demo):
        demo = wl.Demonstration(demo_name, base_dir=str(base_dir))
        replay = wl.Replay.from_demonstration(demo)
        for item in sorted(by_demo[demo_name], key=lambda x: (x["turn"], x["prompt_variant"])):
            try:
                turn = replay[item["turn"]]
                screenshot_path = Path(turn.get_screenshot_path())
                image_bytes = screenshot_path.read_bytes()
                history_depth = (0, 2, 4, 8)[min(item["prompt_variant"], 3)]
                prompt = weblinx_prompt(replay, item["turn"], history_depth)
                answer = action_string(item["action"], item["bbox_1000"], item["value"])
                sample_id = f"weblinx:{item['demo']}:{item['turn']}:prompt{item['prompt_variant']}"
                provenance = {
                    "metadata_repo": SOURCE_REVISIONS["weblinx_meta"],
                    "raw_repo": SOURCE_REVISIONS["weblinx_raw"],
                    "original_id": item["identity"],
                    "demo": item["demo"],
                    "turn": item["turn"],
                    "source_bbox_xyxy": item["bbox"],
                    "viewport_wh": item["viewport"],
                    "bbox_1000": item["bbox_1000"],
                    "prompt_variant": item["prompt_variant"],
                    "preprocessing": "full image; target candidate bbox; credential redaction",
                    "paper_approximation": "Paper did not publish sample IDs or OCR realignment code",
                }
                writer.write(
                    make_record(
                        sample_id=sample_id,
                        source="weblinx",
                        image_bytes=image_bytes,
                        image_path=str(screenshot_path.relative_to(raw_dir)),
                        prompt=prompt,
                        answer=answer,
                        metadata=provenance,
                    )
                )
            except Exception as exc:
                missing += 1
                if missing <= 10:
                    log(f"[weblinx] skip {item['identity']}: {exc}")
    written = writer.close()
    if written != count:
        raise RuntimeError(f"WebLINX wrote {written:,}, expected {count:,}; missing={missing:,}")
    return {
        "rows": written,
        "eligible_source_rows": len({x["identity"] for x in selected}),
        "selection_rejections": reasons,
        "missing": missing,
    }


def valid_normalized_element(element: dict[str, Any]) -> tuple[str, list[float]]:
    instruction = compact_text(element.get("instruction"), 1_000)
    bbox = [float(v) for v in (element.get("bbox") or [])]
    if not instruction or len(bbox) != 4:
        raise ValueError("Missing element instruction or bbox")
    if not (0 <= bbox[0] < bbox[2] <= 1 and 0 <= bbox[1] < bbox[3] <= 1):
        raise ValueError(f"Invalid normalized bbox: {bbox}")
    return instruction, bbox


def element_prompt(instruction: str, domain: str) -> str:
    return f'Locate and click the {domain} UI element described as: "{instruction}".'


def build_os_atlas_desktop(
    raw_dir: Path,
    output_dir: Path,
    *,
    count: int,
    seed: int,
    shard_size: int,
) -> dict[str, Any]:
    files = parquet_files(raw_dir / "data", "train-*.parquet")
    refs: list[RowRef] = []
    skipped = 0
    for path in files:
        pf = pq.ParquetFile(path)
        for row_group in range(pf.num_row_groups):
            rows = pf.read_row_group(
                row_group, columns=["img_filename", "elements"]
            ).to_pylist()
            for row_index, row in enumerate(rows):
                for element_index, element in enumerate(row.get("elements") or []):
                    identity = f"{path.name}:{row_group}:{row_index}:{element_index}"
                    try:
                        instruction, bbox = valid_normalized_element(element)
                        refs.append(
                            RowRef(
                                str(path),
                                row_group,
                                row_index,
                                identity,
                                {
                                    "element_index": element_index,
                                    "img_filename": row["img_filename"],
                                    "instruction": instruction,
                                    "bbox": bbox,
                                },
                            )
                        )
                    except Exception:
                        skipped += 1
    selected = stable_sample(refs, count, seed, lambda x: x.sample_id)
    wanted: dict[tuple[str, int, int], list[RowRef]] = defaultdict(list)
    for ref in selected:
        wanted[(ref.path, ref.row_group, ref.row_index)].append(ref)

    writer = ShardedWriter(output_dir, shard_size)
    for path in files:
        pf = pq.ParquetFile(path)
        for row_group in range(pf.num_row_groups):
            locations = [
                key for key in wanted if key[0] == str(path) and key[1] == row_group
            ]
            if not locations:
                continue
            rows = pf.read_row_group(row_group, columns=["image"]).column(0).to_pylist()
            for _, _, row_index in sorted(locations):
                image_value = rows[row_index]
                for ref in wanted[(str(path), row_group, row_index)]:
                    meta = ref.payload
                    bbox_1000 = scale_unit_bbox(meta["bbox"])
                    sample_id = f"os_atlas_desktop:{stable_int(ref.sample_id):016x}"
                    provenance = {
                        **SOURCE_REVISIONS["os_atlas_desktop"],
                        "original_id": ref.sample_id,
                        "img_filename": meta["img_filename"],
                        "source_bbox_normalized": meta["bbox"],
                        "bbox_1000": bbox_1000,
                        "preprocessing": "full image; source OS-Atlas Windows bbox",
                        "paper_approximation": "Paper did not publish sample IDs or OCR realignment code",
                    }
                    writer.write(
                        make_record(
                            sample_id=sample_id,
                            source="os_atlas_desktop",
                            image_bytes=image_value["bytes"],
                            image_path=image_value.get("path") or meta["img_filename"],
                            prompt=element_prompt(meta["instruction"], "desktop"),
                            answer=action_string("lclick", bbox_1000),
                            metadata=provenance,
                        )
                    )
    written = writer.close()
    if written != count:
        raise RuntimeError(f"OS-Atlas desktop wrote {written:,}, expected {count:,}")
    return {"rows": written, "eligible_source_rows": len(refs), "skipped": skipped}


def zip_member_lookup(
    archive: Any, requested: str, names: Sequence[str] | None = None
) -> str:
    names = archive.namelist() if names is None else names
    if requested in names:
        return requested
    normalized = requested.lstrip("./")
    exact = next((name for name in names if name.lstrip("./") == normalized), None)
    if exact:
        return exact
    basename = Path(requested).name
    matches = [name for name in names if Path(name).name == basename]
    if len(matches) == 1:
        return matches[0]
    raise KeyError(f"Could not uniquely locate {requested} in archive")


def mobile_element_refs(
    records: Sequence[dict[str, Any]], source_variant: str
) -> tuple[list[dict[str, Any]], int]:
    refs: list[dict[str, Any]] = []
    skipped = 0
    for row_index, row in enumerate(records):
        filename = str(row.get("img_filename") or "")
        for element_index, element in enumerate(row.get("elements") or []):
            try:
                instruction, bbox = valid_normalized_element(element)
                refs.append(
                    {
                        "identity": f"{source_variant}:{filename}:{element_index}",
                        "source_variant": source_variant,
                        "filename": filename,
                        "row_index": row_index,
                        "element_index": element_index,
                        "instruction": instruction,
                        "bbox": bbox,
                    }
                )
            except Exception:
                skipped += 1
    return refs, skipped


def build_os_atlas_mobile(
    raw_dir: Path,
    output_dir: Path,
    *,
    count: int,
    seed: int,
    shard_size: int,
) -> dict[str, Any]:
    import zipfile

    domain_dir = raw_dir / "mobile_domain"
    with (domain_dir / "uibert_raw.json").open(encoding="utf-8") as handle:
        uibert_records = json.load(handle)
    with (domain_dir / "aw_mobile.json").open(encoding="utf-8") as handle:
        aw_records = json.load(handle)
    uibert_refs, skipped_uibert = mobile_element_refs(uibert_records, "uibert")
    aw_refs, skipped_aw = mobile_element_refs(aw_records, "aw")

    if len(uibert_refs) >= count:
        selected = stable_sample(uibert_refs, count, seed, lambda x: x["identity"])
    else:
        selected = list(uibert_refs)
        selected.extend(
            stable_sample(
                aw_refs,
                count - len(selected),
                seed,
                lambda x: x["identity"],
            )
        )
    by_variant_and_image: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in selected:
        by_variant_and_image[(item["source_variant"], item["filename"])].append(item)

    archive_paths = {
        "uibert": domain_dir / "UIBert.zip",
        "aw": domain_dir / "mobile_images.zip",
    }
    writer = ShardedWriter(output_dir, shard_size)
    for variant in ("uibert", "aw"):
        wanted_filenames = sorted(
            filename
            for source_variant, filename in by_variant_and_image
            if source_variant == variant
        )
        if not wanted_filenames:
            continue
        with zipfile.ZipFile(archive_paths[variant]) as archive:
            archive_names = archive.namelist()
            archive_name_set = set(archive_names)
            for filename in wanted_filenames:
                member = (
                    filename
                    if filename in archive_name_set
                    else zip_member_lookup(archive, filename, archive_names)
                )
                image_bytes = archive.read(member)
                for item in by_variant_and_image[(variant, filename)]:
                    bbox_1000 = scale_unit_bbox(item["bbox"])
                    sample_id = f"os_atlas_mobile:{stable_int(item['identity']):016x}"
                    provenance = {
                        **SOURCE_REVISIONS["os_atlas"],
                        "original_id": item["identity"],
                        "source_variant": variant,
                        "img_filename": filename,
                        "source_bbox_normalized": item["bbox"],
                        "bbox_1000": bbox_1000,
                        "preprocessing": "full image; source OS-Atlas mobile bbox",
                        "paper_approximation": "Paper did not publish mobile sub-source mix, sample IDs, or OCR code",
                    }
                    writer.write(
                        make_record(
                            sample_id=sample_id,
                            source="os_atlas_mobile",
                            image_bytes=image_bytes,
                            image_path=f"{variant}/{member}",
                            prompt=element_prompt(item["instruction"], "mobile"),
                            answer=action_string("lclick", bbox_1000),
                            metadata=provenance,
                        )
                    )
    written = writer.close()
    if written != count:
        raise RuntimeError(f"OS-Atlas mobile wrote {written:,}, expected {count:,}")
    return {
        "rows": written,
        "eligible_uibert_rows": len(uibert_refs),
        "eligible_aw_rows": len(aw_refs),
        "selected_uibert_rows": sum(x["source_variant"] == "uibert" for x in selected),
        "selected_aw_rows": sum(x["source_variant"] == "aw" for x in selected),
        "skipped": skipped_uibert + skipped_aw,
    }


def build_os_atlas_web(
    annotations_dir: Path,
    images_dir: Path,
    output_dir: Path,
    *,
    count: int,
    seed: int,
    shard_size: int,
) -> dict[str, Any]:
    annotation_files = parquet_files(annotations_dir / "data", "train-*.parquet")
    image_files = parquet_files(images_dir / "data", "train-*.parquet")
    # The public image repack stores the original filename in image.path and
    # converts PNG files to JPEG. Join by the extension-free source stem.
    available_images: dict[str, tuple[str, int, int]] = {}
    for path in image_files:
        pf = pq.ParquetFile(path)
        if "image" not in pf.schema_arrow.names:
            raise RuntimeError(f"Image shard lacks image: {path}")
        for row_group in range(pf.num_row_groups):
            values = (
                pf.read_row_group(row_group, columns=["image.path"])
                .column(0)
                .to_pylist()
            )
            for row_index, value in enumerate(values):
                filename = str(value.get("path") or "")
                if filename:
                    available_images[Path(filename).stem] = (
                        str(path),
                        row_group,
                        row_index,
                    )

    refs_by_id: dict[str, dict[str, Any]] = {}
    skipped = 0
    duplicate_annotations = 0
    for path in annotation_files:
        pf = pq.ParquetFile(path)
        for row_group in range(pf.num_row_groups):
            filenames = (
                pf.read_row_group(row_group, columns=["img_filename"])
                .column(0)
                .to_pylist()
            )
            matched_indexes = [
                row_index
                for row_index, filename in enumerate(filenames)
                if Path(str(filename)).stem in available_images
            ]
            if not matched_indexes:
                continue
            elements = (
                pf.read_row_group(row_group, columns=["elements"])
                .column(0)
                .to_pylist()
            )
            for row_index in matched_indexes:
                filename = str(filenames[row_index])
                image_key = Path(filename).stem
                for element_index, element in enumerate(elements[row_index] or []):
                    try:
                        instruction, bbox = valid_normalized_element(element)
                        identity = f"fineweb:{filename}:{element_index}"
                        if identity in refs_by_id:
                            duplicate_annotations += 1
                            continue
                        refs_by_id[identity] = {
                            "identity": identity,
                            "filename": filename,
                            "image_key": image_key,
                            "element_index": element_index,
                            "instruction": instruction,
                            "bbox": bbox,
                        }
                    except Exception:
                        skipped += 1
    refs = list(refs_by_id.values())
    if len(refs) < count:
        raise RuntimeError(
            f"Downloaded OS-Atlas web image shards cover only {len(refs):,} valid elements; "
            "download additional leading image shards and rerun"
        )
    selected = stable_sample(refs, count, seed, lambda x: x["identity"])
    by_location: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for item in selected:
        by_location[available_images[item["image_key"]]].append(item)

    writer = ShardedWriter(output_dir, shard_size)
    for path in image_files:
        pf = pq.ParquetFile(path)
        for row_group in range(pf.num_row_groups):
            locations = [
                key for key in by_location if key[0] == str(path) and key[1] == row_group
            ]
            if not locations:
                continue
            rows = pf.read_row_group(row_group, columns=["image"]).column(0).to_pylist()
            for _, _, row_index in sorted(locations):
                image_value = rows[row_index]
                for item in by_location[(str(path), row_group, row_index)]:
                    bbox_1000 = scale_unit_bbox(item["bbox"])
                    sample_id = f"os_atlas_web:{stable_int(item['identity']):016x}"
                    provenance = {
                        "annotations_repo": SOURCE_REVISIONS["os_atlas_web_annotations"],
                        "images_repo": SOURCE_REVISIONS["os_atlas_web_images"],
                        "original_id": item["identity"],
                        "img_filename": item["filename"],
                        "source_bbox_normalized": item["bbox"],
                        "bbox_1000": bbox_1000,
                        "preprocessing": "full image; source OS-Atlas FineWeb bbox",
                        "paper_approximation": "Paper did not publish web sub-source mix, sample IDs, or OCR code",
                    }
                    writer.write(
                        make_record(
                            sample_id=sample_id,
                            source="os_atlas_web",
                            image_bytes=image_value["bytes"],
                            image_path=image_value.get("path") or item["filename"],
                            prompt=element_prompt(item["instruction"], "web"),
                            answer=action_string("lclick", bbox_1000),
                            metadata=provenance,
                        )
                    )
    written = writer.close()
    if written != count:
        raise RuntimeError(f"OS-Atlas web wrote {written:,}, expected {count:,}")
    return {
        "rows": written,
        "available_images": len(available_images),
        "eligible_source_rows": len(refs),
        "duplicate_annotations_removed": duplicate_annotations,
        "skipped": skipped,
    }


def validate_output(
    parquet_root: Path,
    expected_counts: dict[str, int],
    deep: bool = False,
) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    invalid = 0
    sample_ids: set[str] = set()
    duplicate_ids = 0
    image_hashes: set[str] = set()
    checked_images = 0
    checked_by_source: dict[str, int] = defaultdict(int)
    files = parquet_files(parquet_root)
    for path in files:
        pf = pq.ParquetFile(path)
        required = set(OUTPUT_SCHEMA.names)
        if not required.issubset(pf.schema_arrow.names):
            raise RuntimeError(f"Schema mismatch in {path}")
        for row_group in range(pf.num_row_groups):
            rows = pf.read_row_group(row_group).to_pylist()
            for row in rows:
                counts[row["source"]] += 1
                if row["sample_id"] in sample_ids:
                    duplicate_ids += 1
                sample_ids.add(row["sample_id"])
                conversations = row["conversations"]
                answer = conversations[-1]["value"] if conversations else ""
                match = ACTION_RE.fullmatch(answer)
                if not match:
                    invalid += 1
                    continue
                coords = [int(v) for v in match.groups()[1:5]]
                if not all(0 <= v <= 1000 for v in coords) or coords[2] <= coords[0] or coords[3] <= coords[1]:
                    invalid += 1
                source = row["source"]
                if deep or checked_by_source[source] < 200:
                    data = row["image"]["bytes"]
                    with Image.open(io.BytesIO(data)) as image:
                        image.verify()
                    image_hashes.add(hashlib.sha256(data).hexdigest())
                    checked_images += 1
                    checked_by_source[source] += 1
    if invalid:
        raise RuntimeError(f"Validation found {invalid:,} invalid rows")
    if duplicate_ids:
        raise RuntimeError(f"Validation found {duplicate_ids:,} duplicate sample IDs")
    bad_counts = {
        source: {"actual": counts.get(source, 0), "expected": expected_counts.get(source)}
        for source in sorted(set(counts) | set(expected_counts))
        if counts.get(source, 0) != expected_counts.get(source)
    }
    if bad_counts:
        raise RuntimeError(f"Unexpected source counts: {bad_counts}")
    return {
        "total_rows": sum(counts.values()),
        "rows_by_source": dict(sorted(counts.items())),
        "checked_images": checked_images,
        "checked_images_by_source": dict(sorted(checked_by_source.items())),
        "unique_checked_images": len(image_hashes),
        "parquet_files": len(files),
    }


def expected_counts_from_manifest(
    parquet_root: Path,
    fallback_count: int,
) -> dict[str, int]:
    manifest_path = parquet_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        expected = {
            source: int(details["rows"])
            for source, details in manifest.get("sources", {}).items()
            if isinstance(details, dict) and "rows" in details
        }
        if expected:
            return expected

    source_dirs = sorted(
        path.name
        for path in parquet_root.iterdir()
        if path.is_dir() and any(path.glob("*.parquet"))
    )
    return {source: fallback_count for source in source_dirs}


def hf_download_many(
    repo_id: str,
    revision: str,
    filenames: Sequence[str],
    local_dir: Path,
    *,
    workers: int,
) -> None:
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    from huggingface_hub import hf_hub_download

    unique_filenames = sorted(set(filenames))
    local_dir.mkdir(parents=True, exist_ok=True)
    pending_filenames = [
        filename
        for filename in unique_filenames
        if not (local_dir / filename).is_file()
        or (local_dir / filename).stat().st_size == 0
    ]
    present = len(unique_filenames) - len(pending_filenames)
    log(
        f"[download] {repo_id}@{revision[:8]}: {len(unique_filenames):,} files "
        f"({present:,} already present)"
    )
    if not pending_filenames:
        return

    def fetch(filename: str) -> str:
        last_error: Exception | None = None
        for attempt in range(5):
            try:
                return hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    revision=revision,
                    filename=filename,
                    local_dir=local_dir,
                )
            except Exception as exc:  # Network retries are intentionally broad.
                last_error = exc
                time.sleep(min(2**attempt, 15))
        raise RuntimeError(f"Failed to download {repo_id}/{filename}") from last_error

    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(fetch, name): name for name in pending_filenames}
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed == len(futures) or completed % 100 == 0:
                log(f"[download] {repo_id}: {completed:,}/{len(futures):,}")


def hf_lfs_download_many(
    repo_id: str,
    revision: str,
    filenames: Sequence[str],
    local_dir: Path,
    *,
    workers: int,
) -> None:
    """Download many public LFS files without one Hub HEAD request per file.

    WebLINX contains thousands of individually stored screenshots. Resolving
    every path with ``hf_hub_download`` quickly exhausts the anonymous Hub API
    quota, even though the underlying LFS objects remain available. Query path
    metadata in batches, authorize objects through the Git-LFS batch endpoint,
    and verify every downloaded object against its LFS SHA-256 instead.
    """

    import requests
    from huggingface_hub import HfApi
    from huggingface_hub.utils import build_hf_headers

    unique_filenames = sorted(set(filenames))
    local_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        filename
        for filename in unique_filenames
        if not (local_dir / filename).is_file()
        or (local_dir / filename).stat().st_size == 0
    ]
    present = len(unique_filenames) - len(pending)
    log(
        f"[download-lfs] {repo_id}@{revision[:8]}: {len(unique_filenames):,} files "
        f"({present:,} already present)"
    )
    if not pending:
        return

    api = HfApi()
    path_info: dict[str, Any] = {}
    for start in range(0, len(pending), 500):
        batch = pending[start : start + 500]
        for item in api.get_paths_info(
            repo_id,
            batch,
            revision=revision,
            repo_type="dataset",
        ):
            path_info[item.path] = item
    missing = [filename for filename in pending if filename not in path_info]
    if missing:
        raise RuntimeError(f"LFS path lookup missed {len(missing):,} requested files")

    filenames_by_oid: dict[str, list[str]] = defaultdict(list)
    object_sizes: dict[str, int] = {}
    for filename in pending:
        item = path_info[filename]
        if item.lfs is None:
            raise RuntimeError(f"Expected an LFS object for {repo_id}/{filename}")
        filenames_by_oid[item.lfs.sha256].append(filename)
        object_sizes[item.lfs.sha256] = item.lfs.size

    lfs_url = f"https://huggingface.co/datasets/{repo_id}.git/info/lfs/objects/batch"
    request_headers = build_hf_headers()
    request_headers.update(
        {
            "Accept": "application/vnd.git-lfs+json",
            "Content-Type": "application/vnd.git-lfs+json",
        }
    )
    actions: dict[str, dict[str, Any]] = {}
    oids = sorted(filenames_by_oid)
    for start in range(0, len(oids), 500):
        batch_oids = oids[start : start + 500]
        response = requests.post(
            lfs_url,
            headers=request_headers,
            json={
                "operation": "download",
                "transfers": ["basic"],
                "ref": {"name": revision},
                "objects": [
                    {"oid": oid, "size": object_sizes[oid]} for oid in batch_oids
                ],
            },
            timeout=120,
        )
        response.raise_for_status()
        for item in response.json().get("objects", []):
            if item.get("error"):
                raise RuntimeError(f"LFS batch error for {item['oid']}: {item['error']}")
            download = item.get("actions", {}).get("download")
            if not download:
                raise RuntimeError(f"LFS batch returned no download action for {item['oid']}")
            actions[item["oid"]] = download
    missing_actions = set(oids) - set(actions)
    if missing_actions:
        raise RuntimeError(f"LFS batch omitted {len(missing_actions):,} objects")

    def fetch(oid: str) -> int:
        action = actions[oid]
        last_error: Exception | None = None
        for attempt in range(5):
            try:
                response = requests.get(
                    action["href"],
                    headers=action.get("header") or {},
                    timeout=(20, 120),
                )
                response.raise_for_status()
                data = response.content
                if len(data) != object_sizes[oid]:
                    raise RuntimeError(
                        f"LFS size mismatch for {oid}: {len(data)} != {object_sizes[oid]}"
                    )
                if hashlib.sha256(data).hexdigest() != oid:
                    raise RuntimeError(f"LFS SHA-256 mismatch for {oid}")
                for filename in filenames_by_oid[oid]:
                    destination = local_dir / filename
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = destination.with_suffix(destination.suffix + ".lfs.tmp")
                    temporary.write_bytes(data)
                    os.replace(temporary, destination)
                return len(filenames_by_oid[oid])
            except Exception as exc:
                last_error = exc
                time.sleep(min(2**attempt, 15))
        raise RuntimeError(f"Failed to download LFS object {oid}") from last_error

    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(fetch, oid): oid for oid in oids}
        for future in as_completed(futures):
            completed += future.result()
            if completed == len(pending) or completed % 500 == 0:
                log(f"[download-lfs] {repo_id}: {completed:,}/{len(pending):,}")


def repo_files(repo_id: str, revision: str) -> list[str]:
    from huggingface_hub import HfApi

    return HfApi().list_repo_files(repo_id, repo_type="dataset", revision=revision)


def download_weblinx(root: Path, count: int, seed: int, workers: int) -> None:
    meta = SOURCE_REVISIONS["weblinx_meta"]
    raw = SOURCE_REVISIONS["weblinx_raw"]
    meta_dir = root / "raw" / "weblinx_meta"
    raw_dir = root / "raw" / "weblinx"
    hf_download_many(
        meta["repo"],
        meta["revision"],
        ["data/train.csv", "splits.json"],
        meta_dir,
        workers=min(workers, 2),
    )
    selected, _ = plan_weblinx(meta_dir / "data" / "train.csv", count, seed)
    selected_demos = sorted({item["demo"] for item in selected})
    replay_files = [f"demonstrations/{demo}/replay.json" for demo in selected_demos]
    hf_download_many(
        raw["repo"], raw["revision"], replay_files, raw_dir, workers=workers
    )

    try:
        import weblinx as wl
    except ImportError as exc:
        raise RuntimeError("Install the official parser with: pip install weblinx==0.3.2") from exc
    screenshots: set[str] = set()
    by_demo: dict[str, set[int]] = defaultdict(set)
    for item in selected:
        by_demo[item["demo"]].add(item["turn"])
    base_dir = raw_dir / "demonstrations"
    for demo_name, turn_indexes in sorted(by_demo.items()):
        replay = wl.Replay.from_demonstration(
            wl.Demonstration(demo_name, base_dir=str(base_dir))
        )
        for turn_index in turn_indexes:
            turn = replay[turn_index]
            screenshot = turn.get("state", {}).get("screenshot")
            if screenshot:
                screenshots.add(f"demonstrations/{demo_name}/screenshots/{screenshot}")
    hf_lfs_download_many(
        raw["repo"], raw["revision"], sorted(screenshots), raw_dir, workers=workers
    )


def download_selected(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    requested = args.sources or [
        "mind2web",
        "weblinx",
        "os_atlas_web",
        "os_atlas_mobile",
        "rico_widget_caption",
        "os_atlas_desktop",
    ]
    if "mind2web" in requested:
        spec = SOURCE_REVISIONS["mind2web"]
        files = [
            name
            for name in repo_files(spec["repo"], spec["revision"])
            if name.startswith("data/train-") and name.endswith(".parquet")
        ]
        hf_download_many(
            spec["repo"], spec["revision"], files, root / "raw" / "mind2web", workers=args.workers
        )
    if "rico_widget_caption" in requested:
        spec = SOURCE_REVISIONS["rico_widget_caption"]
        files = [f"data/train-{index:05d}-of-00021.parquet" for index in range(11)]
        hf_download_many(
            spec["repo"], spec["revision"], files, root / "raw" / "rico", workers=args.workers
        )
    if "os_atlas_desktop" in requested:
        spec = SOURCE_REVISIONS["os_atlas_desktop"]
        files = [
            "data/train-00001-of-00061.parquet",
            "data/train-00003-of-00061.parquet",
        ]
        hf_download_many(
            spec["repo"], spec["revision"], files, root / "raw" / "os_atlas_desktop", workers=args.workers
        )
    if "os_atlas_mobile" in requested:
        spec = SOURCE_REVISIONS["os_atlas"]
        files = [
            "mobile_domain/UIBert.zip",
            "mobile_domain/uibert_raw.json",
            "mobile_domain/aw_mobile.json",
            "mobile_domain/mobile_images.zip",
        ]
        hf_download_many(
            spec["repo"], spec["revision"], files, root / "raw" / "os_atlas_mobile", workers=args.workers
        )
    if "os_atlas_web" in requested:
        annotations = SOURCE_REVISIONS["os_atlas_web_annotations"]
        hf_download_many(
            annotations["repo"],
            annotations["revision"],
            ["data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet"],
            root / "raw" / "os_atlas_web" / "annotations",
            workers=min(args.workers, 2),
        )
        images = SOURCE_REVISIONS["os_atlas_web_images"]
        # Start with three image shards. The builder explicitly asks for more if
        # these do not expose at least 20K element annotations.
        image_files = [f"data/train-{index:05d}-of-00295.parquet" for index in range(3)]
        hf_download_many(
            images["repo"],
            images["revision"],
            image_files,
            root / "raw" / "os_atlas_web" / "images",
            workers=args.workers,
        )
    if "weblinx" in requested:
        download_weblinx(root, args.count, args.seed, args.workers)


def build_selected(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    raw = root / "raw"
    output = root / "parquet"
    output.mkdir(parents=True, exist_ok=True)
    requested = args.sources or [
        "mind2web",
        "weblinx",
        "os_atlas_web",
        "os_atlas_mobile",
        "rico_widget_caption",
        "os_atlas_desktop",
    ]
    builders = {
        "mind2web": lambda path: build_mind2web(
            raw / "mind2web",
            path,
            seed=args.seed,
            crop_size=args.mind2web_crop_size,
            shard_size=args.shard_size,
            prompt_protocol=args.mind2web_prompt_protocol,
            target_count=(
                args.mind2web_count
                if args.mind2web_count is not None
                else args.count
            ),
            random_crop=args.mind2web_crop_mode == "random",
        ),
        "weblinx": lambda path: build_weblinx(
            raw / "weblinx_meta",
            raw / "weblinx",
            path,
            count=args.count,
            seed=args.seed,
            shard_size=args.shard_size,
        ),
        "rico_widget_caption": lambda path: build_rico(
            raw / "rico",
            path,
            count=args.count,
            seed=args.seed,
            shard_size=args.shard_size,
        ),
        "os_atlas_web": lambda path: build_os_atlas_web(
            raw / "os_atlas_web" / "annotations",
            raw / "os_atlas_web" / "images",
            path,
            count=args.count,
            seed=args.seed,
            shard_size=args.shard_size,
        ),
        "os_atlas_mobile": lambda path: build_os_atlas_mobile(
            raw / "os_atlas_mobile",
            path,
            count=args.count,
            seed=args.seed,
            shard_size=args.shard_size,
        ),
        "os_atlas_desktop": lambda path: build_os_atlas_desktop(
            raw / "os_atlas_desktop",
            path,
            count=args.count,
            seed=args.seed,
            shard_size=args.shard_size,
        ),
    }
    manifest_path = output / "manifest.json"
    previous_sources: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            previous_sources = json.loads(manifest_path.read_text()).get("sources", {})
        except (OSError, json.JSONDecodeError):
            previous_sources = {}
    report: dict[str, Any] = {
        "paper": "https://arxiv.org/abs/2603.26211",
        "seed": args.seed,
        "target_rows_per_scalable_source": args.count,
        "mind2web_target_rows": (
            args.mind2web_count
            if args.mind2web_count is not None
            else args.count
        ),
        "mind2web_policy": (
            "scan all 7,775 published train rows; exclude rows without valid "
            "visible target boxes; repeat eligible targets with audited crop "
            "variants to reach the requested allocation"
        ),
        "mind2web_prompt_protocol": args.mind2web_prompt_protocol,
        "mind2web_crop_mode": args.mind2web_crop_mode,
        "exact_reproduction": False,
        "reason": "Paper does not publish sample IDs, crop seed/parameters, or OCR realignment code",
        "sources": previous_sources,
    }
    for source in requested:
        if source not in builders:
            raise ValueError(f"Unknown or not-yet-supported source: {source}")
        source_dir = output / source
        prepare_output_dir(source_dir, args.force)
        log(f"[build] {source} -> {source_dir}")
        report["sources"][source] = builders[source](source_dir)

    report["prepared_rows_by_source"] = {
        source: int(details["rows"])
        for source, details in sorted(report["sources"].items())
        if isinstance(details, dict) and "rows" in details
    }
    report["total_prepared_rows"] = sum(report["prepared_rows_by_source"].values())
    manifest_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    log(f"Wrote manifest: {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser(
        "download", help="Download the pinned raw subsets for the Table 1 mixture"
    )
    download.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    download.add_argument(
        "--sources",
        nargs="+",
        choices=[
            "mind2web",
            "weblinx",
            "os_atlas_web",
            "os_atlas_mobile",
            "rico_widget_caption",
            "os_atlas_desktop",
        ],
        help="Defaults to all six Table 1 source/domain buckets",
    )
    download.add_argument("--count", type=int, default=DEFAULT_COUNT)
    download.add_argument("--seed", type=int, default=DEFAULT_SEED)
    download.add_argument("--workers", type=int, default=4)

    build = subparsers.add_parser("build", help="Convert already downloaded raw data")
    build.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    build.add_argument(
        "--sources",
        nargs="+",
        choices=[
            "mind2web",
            "weblinx",
            "os_atlas_web",
            "os_atlas_mobile",
            "rico_widget_caption",
            "os_atlas_desktop",
        ],
        help="Defaults to all six Table 1 source/domain buckets",
    )
    build.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help="Rows for each paper source bucket (default: 20,000)",
    )
    build.add_argument("--seed", type=int, default=DEFAULT_SEED)
    build.add_argument(
        "--mind2web-count",
        type=int,
        help="Mind2Web output rows; defaults to --count (the paper allocation is 20,000)",
    )
    build.add_argument("--mind2web-crop-size", type=int, default=1280)
    build.add_argument(
        "--mind2web-crop-mode",
        choices=("random", "center"),
        default="random",
        help="Use seeded random crops for paper alignment or centered crops for an ablation",
    )
    build.add_argument(
        "--mind2web-prompt-protocol",
        choices=MIND2WEB_PROMPT_PROTOCOLS,
        default=TARGET_GROUNDING,
        help=(
            "target_grounding uses the step-specific target_action_reprs; "
            "task_history retains the legacy planning-plus-grounding prompt"
        ),
    )
    build.add_argument("--shard-size", type=int, default=1000)
    build.add_argument("--force", action="store_true")

    validate = subparsers.add_parser("validate", help="Validate generated LLaDA-o Parquet")
    validate.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    validate.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help="Fallback expected count when manifest.json is unavailable",
    )
    validate.add_argument("--deep", action="store_true", help="Decode every image")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "download":
        download_selected(args)
    elif args.command == "build":
        build_selected(args)
    elif args.command == "validate":
        parquet_root = args.root.resolve() / "parquet"
        expected_counts = expected_counts_from_manifest(parquet_root, args.count)
        report = validate_output(parquet_root, expected_counts, args.deep)
        report["deep_image_validation"] = args.deep
        payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        (parquet_root / "validation.json").write_text(payload)
        print(payload, end="")
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
