#!/usr/bin/env python3
"""Rewrite Mind2Web training labels with audited OCR-linked text boxes.

Only the Mind2Web bucket is rewritten.  Images, prompts, action types, and
typed values are preserved byte-for-byte; accepted OCR matches replace the
four response coordinates and the metadata target box.  Other Table-1 source
directories are hard-linked during ``finalize``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

try:
    from .ocr_target_realignment import (
        OCR_REALIGNMENT_VERSION,
        OCR_MATCH_CONFIG,
        OcrDetection,
        match_ocr_target,
        replace_action_bbox,
        scale_bbox,
        unscale_bbox,
    )
    from .realign_gui_grounding_ocr import (
        EASYOCR_CONFIG,
        EASYOCR_SPATIAL_PREFILTER_CONFIG,
        EASYOCR_VERSION,
        build_easyocr_reader,
        load_jsonl,
        read_spatially_relevant_text,
        sha256_file,
    )
except ImportError:  # Direct execution: python scripts/data/<this file>.py
    from ocr_target_realignment import (
        OCR_REALIGNMENT_VERSION,
        OCR_MATCH_CONFIG,
        OcrDetection,
        match_ocr_target,
        replace_action_bbox,
        scale_bbox,
        unscale_bbox,
    )
    from realign_gui_grounding_ocr import (
        EASYOCR_CONFIG,
        EASYOCR_SPATIAL_PREFILTER_CONFIG,
        EASYOCR_VERSION,
        build_easyocr_reader,
        load_jsonl,
        read_spatially_relevant_text,
        sha256_file,
    )


MIND2WEB_SOURCE = "mind2web"


def log(message: str) -> None:
    print(message, flush=True)


def image_bytes(row: dict[str, Any]) -> bytes:
    image = row.get("image") or {}
    value = image.get("bytes") if isinstance(image, dict) else None
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise ValueError("Parquet row has no embedded image bytes")
    return bytes(value)


def replace_assistant_answer(
    conversations: list[dict[str, Any]], bbox_1000: list[int]
) -> list[dict[str, Any]]:
    result = [dict(conversation) for conversation in conversations]
    for index in range(len(result) - 1, -1, -1):
        if result[index].get("from") == "gpt":
            result[index]["value"] = replace_action_bbox(
                str(result[index].get("value") or ""), bbox_1000
            )
            return result
    raise ValueError("training row has no assistant response")


class DetectionCache:
    """Small LRU cache for repeated crop images without retaining the corpus."""

    def __init__(self, maximum: int = 64):
        self.maximum = maximum
        self.values: OrderedDict[str, list[OcrDetection]] = OrderedDict()

    def get_or_run(
        self,
        reader: Any,
        data: bytes,
        *,
        source_bbox_xyxy: tuple[float, float, float, float],
        image_width: int,
        image_height: int,
    ) -> list[OcrDetection]:
        digest = hashlib.sha256(data).hexdigest()
        cache_key = f"{digest}:{','.join(f'{value:.3f}' for value in source_bbox_xyxy)}"
        cached = self.values.get(cache_key)
        if cached is not None:
            self.values.move_to_end(cache_key)
            return cached
        import numpy as np

        with Image.open(io.BytesIO(data)) as source:
            source.load()
            array = np.asarray(source.convert("RGB"))
        raw_detections = read_spatially_relevant_text(
            reader,
            array,
            source_bbox_xyxy=source_bbox_xyxy,
            image_width=image_width,
            image_height=image_height,
        )
        detections = [
            OcrDetection.from_easyocr(value)
            for value in raw_detections
        ]
        self.values[cache_key] = detections
        if len(self.values) > self.maximum:
            self.values.popitem(last=False)
        return detections


def realign_row(
    reader: Any, row: dict[str, Any], cache: DetectionCache
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = json.loads(str(row.get("metadata") or "{}"))
    dom_bbox = list(metadata["bbox_1000"])
    description = str(metadata.get("target_description") or "").strip()
    if not description:
        raise ValueError(f"{row.get('sample_id')} has no target_description")
    data = image_bytes(row)
    with Image.open(io.BytesIO(data)) as source:
        width, height = source.size
    dom_bbox_pixels = unscale_bbox(dom_bbox, width, height)
    detections = cache.get_or_run(
        reader,
        data,
        source_bbox_xyxy=dom_bbox_pixels,
        image_width=width,
        image_height=height,
    )
    match = match_ocr_target(
        target_text=description,
        source_bbox_xyxy=dom_bbox_pixels,
        detections=detections,
        image_width=width,
        image_height=height,
    )
    ocr_bbox = (
        scale_bbox(match.bbox_xyxy, width, height)
        if match.accepted and match.bbox_xyxy is not None
        else None
    )
    target_bbox = list(ocr_bbox) if ocr_bbox else dom_bbox

    result = dict(row)
    result["conversations"] = replace_assistant_answer(
        list(row.get("conversations") or []), target_bbox
    )
    metadata["bbox_dom_1000"] = dom_bbox
    metadata["bbox_1000"] = target_bbox
    metadata["ocr_realignment"] = {
        "version": OCR_REALIGNMENT_VERSION,
        "engine": f"easyocr=={EASYOCR_VERSION}",
        "accepted": bool(ocr_bbox),
        "fallback_to_dom": not bool(ocr_bbox),
        "prediction_independent": True,
        "target_bbox_dom_1000": dom_bbox,
        "target_bbox_ocr_1000": ocr_bbox,
        **{
            key: value
            for key, value in match.to_dict().items()
            if key not in {"bbox_xyxy", "accepted", "target_text"}
        },
    }
    result["metadata"] = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    audit = {
        "sample_id": row.get("sample_id"),
        "original_id": metadata.get("original_id"),
        "crop_variant": metadata.get("crop_variant"),
        "target_description": description,
        "target_role": metadata.get("target_role", ""),
        "target_bbox_dom_1000": dom_bbox,
        "target_bbox_ocr_1000": ocr_bbox,
        "match": match.to_dict(),
        "num_ocr_detections": len(detections),
        "error": None,
    }
    return result, audit


def fallback_row(row: dict[str, Any], exc: BaseException) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = json.loads(str(row.get("metadata") or "{}"))
    dom_bbox = list(metadata["bbox_1000"])
    result = dict(row)
    metadata["bbox_dom_1000"] = dom_bbox
    metadata["ocr_realignment"] = {
        "version": OCR_REALIGNMENT_VERSION,
        "engine": f"easyocr=={EASYOCR_VERSION}",
        "accepted": False,
        "fallback_to_dom": True,
        "prediction_independent": True,
        "reason": "ocr_processing_error",
        "error": f"{type(exc).__name__}: {exc}",
        "target_bbox_dom_1000": dom_bbox,
        "target_bbox_ocr_1000": None,
    }
    result["metadata"] = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    return result, {
        "sample_id": row.get("sample_id"),
        "original_id": metadata.get("original_id"),
        "crop_variant": metadata.get("crop_variant"),
        "target_description": metadata.get("target_description", ""),
        "target_role": metadata.get("target_role", ""),
        "target_bbox_dom_1000": dom_bbox,
        "target_bbox_ocr_1000": None,
        "match": {"accepted": False, "reason": "ocr_processing_error"},
        "num_ocr_detections": 0,
        "error": f"{type(exc).__name__}: {exc}",
    }


def rewrite_shard(
    reader: Any,
    source_path: Path,
    destination_path: Path,
    audit_path: Path,
    *,
    fail_fast: bool,
) -> tuple[int, Counter[str]]:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    destination_temporary = destination_path.with_suffix(".parquet.tmp")
    audit_temporary = audit_path.with_suffix(".jsonl.tmp")
    destination_temporary.unlink(missing_ok=True)
    audit_temporary.unlink(missing_ok=True)

    parquet = pq.ParquetFile(source_path)
    writer = pq.ParquetWriter(
        destination_temporary,
        parquet.schema_arrow,
        compression="zstd",
        compression_level=6,
        use_dictionary=["source"],
    )
    counters: Counter[str] = Counter()
    count = 0
    cache = DetectionCache()
    try:
        with audit_temporary.open(
            "w", encoding="utf-8", buffering=1
        ) as audit_handle:
            for row_group in range(parquet.num_row_groups):
                output_rows: list[dict[str, Any]] = []
                for row in parquet.read_row_group(row_group).to_pylist():
                    try:
                        output_row, audit = realign_row(reader, row, cache)
                    except Exception as exc:
                        if fail_fast:
                            raise
                        output_row, audit = fallback_row(row, exc)
                    output_rows.append(output_row)
                    audit_handle.write(
                        json.dumps(audit, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                    counters[
                        "ocr_matched"
                        if (audit.get("match") or {}).get("accepted")
                        else "dom_fallback"
                    ] += 1
                    if audit.get("error"):
                        counters["processing_error"] += 1
                    count += 1
                writer.write_table(pa.Table.from_pylist(output_rows, schema=parquet.schema_arrow))
    except Exception:
        writer.close()
        destination_temporary.unlink(missing_ok=True)
        audit_temporary.unlink(missing_ok=True)
        raise
    writer.close()
    os.replace(destination_temporary, destination_path)
    os.replace(audit_temporary, audit_path)
    return count, counters


def rewrite(args: argparse.Namespace) -> None:
    input_root = args.input_root.expanduser().resolve()
    work_dir = args.work_dir.expanduser().resolve()
    source_files = sorted((input_root / MIND2WEB_SOURCE).glob("*.parquet"))
    if not source_files:
        raise FileNotFoundError(f"no Mind2Web Parquet below {input_root}")
    assigned = [
        path for index, path in enumerate(source_files) if index % args.world_size == args.rank
    ]
    pending = [
        path
        for path in assigned
        if args.no_resume
        or not (work_dir / "parquet" / MIND2WEB_SOURCE / path.name).exists()
    ]
    log(
        f"rank {args.rank}/{args.world_size}: {len(pending)}/{len(assigned)} Mind2Web shards pending"
    )
    if not pending:
        return
    reader = build_easyocr_reader(args)
    for index, source_path in enumerate(pending, start=1):
        destination = work_dir / "parquet" / MIND2WEB_SOURCE / source_path.name
        audit = work_dir / "audit" / f"{source_path.stem}.jsonl"
        count, counters = rewrite_shard(
            reader, source_path, destination, audit, fail_fast=args.fail_fast
        )
        log(
            f"rank {args.rank}: {index}/{len(pending)} {source_path.name}: "
            f"{count:,} rows {dict(counters)}"
        )


def hardlink_tree(source: Path, destination: Path) -> None:
    for directory, directory_names, file_names in os.walk(source):
        relative = Path(directory).relative_to(source)
        target_directory = destination / relative
        target_directory.mkdir(parents=True, exist_ok=True)
        directory_names.sort()
        for file_name in sorted(file_names):
            source_file = Path(directory) / file_name
            target_file = target_directory / file_name
            os.link(source_file, target_file)


def prepare_output(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"output exists: {path}; pass --force")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def finalize(args: argparse.Namespace) -> None:
    input_root = args.input_root.expanduser().resolve()
    work_dir = args.work_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    input_files = sorted((input_root / MIND2WEB_SOURCE).glob("*.parquet"))
    rewritten_files = sorted(
        (work_dir / "parquet" / MIND2WEB_SOURCE).glob("*.parquet")
    )
    if [path.name for path in input_files] != [path.name for path in rewritten_files]:
        raise RuntimeError(
            f"rewritten Mind2Web shard mismatch: input={len(input_files)}, output={len(rewritten_files)}"
        )
    audit_paths = sorted((work_dir / "audit").glob("*.jsonl"))
    audits = [row for path in audit_paths for row in load_jsonl(path)]
    expected_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in input_files)
    rewritten_rows = sum(
        pq.ParquetFile(path).metadata.num_rows for path in rewritten_files
    )
    if len(audits) != expected_rows or rewritten_rows != expected_rows:
        raise RuntimeError(
            f"row coverage mismatch: expected={expected_rows}, rewritten={rewritten_rows}, audits={len(audits)}"
        )

    prepare_output(output_root, args.force)
    for source_directory in sorted(path for path in input_root.iterdir() if path.is_dir()):
        if source_directory.name == MIND2WEB_SOURCE:
            hardlink_tree(work_dir / "parquet" / MIND2WEB_SOURCE, output_root / MIND2WEB_SOURCE)
            rejection = source_directory / "rejections.json"
            if rejection.exists():
                os.link(rejection, output_root / MIND2WEB_SOURCE / "rejections.json")
        else:
            hardlink_tree(source_directory, output_root / source_directory.name)

    counters = Counter(
        "ocr_matched" if (row.get("match") or {}).get("accepted") else "dom_fallback"
        for row in audits
    )
    counters["processing_error"] = sum(bool(row.get("error")) for row in audits)
    manifest = json.loads((input_root / "manifest.json").read_text(encoding="utf-8"))
    manifest["exact_reproduction"] = False
    manifest["reason"] = (
        "Paper does not publish sample IDs, crop seed/parameters, OCR engine, "
        "or OCR-to-target matching thresholds"
    )
    manifest["mind2web_ocr_realignment"] = {
        "version": OCR_REALIGNMENT_VERSION,
        "engine": f"easyocr=={EASYOCR_VERSION}",
        "engine_config": EASYOCR_CONFIG,
        "spatial_prefilter": EASYOCR_SPATIAL_PREFILTER_CONFIG,
        "matcher_config": OCR_MATCH_CONFIG,
        "rows": expected_rows,
        "matched": counters["ocr_matched"],
        "dom_fallback": counters["dom_fallback"],
        "processing_errors": counters["processing_error"],
        "match_rate": counters["ocr_matched"] / expected_rows,
        "prediction_independent": True,
        "audit_shards": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in audit_paths
        ],
    }
    manifest["sources"][MIND2WEB_SOURCE]["annotation_protocol"] = (
        "ocr_linked_text_with_dom_fallback"
    )
    manifest["sources"][MIND2WEB_SOURCE]["ocr_realignment"] = manifest[
        "mind2web_ocr_realignment"
    ]
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    log(
        f"wrote {output_root}: {expected_rows:,} Mind2Web rows, "
        f"OCR matched {counters['ocr_matched']:,} ({100.0 * counters['ocr_matched'] / expected_rows:.2f}%), "
        f"errors {counters['processing_error']:,}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    rewrite_parser = subparsers.add_parser("rewrite", help="rewrite one rank's shards")
    rewrite_parser.add_argument("--input-root", type=Path, required=True)
    rewrite_parser.add_argument("--work-dir", type=Path, required=True)
    rewrite_parser.add_argument("--rank", type=int, default=int(os.environ.get("SLURM_PROCID", 0)))
    rewrite_parser.add_argument(
        "--world-size", type=int, default=int(os.environ.get("SLURM_NTASKS", 1))
    )
    rewrite_parser.add_argument("--languages", default="en")
    rewrite_parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(os.environ.get("EASYOCR_MODULE_PATH", "~/.EasyOCR/model")).expanduser(),
    )
    rewrite_parser.add_argument("--gpu", action=argparse.BooleanOptionalAction, default=True)
    rewrite_parser.add_argument("--no-download", action="store_true")
    rewrite_parser.add_argument("--no-resume", action="store_true")
    rewrite_parser.add_argument("--fail-fast", action="store_true")
    rewrite_parser.set_defaults(handler=rewrite)

    finalize_parser = subparsers.add_parser("finalize", help="assemble the 120K corpus")
    finalize_parser.add_argument("--input-root", type=Path, required=True)
    finalize_parser.add_argument("--work-dir", type=Path, required=True)
    finalize_parser.add_argument("--output-root", type=Path, required=True)
    finalize_parser.add_argument("--force", action="store_true")
    finalize_parser.set_defaults(handler=finalize)

    args = parser.parse_args()
    if args.command == "rewrite" and not (
        0 <= args.rank < args.world_size and args.world_size > 0
    ):
        parser.error("rank must satisfy 0 <= rank < world-size")
    return args


def main() -> None:
    args = parse_args()
    try:
        args.handler(args)
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
