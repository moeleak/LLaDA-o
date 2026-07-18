#!/usr/bin/env python3
"""Build an OCR-aligned Mind2Web benchmark from prepared public samples.

The workflow has two explicit phases so OCR inference can be sharded across
GPUs and the final target construction remains deterministic:

1. ``detect`` runs OCR and records every candidate plus the annotation-side
   text match for its rank.
2. ``finalize`` verifies complete coverage and rewrites target boxes, falling
   back to the original DOM box when no credible nearby OCR text exists.

No model prediction is read by either phase.  Existing grounding predictions
can therefore be rescored against the resulting annotation protocol without
target leakage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

try:
    from .ocr_target_realignment import (
        OCR_REALIGNMENT_VERSION,
        OCR_MATCH_CONFIG,
        OcrDetection,
        match_ocr_target,
        scale_bbox,
        unscale_bbox,
    )
except ImportError:  # Direct execution: python scripts/data/<this file>.py
    from ocr_target_realignment import (
        OCR_REALIGNMENT_VERSION,
        OCR_MATCH_CONFIG,
        OcrDetection,
        match_ocr_target,
        scale_bbox,
        unscale_bbox,
    )


DEFAULT_BENCHMARK = "mind2web"
DEFAULT_LANGUAGES = "en"
EASYOCR_VERSION = "1.7.2"
EASYOCR_CONFIG = {
    "decoder": "greedy",
    "beamWidth": 1,
    "batch_size": 1,
    "workers": 0,
    "detail": 1,
    "paragraph": False,
    "canvas_size": 2560,
    "mag_ratio": 1.0,
    "text_threshold": 0.6,
    "low_text": 0.3,
    "link_threshold": 0.4,
}


def log(message: str) -> None:
    print(message, flush=True)


def load_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"malformed JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise RuntimeError(f"expected object at {path}:{line_number}")
            yield value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    os.replace(temporary, path)
    return count, sha256_file(path)


def manifest_and_samples(root: Path, benchmark: str) -> tuple[dict[str, Any], Path]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        relative = manifest["benchmarks"][benchmark]["path"]
    except KeyError as exc:
        raise KeyError(f"benchmark {benchmark!r} is absent from {manifest_path}") from exc
    return manifest, root / relative


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row["sample_id"]) for row in load_jsonl(path)}


def build_easyocr_reader(args: argparse.Namespace) -> Any:
    try:
        import easyocr
    except ImportError as exc:
        raise RuntimeError(
            f"EasyOCR {EASYOCR_VERSION} is required for detect; install easyocr=={EASYOCR_VERSION}"
        ) from exc

    languages = [value.strip() for value in args.languages.split(",") if value.strip()]
    if not languages:
        raise ValueError("at least one OCR language is required")
    args.model_dir.mkdir(parents=True, exist_ok=True)
    return easyocr.Reader(
        languages,
        gpu=args.gpu,
        model_storage_directory=str(args.model_dir),
        download_enabled=not args.no_download,
        verbose=args.rank == 0,
    )


def target_description(sample: dict[str, Any]) -> str:
    provenance = sample.get("provenance") or {}
    description = str(provenance.get("target_description") or "").strip()
    if not description:
        raise ValueError(f"{sample.get('sample_id')} has no target_description")
    return description


def detect_one(
    reader: Any,
    root: Path,
    sample: dict[str, Any],
    detection_cache: dict[str, list[OcrDetection]],
) -> dict[str, Any]:
    image_relative = str(sample["image"])
    image_path = root / image_relative
    if image_relative not in detection_cache:
        raw_detections = reader.readtext(str(image_path), **EASYOCR_CONFIG)
        detection_cache[image_relative] = [
            OcrDetection.from_easyocr(value) for value in raw_detections
        ]
    detections = detection_cache[image_relative]

    width = int(sample["image_width"])
    height = int(sample["image_height"])
    dom_bbox_1000 = list(sample["target_bbox_1000"])
    dom_bbox_pixels = unscale_bbox(dom_bbox_1000, width, height)
    description = target_description(sample)
    match = match_ocr_target(
        target_text=description,
        source_bbox_xyxy=dom_bbox_pixels,
        detections=detections,
        image_width=width,
        image_height=height,
    )
    match_dict = match.to_dict()
    ocr_bbox_1000 = (
        scale_bbox(match.bbox_xyxy, width, height)
        if match.accepted and match.bbox_xyxy is not None
        else None
    )
    provenance = sample.get("provenance") or {}
    return {
        "sample_id": sample["sample_id"],
        "action_uid": provenance.get("action_uid"),
        "split": sample.get("split"),
        "image": image_relative,
        "image_width": width,
        "image_height": height,
        "target_description": description,
        "target_role": provenance.get("target_role", ""),
        "target_bbox_dom_1000": dom_bbox_1000,
        "target_bbox_ocr_1000": ocr_bbox_1000,
        "match": match_dict,
        "detections": [detection.to_dict() for detection in detections],
        "error": None,
    }


def error_detection(sample: dict[str, Any], exc: BaseException) -> dict[str, Any]:
    provenance = sample.get("provenance") or {}
    return {
        "sample_id": sample.get("sample_id"),
        "action_uid": provenance.get("action_uid"),
        "split": sample.get("split"),
        "image": sample.get("image"),
        "image_width": sample.get("image_width"),
        "image_height": sample.get("image_height"),
        "target_description": provenance.get("target_description", ""),
        "target_role": provenance.get("target_role", ""),
        "target_bbox_dom_1000": sample.get("target_bbox_1000"),
        "target_bbox_ocr_1000": None,
        "match": {
            "accepted": False,
            "reason": "ocr_inference_error",
        },
        "detections": [],
        "error": f"{type(exc).__name__}: {exc}",
    }


def detect(args: argparse.Namespace) -> None:
    root = args.benchmark_root.expanduser().resolve()
    _, samples_path = manifest_and_samples(root, args.benchmark)
    output_dir = args.work_dir.expanduser().resolve() / "detections"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"part-{args.rank:05d}.jsonl"
    if args.no_resume and output_path.exists():
        output_path.unlink()
    completed = completed_ids(output_path)
    pending = [
        sample
        for index, sample in enumerate(load_jsonl(samples_path))
        if index % args.world_size == args.rank and str(sample["sample_id"]) not in completed
    ]
    log(
        f"rank {args.rank}/{args.world_size}: {len(pending):,} pending, "
        f"{len(completed):,} already complete"
    )
    if not pending:
        return

    reader = build_easyocr_reader(args)
    detection_cache: dict[str, list[OcrDetection]] = {}
    with output_path.open("a", encoding="utf-8") as handle:
        for pending_index, sample in enumerate(pending, start=1):
            try:
                result = detect_one(reader, root, sample, detection_cache)
            except Exception as exc:
                if args.fail_fast:
                    raise
                result = error_detection(sample, exc)
            handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            if pending_index % args.log_every == 0 or pending_index == len(pending):
                log(
                    f"rank {args.rank}: {pending_index:,}/{len(pending):,} new OCR records"
                )


def load_detection_records(work_dir: Path) -> dict[str, dict[str, Any]]:
    paths = sorted((work_dir / "detections").glob("part-*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no detection shards below {work_dir / 'detections'}")
    records: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in load_jsonl(path):
            sample_id = str(row["sample_id"])
            if sample_id in records:
                raise RuntimeError(f"duplicate OCR record for {sample_id}")
            records[sample_id] = row
    return records


def prepare_output_root(input_root: Path, output_root: Path, force: bool) -> None:
    if output_root.exists() or output_root.is_symlink():
        if not force:
            raise FileExistsError(f"output already exists: {output_root}; pass --force")
        if output_root.is_symlink() or output_root.is_file():
            output_root.unlink()
        else:
            shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    images_source = input_root / "images"
    images_link = output_root / "images"
    images_link.symlink_to(os.path.relpath(images_source, output_root), target_is_directory=True)


def realign_sample(sample: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    result = dict(sample)
    dom_bbox = list(sample["target_bbox_1000"])
    ocr_bbox = record.get("target_bbox_ocr_1000")
    accepted = bool((record.get("match") or {}).get("accepted") and ocr_bbox)
    result["target_bbox_dom_1000"] = dom_bbox
    result["target_bbox_1000"] = list(ocr_bbox) if accepted else dom_bbox

    provenance = dict(sample.get("provenance") or {})
    provenance["ocr_realignment"] = {
        "version": OCR_REALIGNMENT_VERSION,
        "engine": f"easyocr=={EASYOCR_VERSION}",
        "accepted": accepted,
        "fallback_to_dom": not accepted,
        "matched_text": (record.get("match") or {}).get("matched_text", ""),
        "text_similarity": (record.get("match") or {}).get("text_similarity", 0.0),
        "ocr_confidence": (record.get("match") or {}).get("ocr_confidence", 0.0),
        "edge_distance_normalized": (record.get("match") or {}).get(
            "edge_distance_normalized", 1.0
        ),
        "source_iou": (record.get("match") or {}).get("source_iou", 0.0),
        "reason": (record.get("match") or {}).get("reason", "missing_match"),
        "target_bbox_dom_1000": dom_bbox,
        "target_bbox_ocr_1000": list(ocr_bbox) if ocr_bbox else None,
        "prediction_independent": True,
    }
    result["provenance"] = provenance
    return result


def finalize(args: argparse.Namespace) -> None:
    input_root = args.benchmark_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    work_dir = args.work_dir.expanduser().resolve()
    manifest, primary_samples_path = manifest_and_samples(input_root, args.benchmark)
    detections = load_detection_records(work_dir)
    primary_samples = list(load_jsonl(primary_samples_path))
    expected_ids = {str(sample["sample_id"]) for sample in primary_samples}
    missing = sorted(expected_ids - set(detections))
    unexpected = sorted(set(detections) - expected_ids)
    if missing or unexpected:
        raise RuntimeError(
            f"OCR coverage mismatch: missing={len(missing):,}, unexpected={len(unexpected):,}"
        )

    by_action_uid: dict[str, dict[str, Any]] = {}
    for record in detections.values():
        action_uid = str(record.get("action_uid") or "")
        if not action_uid:
            raise RuntimeError(f"OCR record {record.get('sample_id')} has no action_uid")
        if action_uid in by_action_uid:
            raise RuntimeError(f"duplicate action_uid in OCR records: {action_uid}")
        by_action_uid[action_uid] = record

    prepare_output_root(input_root, output_root, args.force)
    output_manifest = json.loads(json.dumps(manifest))
    counters: Counter[str] = Counter()
    output_manifest["source_benchmark_manifest"] = str(
        (input_root / "manifest.json").resolve()
    )
    output_manifest["exact_paper_reproduction"] = False
    notes = list(output_manifest.get("protocol_notes") or [])
    notes.append(
        "Mind2Web target boxes use prediction-independent EasyOCR linked-text "
        "realignment with audited DOM fallback; the paper does not publish its OCR engine or matcher."
    )
    output_manifest["protocol_notes"] = list(dict.fromkeys(notes))

    for benchmark, info in output_manifest["benchmarks"].items():
        source_path = input_root / info["path"]
        destination_path = output_root / info["path"]
        if benchmark not in {args.benchmark, f"{args.benchmark}_task_history"}:
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination_path)
            info["sha256"] = sha256_file(destination_path)
            continue

        def rows() -> Iterator[dict[str, Any]]:
            for sample in load_jsonl(source_path):
                if benchmark == args.benchmark:
                    record = detections[str(sample["sample_id"])]
                else:
                    action_uid = str((sample.get("provenance") or {}).get("action_uid") or "")
                    record = by_action_uid[action_uid]
                result = realign_sample(sample, record)
                accepted = bool(
                    (result.get("provenance") or {})
                    .get("ocr_realignment", {})
                    .get("accepted")
                )
                counters[f"{benchmark}:ocr_matched" if accepted else f"{benchmark}:dom_fallback"] += 1
                if record.get("error"):
                    counters[f"{benchmark}:ocr_error"] += 1
                yield result

        row_count, digest = write_jsonl(destination_path, rows())
        if row_count != int(info["rows"]):
            raise RuntimeError(
                f"{benchmark} wrote {row_count:,} rows, expected {int(info['rows']):,}"
            )
        info["sha256"] = digest
        info["annotation_protocol"] = "ocr_linked_text_with_dom_fallback"
        info["ocr_realignment_version"] = OCR_REALIGNMENT_VERSION

    primary_matches = sum(
        bool((record.get("match") or {}).get("accepted")) for record in detections.values()
    )
    output_manifest["ocr_target_realignment"] = {
        "version": OCR_REALIGNMENT_VERSION,
        "engine": f"easyocr=={EASYOCR_VERSION}",
        "engine_config": EASYOCR_CONFIG,
        "matcher_config": OCR_MATCH_CONFIG,
        "benchmark": args.benchmark,
        "samples": len(primary_samples),
        "matched": primary_matches,
        "dom_fallback": len(primary_samples) - primary_matches,
        "match_rate": primary_matches / len(primary_samples),
        "prediction_independent": True,
        "detection_shards": [
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for path in sorted((work_dir / "detections").glob("part-*.jsonl"))
        ],
        "counters": dict(sorted(counters.items())),
        "paper_reproduction_boundary": (
            "The paper states OCR-guided targets but does not release its OCR engine, "
            "matching thresholds, sample IDs, or transformed annotations."
        ),
    }
    (output_root / "manifest.json").write_text(
        json.dumps(output_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    log(
        f"wrote {output_root}: OCR matched {primary_matches:,}/{len(primary_samples):,} "
        f"({100.0 * primary_matches / len(primary_samples):.2f}%)"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    detect_parser = subparsers.add_parser("detect", help="run one sharded OCR worker")
    detect_parser.add_argument("--benchmark-root", type=Path, required=True)
    detect_parser.add_argument("--work-dir", type=Path, required=True)
    detect_parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    detect_parser.add_argument("--rank", type=int, default=int(os.environ.get("SLURM_PROCID", 0)))
    detect_parser.add_argument(
        "--world-size", type=int, default=int(os.environ.get("SLURM_NTASKS", 1))
    )
    detect_parser.add_argument("--languages", default=DEFAULT_LANGUAGES)
    detect_parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(os.environ.get("EASYOCR_MODULE_PATH", "~/.EasyOCR/model")).expanduser(),
    )
    detect_parser.add_argument(
        "--gpu", action=argparse.BooleanOptionalAction, default=True
    )
    detect_parser.add_argument("--no-download", action="store_true")
    detect_parser.add_argument("--no-resume", action="store_true")
    detect_parser.add_argument("--fail-fast", action="store_true")
    detect_parser.add_argument("--log-every", type=int, default=25)
    detect_parser.set_defaults(handler=detect)

    finalize_parser = subparsers.add_parser(
        "finalize", help="verify OCR shards and construct the aligned benchmark"
    )
    finalize_parser.add_argument("--benchmark-root", type=Path, required=True)
    finalize_parser.add_argument("--work-dir", type=Path, required=True)
    finalize_parser.add_argument("--output-root", type=Path, required=True)
    finalize_parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    finalize_parser.add_argument("--force", action="store_true")
    finalize_parser.set_defaults(handler=finalize)

    args = parser.parse_args()
    if args.command == "detect":
        if args.rank < 0 or args.world_size <= 0 or args.rank >= args.world_size:
            parser.error("rank must satisfy 0 <= rank < world-size")
        if args.log_every <= 0:
            parser.error("--log-every must be positive")
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
