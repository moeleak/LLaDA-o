#!/usr/bin/env python3
"""Prepare prompt-only OCR retrieval crops for native GUI grounding."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from eval.gui_grounding.ocr_fullpage_retrieval import (
    native_instruction,
    write_jsonl,
)
from eval.gui_grounding.score_benchmark import load_predictions, load_targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--retrieval-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--benchmark", default="mind2web_fullpage")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--crop-size", type=int, default=980)
    parser.add_argument("--target-anchor-y", type=float, default=0.35)
    args = parser.parse_args()
    if args.limit <= 0 or args.limit > 100:
        parser.error("--limit must be in [1, 100]")
    if args.crop_size <= 0:
        parser.error("--crop-size must be positive")
    if not 0.0 < args.target_anchor_y < 1.0:
        parser.error("--target-anchor-y must be in (0, 1)")
    return args


def retrieval_crop_box(
    bbox_xyxy: Sequence[float],
    *,
    image_width: int,
    image_height: int,
    crop_size: int,
    target_anchor_y: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
    crop_width = min(crop_size, image_width)
    crop_height = min(crop_size, image_height)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    left = round(
        max(0.0, min(image_width - crop_width, center_x - crop_width / 2.0))
    )
    top = round(
        max(
            0.0,
            min(
                image_height - crop_height,
                center_y - crop_height * target_anchor_y,
            ),
        )
    )
    return left, top, left + crop_width, top + crop_height


def bbox_to_crop_coordinates(
    bbox_1000: Sequence[float],
    *,
    source_width: int,
    source_height: int,
    crop_box: Sequence[int],
) -> list[int]:
    left, top, right, bottom = crop_box
    crop_width = right - left
    crop_height = bottom - top
    x1, y1, x2, y2 = (float(value) for value in bbox_1000)
    source_bbox = (
        source_width * x1 / 1000.0,
        source_height * y1 / 1000.0,
        source_width * x2 / 1000.0,
        source_height * y2 / 1000.0,
    )
    return [
        round(1000.0 * (source_bbox[0] - left) / crop_width),
        round(1000.0 * (source_bbox[1] - top) / crop_height),
        round(1000.0 * (source_bbox[2] - left) / crop_width),
        round(1000.0 * (source_bbox[3] - top) / crop_height),
    ]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    benchmark_root = args.benchmark_root.expanduser().resolve()
    retrieval_dir = args.retrieval_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    manifest = json.loads((benchmark_root / "manifest.json").read_text())
    targets = load_targets(
        benchmark_root,
        manifest,
        args.benchmark,
        args.limit,
    )
    source_predictions = load_predictions(retrieval_dir, args.benchmark)
    retrieval = {
        sample_id: source_predictions[sample_id]
        for sample_id in targets
        if sample_id in source_predictions
    }
    missing = set(targets) - set(retrieval)
    if missing:
        raise RuntimeError(f"retrieval coverage mismatch: missing={len(missing)}")

    rows: list[dict[str, Any]] = []
    accepted = 0
    for index, (sample_id, sample) in enumerate(targets.items(), 1):
        retrieval_row = retrieval[sample_id]
        audit = retrieval_row.get("ocr_retrieval") or {}
        raw_bbox = audit.get("raw_bbox_xyxy")
        source_width = int(sample["image_width"])
        source_height = int(sample["image_height"])
        if audit.get("accepted") and raw_bbox is not None:
            crop_box = retrieval_crop_box(
                raw_bbox,
                image_width=source_width,
                image_height=source_height,
                crop_size=args.crop_size,
                target_anchor_y=args.target_anchor_y,
            )
            accepted += 1
        else:
            crop_box = (0, 0, source_width, source_height)

        source_path = benchmark_root / sample["image"]
        relative_image = (
            Path("images")
            / args.benchmark
            / f"{hashlib.sha256(sample_id.encode()).hexdigest()}.png"
        )
        output_image = output_root / relative_image
        output_image.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source_path) as source:
            crop = source.convert("RGB").crop(crop_box)
            crop.save(output_image, format="PNG", optimize=False)

        row = dict(sample)
        row.pop("tile_layout", None)
        row["image"] = str(relative_image)
        row["image_width"] = crop_box[2] - crop_box[0]
        row["image_height"] = crop_box[3] - crop_box[1]
        row["input_protocol"] = "prompt_ocr_retrieval_crop"
        row["prompt"] = native_instruction(str(sample["prompt"]))
        row["source_target_bbox_1000"] = list(sample["target_bbox_1000"])
        row["target_bbox_1000"] = bbox_to_crop_coordinates(
            sample["target_bbox_1000"],
            source_width=source_width,
            source_height=source_height,
            crop_box=crop_box,
        )
        row["retrieval_crop"] = {
            "accepted": bool(audit.get("accepted")),
            "crop_box_xyxy": list(crop_box),
            "matched_text": audit.get("matched_text", ""),
            "retrieval_bbox_xyxy": raw_bbox,
            "source_image": str(sample["image"]),
            "source_width": source_width,
            "source_height": source_height,
            "uses_ground_truth_location": False,
        }
        rows.append(row)
        if index == 1 or index % 10 == 0:
            print(
                f"retrieval crops {index}/{len(targets)} accepted={accepted}",
                flush=True,
            )

    samples_path = output_root / "samples" / f"{args.benchmark}.jsonl"
    count = write_jsonl(samples_path, rows)
    output_manifest = {
        "benchmarks": {
            args.benchmark: {
                "input_protocol": "prompt_ocr_retrieval_crop",
                "paper_comparison_eligible": False,
                "path": str(samples_path.relative_to(output_root)),
                "prompt_protocol": "target_grounding",
                "rows": count,
                "sha256": file_sha256(samples_path),
            }
        },
        "exact_paper_reproduction": False,
        "retrieval_crop": {
            "accepted": accepted,
            "crop_size": args.crop_size,
            "retrieval_dir": str(retrieval_dir),
            "source_benchmark_root": str(benchmark_root),
            "target_anchor_y": args.target_anchor_y,
            "uses_ground_truth_location": False,
        },
        "protocol_notes": [
            "Crop selection uses only the visible instruction and OCR output.",
            "Ground-truth boxes are transformed only for evaluation.",
        ],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(output_manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {count} retrieval crops to {output_root}", flush=True)


if __name__ == "__main__":
    main()
