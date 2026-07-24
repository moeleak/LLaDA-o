#!/usr/bin/env python3
"""Fuse full-page OCR retrieval with native grounding on retrieval crops."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from eval.gui_grounding.ocr_fullpage_retrieval import (
    format_prediction,
    instruction_target,
    label_points_to_control,
    write_jsonl,
)
from eval.gui_grounding.score_benchmark import load_predictions, load_targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--ocr-predictions-dir", type=Path, required=True)
    parser.add_argument("--crop-benchmark-root", type=Path, required=True)
    parser.add_argument("--crop-predictions-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmark", default="mind2web_fullpage")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    if args.limit <= 0 or args.limit > 100:
        parser.error("--limit must be in [1, 100]")
    return args


def crop_bbox_to_source(
    bbox_1000: Sequence[float],
    *,
    crop_box: Sequence[int],
    source_width: int,
    source_height: int,
) -> list[int]:
    left, top, right, bottom = (int(value) for value in crop_box)
    crop_width = right - left
    crop_height = bottom - top
    x1, y1, x2, y2 = (float(value) for value in bbox_1000)
    source_bbox = (
        left + crop_width * x1 / 1000.0,
        top + crop_height * y1 / 1000.0,
        left + crop_width * x2 / 1000.0,
        top + crop_height * y2 / 1000.0,
    )
    return [
        round(1000.0 * source_bbox[0] / source_width),
        round(1000.0 * source_bbox[1] / source_height),
        round(1000.0 * source_bbox[2] / source_width),
        round(1000.0 * source_bbox[3] / source_height),
    ]


def prefer_crop_model(action: str, target_text: str) -> bool:
    return action == "lclick" and label_points_to_control(action, target_text)


def main() -> None:
    args = parse_args()
    benchmark_root = args.benchmark_root.expanduser().resolve()
    crop_root = args.crop_benchmark_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    manifest = json.loads((benchmark_root / "manifest.json").read_text())
    crop_manifest = json.loads((crop_root / "manifest.json").read_text())
    targets = load_targets(
        benchmark_root,
        manifest,
        args.benchmark,
        args.limit,
    )
    crop_targets = load_targets(
        crop_root,
        crop_manifest,
        args.benchmark,
        args.limit,
    )
    ocr_all = load_predictions(
        args.ocr_predictions_dir.expanduser().resolve(),
        args.benchmark,
    )
    crop_all = load_predictions(
        args.crop_predictions_dir.expanduser().resolve(),
        args.benchmark,
    )
    missing = set(targets) - set(ocr_all) | set(targets) - set(crop_all)
    if missing or set(targets) != set(crop_targets):
        raise RuntimeError(
            f"fusion coverage mismatch: missing={len(missing)}, "
            f"crop_targets={len(crop_targets)}, targets={len(targets)}"
        )

    rows: list[dict[str, Any]] = []
    crop_selected = 0
    for sample_id, target in targets.items():
        ocr_row = ocr_all[sample_id]
        crop_row = crop_all[sample_id]
        crop_target = crop_targets[sample_id]
        action, target_text, value = instruction_target(str(target["prompt"]))
        crop_metadata = crop_target["retrieval_crop"]
        local_bbox = crop_row.get("predicted_bbox_1000")
        crop_bbox = (
            crop_bbox_to_source(
                local_bbox,
                crop_box=crop_metadata["crop_box_xyxy"],
                source_width=int(crop_metadata["source_width"]),
                source_height=int(crop_metadata["source_height"]),
            )
            if isinstance(local_bbox, (list, tuple)) and len(local_bbox) == 4
            else None
        )
        ocr_bbox = ocr_row.get("predicted_bbox_1000")
        choose_crop = prefer_crop_model(action, target_text)
        if choose_crop and crop_bbox is not None:
            selected_bbox = crop_bbox
            selected_source = "crop_model"
            crop_selected += 1
        elif isinstance(ocr_bbox, (list, tuple)) and len(ocr_bbox) == 4:
            selected_bbox = [round(float(item)) for item in ocr_bbox]
            selected_source = "full_page_ocr"
        else:
            selected_bbox = crop_bbox
            selected_source = "crop_model_fallback"

        row = dict(ocr_row)
        row["predicted_action"] = action
        row["predicted_bbox_1000"] = selected_bbox
        row["predicted_value"] = value
        row["parse_error"] = None if selected_bbox is not None else "no_bbox"
        row["prediction"] = format_prediction(action, selected_bbox, value)
        row["ocr_crop_fusion"] = {
            "crop_bbox_1000": crop_bbox,
            "crop_model_prediction": crop_row.get("prediction", ""),
            "ocr_bbox_1000": ocr_bbox,
            "policy": (
                "crop_for_click_labels_otherwise_ocr"
            ),
            "selected_source": selected_source,
            "target_text": target_text,
            "uses_ground_truth_location": False,
        }
        rows.append(row)

    count = write_jsonl(
        output / args.benchmark / "part-00000.jsonl",
        rows,
    )
    (output / "fusion-config.json").write_text(
        json.dumps(
            {
                "benchmark_root": str(benchmark_root),
                "crop_benchmark_root": str(crop_root),
                "crop_predictions_dir": str(
                    args.crop_predictions_dir.expanduser().resolve()
                ),
                "crop_selected": crop_selected,
                "ocr_predictions_dir": str(
                    args.ocr_predictions_dir.expanduser().resolve()
                ),
                "policy": "crop_for_click_labels_otherwise_ocr",
                "samples": count,
                "uses_ground_truth_location": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(
        f"wrote {count} fused predictions to {output}; "
        f"crop_selected={crop_selected}",
        flush=True,
    )


if __name__ == "__main__":
    main()
