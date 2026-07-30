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
    parser.add_argument(
        "--policy",
        choices=("selective", "crop", "ocr"),
        default="selective",
    )
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


def valid_prediction_bbox(row: dict[str, Any]) -> list[float] | None:
    raw = row.get("predicted_bbox_1000")
    if row.get("parse_error") not in (None, ""):
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    bbox = [float(value) for value in raw]
    x1, y1, x2, y2 = bbox
    if (
        not all(0.0 <= value <= 1000.0 for value in bbox)
        or x2 <= x1
        or y2 <= y1
    ):
        return None
    return bbox


def prefer_crop_model(action: str, target_text: str) -> bool:
    return action == "lclick" and label_points_to_control(action, target_text)


def use_crop_prediction(policy: str, action: str, target_text: str) -> bool:
    if policy == "crop":
        return True
    if policy == "ocr":
        return False
    if policy == "selective":
        return prefer_crop_model(action, target_text)
    raise ValueError(f"unsupported fusion policy: {policy}")


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
        local_bbox = valid_prediction_bbox(crop_row)
        crop_bbox = (
            crop_bbox_to_source(
                local_bbox,
                crop_box=crop_metadata["crop_box_xyxy"],
                source_width=int(crop_metadata["source_width"]),
                source_height=int(crop_metadata["source_height"]),
            )
            if local_bbox is not None
            else None
        )
        ocr_bbox = valid_prediction_bbox(ocr_row)
        choose_crop = use_crop_prediction(
            args.policy,
            action,
            target_text,
        )
        if choose_crop and crop_bbox is not None:
            selected_bbox = crop_bbox
            selected_source = "crop_model"
            selected_action = str(crop_row.get("predicted_action") or action)
            selected_value = str(crop_row.get("predicted_value") or "")
            crop_selected += 1
        elif ocr_bbox is not None:
            selected_bbox = [round(float(item)) for item in ocr_bbox]
            selected_source = "full_page_ocr"
            selected_action = action
            selected_value = value
        else:
            selected_bbox = crop_bbox
            selected_source = "crop_model_fallback"
            selected_action = str(crop_row.get("predicted_action") or action)
            selected_value = str(crop_row.get("predicted_value") or "")

        row = dict(ocr_row)
        for key in ("target_action", "target_bbox_1000", "target_value"):
            row.pop(key, None)
        if selected_source.startswith("crop_model"):
            for key in (
                "convergence_steps",
                "error",
                "generated_tokens",
                "generation_seconds",
                "latency_seconds",
                "peak_gpu_memory_mib",
                "runner_returncode",
                "runtime_input_protocol",
            ):
                if key in crop_row:
                    row[key] = crop_row[key]
            row["backend"] = (
                f"{crop_row.get('backend', 'model')}+ocr-retrieval-crop"
            )
            row["latency_scope"] = "crop_model_only_excludes_ocr"
        else:
            row["latency_scope"] = "source_prediction_only_excludes_ocr"
        row["predicted_action"] = selected_action
        row["predicted_bbox_1000"] = selected_bbox
        row["predicted_value"] = selected_value
        row["parse_error"] = None if selected_bbox is not None else "no_bbox"
        row["prediction"] = format_prediction(
            selected_action,
            selected_bbox,
            selected_value,
        )
        row["ocr_crop_fusion"] = {
            "crop_bbox_1000": crop_bbox,
            "crop_model_prediction": crop_row.get("prediction", ""),
            "ocr_bbox_1000": ocr_bbox,
            "policy": args.policy,
            "selected_source": selected_source,
            "target_text": target_text,
            "crop_model_latency_seconds": crop_row.get("latency_seconds"),
            "source_prediction_latency_seconds": ocr_row.get(
                "latency_seconds"
            ),
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
                "policy": args.policy,
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
        f"policy={args.policy} crop_selected={crop_selected}",
        flush=True,
    )


if __name__ == "__main__":
    main()
