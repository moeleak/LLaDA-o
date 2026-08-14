#!/usr/bin/env python3
"""Build paired held-out Grounder benchmarks with clean and incorrect hints.

Each output pair keeps the screenshot and gold box fixed. The clean arm uses
the authoritative Planner target, while the hard-hint arm substitutes a
different actionable control mined from the same held-out screenshot. Both
arms include task, app, visible-package, and redacted history context. This
isolates whether a Grounder can use intent to correct a bad Planner hint.

The script only reads the already-selected mobile validation/test benchmark
rows. It never writes training shards and never mixes held-out records into
training data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

if __package__:
    from .harden_residual_mobile_grounding import (
        GroundingHardeningError,
        build_context_prompt,
        compact_text,
        iter_jsonl,
        load_hierarchy,
        position_phrase,
        select_hard_negative,
        sha256_file,
        visible_packages,
    )
else:
    from harden_residual_mobile_grounding import (  # type: ignore[no-redef]
        GroundingHardeningError,
        build_context_prompt,
        compact_text,
        iter_jsonl,
        load_hierarchy,
        position_phrase,
        select_hard_negative,
        sha256_file,
        visible_packages,
    )


PROTOCOL = "task-app-history-target-hint-v1"


def sample_ids_sha256(rows: Iterable[dict[str, Any]]) -> str:
    sample_ids = [str(row["sample_id"]) for row in rows]
    return hashlib.sha256(("\n".join(sample_ids) + "\n").encode()).hexdigest()


def target_text(planner: dict[str, Any]) -> str:
    action = planner.get("planner_action") or {}
    label = (planner.get("ground_truth") or {}).get("target_label") or {}
    value = compact_text(action.get("target") or label.get("visible_text"), 500)
    if not value:
        raise GroundingHardeningError(
            f"Planner row has no spatial target text: {planner.get('id')}"
        )
    return value


def build_pair(
    *,
    sample: dict[str, Any],
    planner: dict[str, Any],
    hierarchy: dict[str, Any],
    clean_benchmark: str,
    hard_benchmark: str,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    elements = hierarchy.get("ui_elements") or []
    width, height = hierarchy.get("logical_screen_size") or (0, 0)
    gold_target = target_text(planner)
    gold_bbox = sample.get("target_bbox_1000")
    label = (planner.get("ground_truth") or {}).get("target_label") or {}
    distractor = select_hard_negative(
        elements=elements,
        target=gold_target,
        target_bbox=gold_bbox,
        target_role=compact_text(label.get("source_role"), 100),
        width=int(width),
        height=int(height),
    )
    if distractor is None:
        return None

    hard_hint = f"{distractor['label']} {position_phrase(distractor['bbox_1000'])}"
    context = {
        "task": str(planner.get("task") or ""),
        "task_app": str(planner.get("app") or ""),
        "task_package": str(planner.get("app_package") or ""),
        "packages": visible_packages(elements),
        "history": planner.get("history") or [],
    }
    clean_prompt = build_context_prompt(target_hint=gold_target, **context)
    hard_prompt = build_context_prompt(target_hint=hard_hint, **context)

    shared = dict(sample)
    shared.update(
        {
            "source_benchmark": sample.get("benchmark"),
            "source_prompt": sample.get("prompt"),
            "context_protocol": PROTOCOL,
            "original_target_hint": gold_target,
            "hard_negative_hint": hard_hint,
            "hard_negative_label": distractor["label"],
            "hard_negative_bbox_1000": distractor["bbox_1000"],
            "hard_negative_role": distractor["role"],
            "hard_negative_package": distractor["package"],
        }
    )
    clean = dict(shared)
    clean.update(
        benchmark=clean_benchmark,
        prompt=clean_prompt,
        native_prompt=clean_prompt,
        planner_target_hint=gold_target,
        hint_is_hard_negative=False,
    )
    hard = dict(shared)
    hard.update(
        benchmark=hard_benchmark,
        prompt=hard_prompt,
        native_prompt=hard_prompt,
        planner_target_hint=hard_hint,
        hint_is_hard_negative=True,
    )
    return clean, hard


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


def copy_image(input_root: Path, output_root: Path, relative: str) -> None:
    source = (input_root / relative).resolve()
    if not source.is_relative_to(input_root) or not source.is_file():
        raise GroundingHardeningError(f"unsafe or missing benchmark image: {source}")
    destination = output_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def build_split(
    *,
    split: str,
    input_root: Path,
    planner_root: Path,
    image_root: Path,
    output_root: Path,
    limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    source_benchmark = f"mobile_{split}"
    samples = list(iter_jsonl(input_root / "samples" / f"{source_benchmark}.jsonl"))
    planners = {
        str(row["id"]): row for row in iter_jsonl(planner_root / f"{split}.jsonl")
    }
    hierarchy_cache: dict[str, dict[str, Any] | None] = {}
    clean_rows: list[dict[str, Any]] = []
    hard_rows: list[dict[str, Any]] = []
    counts = {"source_rows": len(samples), "missing_hierarchy": 0, "no_distractor": 0}
    clean_benchmark = f"{source_benchmark}_context_clean"
    hard_benchmark = f"{source_benchmark}_context_hard_hint"

    for sample in samples:
        if len(clean_rows) >= limit:
            break
        source_id = str(sample.get("source_sample_id") or "")
        planner = planners.get(source_id)
        if planner is None:
            raise GroundingHardeningError(
                f"benchmark sample has no Planner source: {source_id}"
            )
        hierarchy = load_hierarchy(
            image_root,
            str(planner["trajectory_id"]),
            int(planner["step"]),
            hierarchy_cache,
        )
        if hierarchy is None:
            counts["missing_hierarchy"] += 1
            continue
        pair = build_pair(
            sample=sample,
            planner=planner,
            hierarchy=hierarchy,
            clean_benchmark=clean_benchmark,
            hard_benchmark=hard_benchmark,
        )
        if pair is None:
            counts["no_distractor"] += 1
            continue
        clean, hard = pair
        copy_image(input_root, output_root, str(sample["image"]))
        clean_rows.append(clean)
        hard_rows.append(hard)

    counts["paired_rows"] = len(clean_rows)
    if not clean_rows:
        raise GroundingHardeningError(f"no eligible paired rows for {split}")
    return clean_rows, hard_rows, counts


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    input_root = args.input_benchmark_root.expanduser().resolve()
    planner_root = args.planner_root.expanduser().resolve()
    image_root = args.image_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not 1 <= args.limit <= 100:
        raise ValueError("limit must be in [1, 100]")
    if output_root.exists() and any(output_root.iterdir()):
        if not args.force:
            raise FileExistsError(
                f"output already exists: {output_root}; pass --force to rebuild"
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    input_manifest = input_root / "manifest.json"
    benchmarks: dict[str, Any] = {}
    split_counts: dict[str, Any] = {}
    for split in ("validation", "test"):
        clean, hard, counts = build_split(
            split=split,
            input_root=input_root,
            planner_root=planner_root,
            image_root=image_root,
            output_root=output_root,
            limit=args.limit,
        )
        split_counts[split] = counts
        ids_digest = sample_ids_sha256(clean)
        clean_name = f"mobile_{split}_context_clean"
        hard_name = f"mobile_{split}_context_hard_hint"
        for name, rows, is_hard, paired_name in (
            (clean_name, clean, False, hard_name),
            (hard_name, hard, True, clean_name),
        ):
            relative = Path("samples") / f"{name}.jsonl"
            row_count, digest = write_jsonl(output_root / relative, rows)
            benchmarks[name] = {
                "path": relative.as_posix(),
                "rows": row_count,
                "sha256": digest,
                "sample_ids_sha256": ids_digest,
                "prompt_protocol": PROTOCOL,
                "hint_is_hard_negative": is_hard,
                "paired_benchmark": paired_name,
                "paper_comparison_eligible": False,
            }

    manifest = {
        "schema_version": 1,
        "format": "lladao-context-grounding-benchmark-v1",
        "base_benchmark_manifest": {
            "path": str(input_manifest),
            "sha256": sha256_file(input_manifest),
        },
        "policy": {
            "protocol": PROTOCOL,
            "selection": "eligible-pairs-from-existing-stable-100",
            "limit_per_split": args.limit,
            "held_out_only": True,
            "training_rows_written": 0,
        },
        "split_counts": split_counts,
        "benchmarks": benchmarks,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-benchmark-root", required=True, type=Path)
    parser.add_argument("--planner-root", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    manifest = prepare(parse_args())
    print(json.dumps(manifest["split_counts"], sort_keys=True))


if __name__ == "__main__":
    main()
