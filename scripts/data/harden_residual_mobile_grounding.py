#!/usr/bin/env python3
"""Build context-aware mobile Grounder training data with hard target hints.

The residual Grounder historically saw only ``Click on <planner target>.``. A
wrong or ambiguous Planner target therefore forced the Grounder to faithfully
localize the wrong control. This post-processor keeps every clean training row
and gold box, but replaces its prompt with a task-aware protocol. For a
deterministic fraction of train rows, it appends a paired row whose target hint
is a different clickable element from the same screenshot's UI hierarchy. The
paired answer remains the intended action box, teaching the Grounder to use the
task, foreground package, and recent actions to reject a bad hint without
forgetting the original direct-target example.

Validation, test, and benchmark artifacts are hard-linked unchanged.  They
remain strict direct-target regression gates and are never used to mine hints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import pyarrow as pa
import pyarrow.parquet as pq


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

TOKEN_RE = re.compile(r"[a-z0-9]+")
GENERIC_LABELS = {
    "android view view",
    "android widget framelayout",
    "android widget linearlayout",
    "android widget scrollview",
    "root",
    "screen",
    "view",
    "workspace",
}
GENERIC_APP_TOKENS = {"android", "app", "application", "mobile", "pro"}


class GroundingHardeningError(ValueError):
    """Raised when an input violates the context-grounding contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_fraction(*parts: Any) -> float:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value / float(1 << 64)


def compact_text(value: Any, limit: int = 1_000) -> str:
    return " ".join(str(value or "").replace("\x00", " ").split())[:limit]


def normalize_ui_label(element: dict[str, Any]) -> str:
    """Return the best human-readable label for one hierarchy element."""

    for key in ("text", "content_description", "hint_text"):
        label = compact_text(element.get(key), 240)
        if label:
            return label
    resource = compact_text(
        element.get("resource_name") or element.get("resource_id"), 240
    )
    if resource:
        resource = resource.rsplit("/", 1)[-1].rsplit(":id/", 1)[-1]
        resource = re.sub(r"[_-]+", " ", resource).strip()
        if resource and resource.casefold() not in GENERIC_LABELS:
            return resource
    return ""


def normalize_bbox_1000(value: Any) -> tuple[int, int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise GroundingHardeningError("bbox_1000 must contain four values")
    bbox = tuple(int(round(float(item))) for item in value)
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
        raise GroundingHardeningError(f"invalid bbox_1000: {bbox}")
    return bbox


def hierarchy_bbox_1000(
    element: dict[str, Any], width: int, height: int
) -> tuple[int, int, int, int] | None:
    raw = element.get("bbox_pixels")
    if not isinstance(raw, dict) or width <= 0 or height <= 0:
        return None
    try:
        x1 = round(1000 * float(raw["x_min"]) / width)
        y1 = round(1000 * float(raw["y_min"]) / height)
        x2 = round(1000 * float(raw["x_max"]) / width)
        y2 = round(1000 * float(raw["y_max"]) / height)
    except (KeyError, TypeError, ValueError):
        return None
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(1000, x2), min(1000, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def bbox_center(bbox: Sequence[int]) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2


def point_in_box(point: Sequence[float], bbox: Sequence[int]) -> bool:
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]


def bbox_iou(left: Sequence[int], right: Sequence[int]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if not intersection:
        return 0.0
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / (left_area + right_area - intersection)


def label_tokens(value: str) -> set[str]:
    return set(TOKEN_RE.findall(value.casefold()))


def position_phrase(bbox: Sequence[int]) -> str:
    x, y = bbox_center(bbox)
    horizontal = "left" if x < 333 else "right" if x > 667 else "center"
    vertical = "top" if y < 333 else "bottom" if y > 667 else "middle"
    if horizontal == "center" and vertical == "middle":
        return "near the center"
    return f"in the {vertical} {horizontal}".replace(" center", "")


def visible_packages(elements: Iterable[dict[str, Any]]) -> list[str]:
    weighted: Counter[str] = Counter()
    for element in elements:
        package = compact_text(element.get("package_name"), 200)
        bbox = element.get("bbox_pixels") or {}
        if not package or element.get("is_visible") is False:
            continue
        try:
            area = max(1, int(bbox["x_max"]) - int(bbox["x_min"])) * max(
                1, int(bbox["y_max"]) - int(bbox["y_min"])
            )
        except (KeyError, TypeError, ValueError):
            area = 1
        weighted[package] += area
    return [package for package, _ in weighted.most_common(3)]


def candidate_role(element: dict[str, Any]) -> str:
    class_name = compact_text(element.get("class_name"), 200).casefold()
    if "button" in class_name or element.get("is_clickable"):
        return "button"
    if "image" in class_name:
        return "icon"
    if "text" in class_name:
        return "text"
    return class_name.rsplit(".", 1)[-1]


def select_hard_negative(
    *,
    elements: Iterable[dict[str, Any]],
    target: str,
    target_bbox: Sequence[int],
    target_role: str,
    width: int,
    height: int,
) -> dict[str, Any] | None:
    """Choose a labeled, actionable, non-target element from the same screen."""

    target_tokens = label_tokens(target)
    target_center = bbox_center(target_bbox)
    normalized_role = compact_text(target_role, 100).casefold()
    candidates: list[tuple[tuple[float, float, float, str], dict[str, Any]]] = []
    for element in elements:
        if element.get("is_visible") is False or element.get("is_enabled") is False:
            continue
        if not any(
            bool(element.get(key))
            for key in ("is_clickable", "is_long_clickable", "is_focusable")
        ):
            continue
        label = normalize_ui_label(element)
        bbox = hierarchy_bbox_1000(element, width, height)
        if not label or bbox is None:
            continue
        if (
            bbox_iou(bbox, target_bbox) >= 0.15
            or point_in_box(bbox_center(bbox), target_bbox)
            or point_in_box(target_center, bbox)
        ):
            continue
        tokens = label_tokens(label)
        union = target_tokens | tokens
        lexical = len(target_tokens & tokens) / len(union) if union else 0.0
        role = candidate_role(element)
        role_match = float(bool(normalized_role and normalized_role in role))
        cx, cy = bbox_center(bbox)
        distance = math.hypot(cx - target_center[0], cy - target_center[1])
        # Prefer same-name/same-role controls.  Distance is only a tie breaker:
        # far-away controls are still valuable hard negatives.
        score = (lexical, role_match, -distance, label.casefold())
        candidates.append(
            (
                score,
                {
                    "label": label,
                    "bbox_1000": list(bbox),
                    "role": role,
                    "package": compact_text(element.get("package_name"), 200),
                },
            )
        )
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def summarize_action(action: Any) -> str:
    if not isinstance(action, dict):
        return "unknown action"
    kind = compact_text(action.get("action"), 60).casefold()
    if kind in {"click", "long_press", "open"}:
        target = compact_text(action.get("target") or action.get("app"), 240)
        return f"{kind} {target}".strip()
    if kind == "type":
        return "type <redacted text>"
    if kind == "swipe":
        return "swipe"
    return kind or "unknown action"


def build_context_prompt(
    *,
    task: str,
    task_app: str,
    task_package: str,
    packages: Sequence[str],
    history: Sequence[Any],
    target_hint: str,
) -> str:
    recent = [summarize_action(action) for action in history[-4:]]
    lines = [
        "Ground the intended next Android action on the screenshot.",
        f"Task: {compact_text(task, 1_500)}",
    ]
    app = compact_text(task_app, 200)
    package = compact_text(task_package, 200)
    app_tokens = label_tokens(app) - GENERIC_APP_TOKENS
    task_tokens = label_tokens(task)
    package_is_visible = bool(package and package in packages)
    app_is_mentioned = bool(app_tokens & task_tokens)
    if (app or package) and (package_is_visible or app_is_mentioned):
        lines.append(f"Task app: {app or 'unknown'} ({package or 'unknown package'})")
    if packages:
        lines.append("Visible package(s): " + ", ".join(packages[:3]))
    lines.append("Recent actions:")
    lines.extend(f"- {action}" for action in recent or ["none"])
    lines.extend(
        [
            f"Planner target hint (may be imprecise): {compact_text(target_hint, 500)}",
            (
                "Use the task, visible UI, and recent actions to resolve ambiguous or "
                "incorrect hints. Return only lclick [x1,y1,x2,y2] with coordinates "
                "normalized to 0..1000."
            ),
        ]
    )
    return "\n".join(lines)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise GroundingHardeningError(
                    f"invalid JSON at {path}:{line_number}"
                ) from error
            if not isinstance(value, dict):
                raise GroundingHardeningError(
                    f"non-object JSON at {path}:{line_number}"
                )
            yield value


def hardlink_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def load_hierarchy(
    image_root: Path,
    trajectory_id: str,
    step: int,
    cache: dict[str, dict[str, Any] | None],
) -> dict[str, Any] | None:
    if trajectory_id not in cache:
        path = (image_root / trajectory_id / "metadata.json").resolve()
        if not path.is_relative_to(image_root):
            raise GroundingHardeningError(f"unsafe UI hierarchy path: {path}")
        cache[trajectory_id] = (
            json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
        )
    metadata = cache[trajectory_id]
    if metadata is None:
        return None
    steps = metadata.get("steps") or []
    for value in steps:
        if int(value.get("step", -1)) == step:
            return value
    return None


def write_train_shards(
    *,
    input_root: Path,
    planner_rows: dict[str, dict[str, Any]],
    image_root: Path,
    output_root: Path,
    seed: int,
    hard_negative_rate: float,
    shard_size: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    output_root.mkdir(parents=True, exist_ok=True)
    buffer: list[dict[str, Any]] = []
    shards: list[dict[str, Any]] = []
    hierarchy_cache: dict[str, dict[str, Any] | None] = {}
    counts = Counter()

    def flush() -> None:
        if not buffer:
            return
        path = output_root / f"shard-{len(shards):05d}.parquet"
        temporary = path.with_suffix(".parquet.tmp")
        pq.write_table(
            pa.Table.from_pylist(buffer, schema=OUTPUT_SCHEMA),
            temporary,
            compression="zstd",
            compression_level=6,
            row_group_size=min(256, len(buffer)),
            use_dictionary=["source"],
        )
        os.replace(temporary, path)
        shards.append(
            {
                "path": f"train/{path.name}",
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
        buffer.clear()

    for path in sorted(input_root.glob("*.parquet")):
        parquet = pq.ParquetFile(path)
        for group in range(parquet.num_row_groups):
            rows = parquet.read_row_group(group).to_pylist()
            for record in rows:
                metadata = json.loads(record["metadata"])
                source_id = str(metadata["source_sample_id"])
                planner = planner_rows.get(source_id)
                if planner is None:
                    raise GroundingHardeningError(
                        f"mobile training row has no Planner source: {source_id}"
                    )
                trajectory_id = str(planner["trajectory_id"])
                step = int(planner["step"])
                hierarchy = load_hierarchy(
                    image_root, trajectory_id, step, hierarchy_cache
                )
                if hierarchy is None:
                    counts["missing_ui_hierarchy"] += 1
                    hierarchy = {}
                elements = hierarchy.get("ui_elements") or []
                width, height = hierarchy.get("logical_screen_size") or (0, 0)
                target = compact_text(metadata["target"], 500)
                target_bbox = normalize_bbox_1000(metadata["bbox_1000"])
                target_role = compact_text(
                    ((planner.get("ground_truth") or {}).get("target_label") or {}).get(
                        "source_role"
                    ),
                    100,
                )
                distractor = select_hard_negative(
                    elements=elements,
                    target=target,
                    target_bbox=target_bbox,
                    target_role=target_role,
                    width=int(width),
                    height=int(height),
                )
                append_hard_negative = bool(
                    distractor
                    and stable_fraction(seed, source_id, "hard-negative")
                    < hard_negative_rate
                )
                if distractor is None:
                    counts["no_eligible_distractor"] += 1

                packages = visible_packages(elements)

                def append_variant(target_hint: str, *, hard_negative: bool) -> None:
                    output_record = dict(record)
                    if hard_negative:
                        output_record["sample_id"] = f"{record['sample_id']}:hard-hint"
                    prompt = build_context_prompt(
                        task=str(planner.get("task") or ""),
                        task_app=str(planner.get("app") or ""),
                        task_package=str(planner.get("app_package") or ""),
                        packages=packages,
                        history=planner.get("history") or [],
                        target_hint=target_hint,
                    )
                    output_metadata = dict(metadata)
                    output_metadata.update(
                        {
                            "annotation": "planner-context-hard-negative-v2",
                            "context_protocol": "task-app-history-target-hint-v1",
                            "hard_negative": hard_negative,
                            "paired_augmentation": True,
                            "original_target_hint": target,
                            "training_target_hint": target_hint,
                            "visible_packages": packages,
                        }
                    )
                    if distractor is not None:
                        output_metadata["distractor"] = distractor
                    output_record["conversations"] = [
                        {"from": "human", "value": f"<image>\n{prompt}"},
                        record["conversations"][-1],
                    ]
                    output_record["metadata"] = json.dumps(
                        output_metadata, ensure_ascii=False, sort_keys=True
                    )
                    buffer.append(output_record)
                    counts["rows"] += 1
                    if len(buffer) >= shard_size:
                        flush()

                append_variant(target, hard_negative=False)
                counts["clean_context"] += 1
                counts["source_rows"] += 1
                if append_hard_negative:
                    hard_hint = (
                        f"{distractor['label']} "
                        f"{position_phrase(distractor['bbox_1000'])}"
                    )
                    append_variant(hard_hint, hard_negative=True)
                    counts["hard_negative"] += 1
    flush()
    if counts["rows"] == 0:
        raise GroundingHardeningError("input mobile train split is empty")
    return shards, dict(sorted(counts.items()))


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    input_root = args.input_root.expanduser().resolve()
    planner_root = args.planner_root.expanduser().resolve()
    image_root = args.image_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not 0 <= args.hard_negative_rate <= 1:
        raise ValueError("hard-negative-rate must be in [0, 1]")
    if args.shard_size <= 0:
        raise ValueError("shard-size must be positive")
    if output_root.exists() and any(output_root.iterdir()):
        if not args.force:
            raise FileExistsError(
                f"output already exists: {output_root}; pass --force to rebuild"
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    input_manifest_path = input_root / "manifest.json"
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    planner_rows = {
        str(row["id"]): row
        for row in iter_jsonl(planner_root / "train.jsonl")
        if str(((row.get("planner_action") or {}).get("action") or "")).casefold()
        in {"click", "long_press"}
    }
    shards, counts = write_train_shards(
        input_root=input_root / "train",
        planner_rows=planner_rows,
        image_root=image_root,
        output_root=output_root / "train",
        seed=args.seed,
        hard_negative_rate=args.hard_negative_rate,
        shard_size=args.shard_size,
    )

    for name in ("validation", "test", "benchmark"):
        source = input_root / name
        if not source.exists():
            raise FileNotFoundError(f"required input artifact is missing: {source}")
        shutil.copytree(source, output_root / name, copy_function=hardlink_or_copy)

    manifest = dict(input_manifest)
    manifest.update(
        {
            "schema_version": 2,
            "format": "lladao-residual-mobile-grounding-context-v3",
            "base_manifest": {
                "path": str(input_manifest_path),
                "sha256": sha256_file(input_manifest_path),
            },
            "context_hardening": {
                "protocol": "task-app-history-target-hint-v1",
                "seed": args.seed,
                "hard_negative_rate": args.hard_negative_rate,
                "selection": "stable-sha256",
                "augmentation": "paired-clean-and-hard-hint",
                "source_splits_used": ["train"],
                "held_out_prompts_unchanged": True,
                "counts": counts,
            },
        }
    )
    splits = dict(manifest.get("splits") or {})
    train = dict(splits.get("train") or {})
    train.update(rows=counts["rows"], shards=shards)
    splits["train"] = train
    manifest["splits"] = splits
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--planner-root", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hard-negative-rate", type=float, default=0.25)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = prepare(args)
    counts = manifest["context_hardening"]["counts"]
    print(json.dumps({"output": str(args.output_root), **counts}, sort_keys=True))


if __name__ == "__main__":
    main()
