#!/usr/bin/env python3
"""Analyze a paired Mind2Web target-grounding/planning diagnostic.

The direct arm names the next target and measures the model's grounding upper
bound.  The planner arm provides only the high-level task and previous actions.
Both arms must contain the same action UIDs, screenshots, and targets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from eval.gui_grounding.metrics import ACTION_TYPES, bbox_center, parse_action, point_in_box


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--direct-predictions-dir", type=Path, required=True)
    parser.add_argument("--planner-predictions-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--direct-benchmark", default="mind2web")
    parser.add_argument("--planner-benchmark", default="mind2web_task_history")
    parser.add_argument("--examples", type=int, default=10)
    args = parser.parse_args()
    if args.examples < 0:
        parser.error("--examples must be non-negative")
    return args


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"malformed JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def load_targets(
    root: Path, manifest: Mapping[str, Any], benchmark: str
) -> dict[str, dict[str, Any]]:
    details = manifest.get("benchmarks", {}).get(benchmark)
    if not isinstance(details, Mapping):
        raise KeyError(f"benchmark is not prepared: {benchmark}")
    path = root / str(details["path"])
    targets: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        provenance = row.get("provenance")
        action_uid = (
            str(provenance.get("action_uid"))
            if isinstance(provenance, Mapping) and provenance.get("action_uid")
            else ""
        )
        if not action_uid:
            raise RuntimeError(f"target lacks provenance.action_uid: {row.get('sample_id')}")
        if action_uid in targets:
            raise RuntimeError(f"duplicate target action UID in {benchmark}: {action_uid}")
        targets[action_uid] = row
    return targets


def load_predictions(directory: Path, benchmark: str) -> dict[str, dict[str, Any]]:
    paths = sorted((directory / benchmark).glob("part-*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no prediction shards for {benchmark} below {directory}")
    predictions: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in load_jsonl(path):
            sample_id = str(row.get("sample_id", ""))
            if not sample_id:
                raise RuntimeError(f"prediction lacks sample_id in {path}")
            if sample_id in predictions:
                raise RuntimeError(f"duplicate prediction: {sample_id}")
            predictions[sample_id] = row
    return predictions


def normalized_value(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(text.split()).casefold()


def parsed_prediction(row: Mapping[str, Any]) -> tuple[str | None, tuple[float, ...] | None, str, bool]:
    if "predicted_action" not in row and "predicted_bbox_1000" not in row:
        parsed = parse_action(row.get("prediction"))
        return parsed.action, parsed.bbox_1000, parsed.value, parsed.valid

    action = row.get("predicted_action")
    action = str(action).lower() if action is not None else None
    raw_bbox = row.get("predicted_bbox_1000")
    bbox = (
        tuple(float(value) for value in raw_bbox)
        if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4
        else None
    )
    valid = bool(
        action in ACTION_TYPES
        and bbox is not None
        and all(math.isfinite(value) and 0.0 <= value <= 1000.0 for value in bbox)
        and bbox[2] > bbox[0]
        and bbox[3] > bbox[1]
    )
    return action, bbox, str(row.get("predicted_value") or ""), valid


def validate_pair(
    direct: Mapping[str, Any], planner: Mapping[str, Any], action_uid: str
) -> None:
    fields = ("image", "split", "target_action", "target_bbox_1000", "target_value")
    differences = [field for field in fields if direct.get(field) != planner.get(field)]
    if differences:
        raise RuntimeError(
            f"paired target mismatch for {action_uid}: {', '.join(differences)}"
        )


def evaluate_one(
    target: Mapping[str, Any], prediction: Mapping[str, Any]
) -> dict[str, Any]:
    action, bbox, value, valid = parsed_prediction(prediction)
    target_action = str(target["target_action"]).lower()
    # Keep action accuracy comparable to the repository scorer: a recognizable
    # action label can be correct even when its accompanying box is invalid.
    action_hit = action == target_action
    point_hit = bool(
        valid
        and bbox is not None
        and point_in_box(bbox_center(bbox), target["target_bbox_1000"])
    )
    value_hit = (
        normalized_value(value) == normalized_value(target.get("target_value"))
        if target_action == "type_in"
        else True
    )
    return {
        "sample_id": str(target["sample_id"]),
        "split": str(target.get("split") or "test"),
        "target_action": target_action,
        "target_value": str(target.get("target_value") or ""),
        "prompt": str(target.get("prompt") or ""),
        "prediction": str(prediction.get("prediction") or ""),
        "parsed": valid,
        "action_hit": bool(action_hit),
        "point_hit": point_hit,
        "value_hit": value_hit,
        "strict_hit": bool(action_hit and point_hit and value_hit),
        "latency_seconds": prediction.get("latency_seconds"),
    }


def wilson_interval(successes: int, count: int, z: float = 1.959963984540054) -> list[float] | None:
    if count == 0:
        return None
    rate = successes / count
    denominator = 1.0 + z * z / count
    center = (rate + z * z / (2.0 * count)) / denominator
    margin = z * math.sqrt(rate * (1.0 - rate) / count + z * z / (4.0 * count * count)) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def mean_or_none(values: Iterable[Any]) -> float | None:
    finite = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    return statistics.fmean(finite) if finite else None


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(rows)

    def rate(field: str) -> float:
        return sum(bool(row[field]) for row in rows) / count if count else 0.0

    strict_successes = sum(bool(row["strict_hit"]) for row in rows)
    type_rows = [row for row in rows if row["target_action"] == "type_in"]
    return {
        "samples": count,
        "parse_rate": rate("parsed"),
        "action_accuracy": rate("action_hit"),
        "point_success": rate("point_hit"),
        "strict_next_action_success": rate("strict_hit"),
        "strict_next_action_success_95ci": wilson_interval(strict_successes, count),
        "type_value_accuracy": (
            sum(bool(row["value_hit"]) for row in type_rows) / len(type_rows)
            if type_rows
            else None
        ),
        "type_samples": len(type_rows),
        "mean_latency_seconds": mean_or_none(row.get("latency_seconds") for row in rows),
    }


def subgroup_summaries(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {name: summarize(group) for name, group in sorted(groups.items())}


def mcnemar_exact_two_sided(direct_only: int, planner_only: int) -> float:
    discordant = direct_only + planner_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(min(direct_only, planner_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_hashes(paths: Iterable[Path]) -> dict[str, str]:
    return {str(path.resolve()): sha256_file(path) for path in sorted(paths)}


def build_report(
    *,
    benchmark_root: Path,
    direct_predictions_dir: Path,
    planner_predictions_dir: Path,
    direct_benchmark: str = "mind2web",
    planner_benchmark: str = "mind2web_task_history",
    example_limit: int = 10,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = benchmark_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    direct_targets = load_targets(benchmark_root, manifest, direct_benchmark)
    planner_targets = load_targets(benchmark_root, manifest, planner_benchmark)
    if set(direct_targets) != set(planner_targets):
        raise RuntimeError("direct and planner target action UID sets differ")

    direct_predictions = load_predictions(direct_predictions_dir, direct_benchmark)
    planner_predictions = load_predictions(planner_predictions_dir, planner_benchmark)
    expected_direct_ids = {str(row["sample_id"]) for row in direct_targets.values()}
    expected_planner_ids = {str(row["sample_id"]) for row in planner_targets.values()}
    if set(direct_predictions) != expected_direct_ids:
        raise RuntimeError("direct prediction coverage does not exactly match targets")
    if set(planner_predictions) != expected_planner_ids:
        raise RuntimeError("planner prediction coverage does not exactly match targets")

    paired_rows: list[dict[str, Any]] = []
    direct_rows: list[dict[str, Any]] = []
    planner_rows: list[dict[str, Any]] = []
    for action_uid, direct_target in direct_targets.items():
        planner_target = planner_targets[action_uid]
        validate_pair(direct_target, planner_target, action_uid)
        direct = evaluate_one(
            direct_target, direct_predictions[str(direct_target["sample_id"])]
        )
        planner = evaluate_one(
            planner_target, planner_predictions[str(planner_target["sample_id"])]
        )
        direct_rows.append(direct)
        planner_rows.append(planner)
        paired_rows.append({"action_uid": action_uid, "direct": direct, "planner": planner})

    direct_only = sum(
        row["direct"]["strict_hit"] and not row["planner"]["strict_hit"]
        for row in paired_rows
    )
    planner_only = sum(
        row["planner"]["strict_hit"] and not row["direct"]["strict_hit"]
        for row in paired_rows
    )
    both = sum(
        row["direct"]["strict_hit"] and row["planner"]["strict_hit"]
        for row in paired_rows
    )
    neither = len(paired_rows) - direct_only - planner_only - both
    direct_summary = summarize(direct_rows)
    planner_summary = summarize(planner_rows)
    direct_success = direct_summary["strict_next_action_success"]
    planner_success = planner_summary["strict_next_action_success"]

    input_paths = list((direct_predictions_dir / direct_benchmark).glob("part-*.jsonl"))
    input_paths += list((planner_predictions_dir / planner_benchmark).glob("part-*.jsonl"))
    input_paths += list(direct_predictions_dir.glob("run-config*.json"))
    input_paths += list(planner_predictions_dir.glob("run-config*.json"))
    report = {
        "protocol": {
            "direct_benchmark": direct_benchmark,
            "planner_benchmark": planner_benchmark,
            "pairing_key": "provenance.action_uid",
            "strict_success_definition": (
                "valid parse, correct action type, predicted-box center in target box, "
                "and normalized value equality for type_in"
            ),
            "scope": "single-step state-conditioned GUI planning, not open-loop multi-step planning",
        },
        "inputs": {
            "benchmark_manifest": str(manifest_path.resolve()),
            "paired_action_uids": list(direct_targets),
            "sha256": input_hashes([manifest_path, *input_paths]),
        },
        "arms": {
            "direct_target_grounding": {
                "overall": direct_summary,
                "by_split": subgroup_summaries(direct_rows, "split"),
                "by_action": subgroup_summaries(direct_rows, "target_action"),
            },
            "task_history_planner": {
                "overall": planner_summary,
                "by_split": subgroup_summaries(planner_rows, "split"),
                "by_action": subgroup_summaries(planner_rows, "target_action"),
            },
        },
        "paired_comparison": {
            "both_success": both,
            "direct_only_success": direct_only,
            "planner_only_success": planner_only,
            "neither_success": neither,
            "planner_minus_direct_percentage_points": 100.0 * (planner_success - direct_success),
            "planner_retention_of_direct_success": (
                planner_success / direct_success if direct_success else None
            ),
            "mcnemar_exact_two_sided_p": mcnemar_exact_two_sided(direct_only, planner_only),
        },
        "failure_examples": [
            {
                "action_uid": row["action_uid"],
                "split": row["planner"]["split"],
                "target_action": row["planner"]["target_action"],
                "target_value": row["planner"]["target_value"],
                "direct_prompt": row["direct"]["prompt"],
                "planner_prompt": row["planner"]["prompt"],
                "direct_prediction": row["direct"]["prediction"],
                "planner_prediction": row["planner"]["prediction"],
            }
            for row in paired_rows
            if row["direct"]["strict_hit"] and not row["planner"]["strict_hit"]
        ][:example_limit],
    }

    table: list[dict[str, Any]] = []
    for arm_name, rows in (
        ("direct_target_grounding", direct_rows),
        ("task_history_planner", planner_rows),
    ):
        for group_type, group_name, group_rows in [
            ("overall", "all", rows),
            *(
                ("split", name, [row for row in rows if row["split"] == name])
                for name in sorted({row["split"] for row in rows})
            ),
            *(
                ("action", name, [row for row in rows if row["target_action"] == name])
                for name in sorted({row["target_action"] for row in rows})
            ),
        ]:
            summary = summarize(group_rows)
            table.append({"arm": arm_name, "group_type": group_type, "group": group_name, **summary})
    return report, table


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = [name for name in rows[0] if not name.endswith("_95ci")]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report, table = build_report(
        benchmark_root=args.benchmark_root.expanduser().resolve(),
        direct_predictions_dir=args.direct_predictions_dir.expanduser().resolve(),
        planner_predictions_dir=args.planner_predictions_dir.expanduser().resolve(),
        direct_benchmark=args.direct_benchmark,
        planner_benchmark=args.planner_benchmark,
        example_limit=args.examples,
    )
    (output_dir / "planner_suitability.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(output_dir / "planner_suitability.csv", table)
    print(json.dumps(report["arms"], indent=2, ensure_ascii=False), flush=True)
    print(json.dumps(report["paired_comparison"], indent=2), flush=True)


if __name__ == "__main__":
    main()
