#!/usr/bin/env python3
"""Merge sharded GUI-grounding predictions and produce paper-style metrics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from eval.gui_grounding.metrics import score_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--benchmarks", help="comma-separated subset; default is all prepared")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--limit", type=int, help="expected maximum rows per benchmark")
    return parser.parse_args()


def load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
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


def load_targets(
    root: Path, manifest: dict[str, Any], benchmark: str, limit: int | None
) -> dict[str, dict[str, Any]]:
    targets: dict[str, dict[str, Any]] = {}
    path = root / manifest["benchmarks"][benchmark]["path"]
    for index, row in enumerate(load_jsonl(path)):
        if limit is not None and index >= limit:
            break
        sample_id = str(row["sample_id"])
        if sample_id in targets:
            raise RuntimeError(f"duplicate target sample: {sample_id}")
        targets[sample_id] = row
    return targets


def load_predictions(directory: Path, benchmark: str) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    paths = sorted((directory / benchmark).glob("part-*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no prediction shards for {benchmark} below {directory}")
    for path in paths:
        for row in load_jsonl(path):
            sample_id = str(row["sample_id"])
            if sample_id in predictions:
                raise RuntimeError(f"duplicate prediction for {sample_id}")
            predictions[sample_id] = row
    return predictions


def joined_records(
    targets: dict[str, dict[str, Any]], predictions: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    joined: list[dict[str, Any]] = []
    for sample_id, target in targets.items():
        prediction = predictions.get(sample_id)
        if prediction is None:
            continue
        row = dict(prediction)
        row["target_action"] = target["target_action"]
        row["target_bbox_1000"] = target["target_bbox_1000"]
        row["split"] = target.get("split", "test")
        joined.append(row)
    return joined


def paper_row(benchmark: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "benchmark": benchmark,
        "samples": metrics["num_samples"],
        "SSR (%)": 100.0 * metrics["ssr_point_only"],
        "Joint SSR (%)": 100.0 * metrics["joint_step_success"],
        "Action F1 (%)": 100.0 * metrics["action_f1_macro_present"],
        "Action macro-F1 all (%)": 100.0 * metrics["action_f1_macro_all"],
        "Action accuracy (%)": 100.0 * metrics["action_accuracy"],
        "Parse rate (%)": 100.0 * metrics["parse_rate"],
        "Conv. steps": metrics["convergence_steps"]["mean"],
        "Avg latency (s)": metrics["latency_seconds"]["mean"],
        "P95 latency (s)": metrics["latency_seconds"]["p95"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = args.benchmark_root.expanduser().resolve()
    predictions_dir = args.predictions_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else predictions_dir / "scores"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    prepared = list(manifest["benchmarks"])
    requested = (
        [value.strip() for value in args.benchmarks.split(",") if value.strip()]
        if args.benchmarks
        else prepared
    )
    unavailable = [benchmark for benchmark in requested if benchmark not in manifest["benchmarks"]]
    if unavailable:
        print(
            "Skipping unavailable benchmarks: " + ", ".join(unavailable),
            file=sys.stderr,
            flush=True,
        )
    benchmarks = [benchmark for benchmark in requested if benchmark in manifest["benchmarks"]]
    if not benchmarks:
        raise RuntimeError("none of the requested benchmarks is prepared")

    result: dict[str, Any] = {
        "paper": manifest.get("paper"),
        "benchmark_manifest": str((root / "manifest.json").resolve()),
        "exact_paper_reproduction": manifest.get("exact_paper_reproduction", False),
        "protocol_notes": manifest.get("protocol_notes", []),
        "benchmarks": {},
        "subgroups": {},
        "coverage": {},
    }
    table: list[dict[str, Any]] = []

    for benchmark in benchmarks:
        targets = load_targets(root, manifest, benchmark, args.limit)
        predictions = load_predictions(predictions_dir, benchmark)
        unexpected = sorted(set(predictions) - set(targets))
        missing = sorted(set(targets) - set(predictions))
        if unexpected:
            raise RuntimeError(
                f"{benchmark} has {len(unexpected)} predictions outside the target set"
            )
        if missing and not args.allow_partial:
            raise RuntimeError(
                f"{benchmark} is missing {len(missing)}/{len(targets)} predictions"
            )
        joined = joined_records(targets, predictions)
        metrics = score_records(joined)
        result["benchmarks"][benchmark] = metrics
        result["coverage"][benchmark] = {
            "targets": len(targets),
            "predictions": len(predictions),
            "joined": len(joined),
            "missing": len(missing),
        }
        table.append(paper_row(benchmark, metrics))

        by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in joined:
            by_split[str(row.get("split") or "test")].append(row)
        if len(by_split) > 1:
            result["subgroups"][benchmark] = {
                split: score_records(rows) for split, rows in sorted(by_split.items())
            }

    if table:
        latency_values = [
            row["Avg latency (s)"]
            for row in table
            if row["Avg latency (s)"] is not None
        ]
        result["macro_average_across_benchmarks"] = {
            "SSR (%)": statistics.fmean(row["SSR (%)"] for row in table),
            "Joint SSR (%)": statistics.fmean(row["Joint SSR (%)"] for row in table),
            "Action F1 (%)": statistics.fmean(row["Action F1 (%)"] for row in table),
            "Avg latency (s)": (
                statistics.fmean(latency_values) if latency_values else None
            ),
        }

    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(output_dir / "results.csv", table)
    print(json.dumps(table, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
