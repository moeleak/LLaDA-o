#!/usr/bin/env python3
"""Summarize a Table 3 checkpoint sweep with fixed paper-comparison fields."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from eval.gui_grounding.metrics import score_records
from eval.gui_grounding.score_benchmark import (
    joined_records,
    load_predictions,
    load_targets,
)


TABLE3_SSR_PERCENT = 83.31
TABLE3_ACTION_F1_PERCENT = 99.0
_STEP_DIRECTORY = re.compile(r"^step-(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="directory containing step-XXXXXXX evaluation directories",
    )
    parser.add_argument("--run-name", default="s64-b64-ct095")
    parser.add_argument("--benchmark", default="mind2web")
    parser.add_argument(
        "--dom-benchmark-root",
        type=Path,
        help="optional original-target benchmark root used to rescore the same predictions",
    )
    parser.add_argument(
        "--require-steps",
        help="comma-separated steps that must have complete result files",
    )
    parser.add_argument(
        "--primary-step",
        type=int,
        help="predeclared primary checkpoint (for the reproduction, use the 10-epoch step)",
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=float,
        help="optional denominator used only to report estimated completed epochs",
    )
    parser.add_argument("--paper-ssr", type=float, default=TABLE3_SSR_PERCENT)
    parser.add_argument(
        "--paper-action-f1", type=float, default=TABLE3_ACTION_F1_PERCENT
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args()


def parse_required_steps(value: str | None) -> tuple[int, ...]:
    if not value:
        return ()
    steps: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        step = int(item)
        if step < 0:
            raise ValueError("required steps must be non-negative")
        steps.append(step)
    if len(steps) != len(set(steps)):
        raise ValueError("required steps must not contain duplicates")
    return tuple(steps)


def _percent(value: Any) -> float | None:
    return None if value is None else 100.0 * float(value)


def _step_from_result_path(path: Path) -> int:
    step_directory = path.parents[2].name
    match = _STEP_DIRECTORY.fullmatch(step_directory)
    if match is None:
        raise RuntimeError(f"cannot parse checkpoint step from {path}")
    return int(match.group(1))


def discover_results(results_root: Path, run_name: str) -> dict[int, Path]:
    discovered: dict[int, Path] = {}
    pattern = f"step-*/{run_name}/scores/results.json"
    for path in sorted(results_root.glob(pattern)):
        step = _step_from_result_path(path)
        if step in discovered:
            raise RuntimeError(f"duplicate evaluation result for step {step}")
        discovered[step] = path
    if not discovered:
        raise FileNotFoundError(f"no evaluation results matching {results_root / pattern}")
    return discovered


def _require_complete_coverage(result: dict[str, Any], benchmark: str, path: Path) -> None:
    coverage = result.get("coverage", {}).get(benchmark)
    if not isinstance(coverage, dict):
        raise RuntimeError(f"{path} has no coverage entry for {benchmark}")
    targets = int(coverage.get("targets", -1))
    predictions = int(coverage.get("predictions", -1))
    joined = int(coverage.get("joined", -1))
    missing = int(coverage.get("missing", -1))
    if targets <= 0 or predictions != targets or joined != targets or missing != 0:
        raise RuntimeError(
            f"{path} is not a complete {benchmark} evaluation: {coverage}"
        )


def rescore_with_target_root(
    benchmark_root: Path, predictions_dir: Path, benchmark: str
) -> dict[str, Any]:
    manifest_path = benchmark_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if benchmark not in manifest.get("benchmarks", {}):
        raise RuntimeError(f"{manifest_path} does not contain {benchmark}")

    targets = load_targets(benchmark_root, manifest, benchmark, None)
    predictions = load_predictions(predictions_dir, benchmark)
    missing = sorted(set(targets) - set(predictions))
    unexpected = sorted(set(predictions) - set(targets))
    if missing or unexpected:
        raise RuntimeError(
            f"cannot rescore {benchmark}: {len(missing)} missing and "
            f"{len(unexpected)} unexpected predictions"
        )
    return score_records(joined_records(targets, predictions))


def _subgroup_ssr(
    result: dict[str, Any], benchmark: str, split: str
) -> float | None:
    metrics = result.get("subgroups", {}).get(benchmark, {}).get(split)
    if not isinstance(metrics, dict):
        return None
    return _percent(metrics.get("ssr_point_only"))


def _optional_benchmark_ssr(result: dict[str, Any], benchmark: str) -> float | None:
    metrics = result.get("benchmarks", {}).get(benchmark)
    if not isinstance(metrics, dict):
        return None
    return _percent(metrics.get("ssr_point_only"))


def summarize_result(
    path: Path,
    *,
    benchmark: str,
    dom_benchmark_root: Path | None,
    paper_ssr: float,
    paper_action_f1: float,
    steps_per_epoch: float | None,
) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    _require_complete_coverage(result, benchmark, path)
    metrics = result.get("benchmarks", {}).get(benchmark)
    if not isinstance(metrics, dict):
        raise RuntimeError(f"{path} has no metrics for {benchmark}")

    step = _step_from_result_path(path)
    ssr = _percent(metrics.get("ssr_point_only"))
    action_f1 = _percent(metrics.get("action_f1_macro_present"))
    if ssr is None or action_f1 is None:
        raise RuntimeError(f"{path} is missing required Table 3 metrics")

    domain_ssr = _subgroup_ssr(result, benchmark, "test_domain")
    row: dict[str, Any] = {
        "step": step,
        "estimated_epochs": (
            None if steps_per_epoch is None else step / steps_per_epoch
        ),
        "samples": int(metrics["num_samples"]),
        "mind2web_ssr_pct": ssr,
        "mind2web_joint_ssr_pct": _percent(metrics.get("joint_step_success")),
        "mind2web_action_f1_pct": action_f1,
        "mind2web_action_accuracy_pct": _percent(metrics.get("action_accuracy")),
        "mind2web_parse_rate_pct": _percent(metrics.get("parse_rate")),
        "convergence_steps_mean": metrics.get("convergence_steps", {}).get("mean"),
        "latency_seconds_mean": metrics.get("latency_seconds", {}).get("mean"),
        "test_domain_ssr_pct": domain_ssr,
        "test_task_ssr_pct": _subgroup_ssr(result, benchmark, "test_task"),
        "test_website_ssr_pct": _subgroup_ssr(result, benchmark, "test_website"),
        "screenspot_web_text_ssr_pct": _optional_benchmark_ssr(
            result, "screenspot_web_text"
        ),
        "screenspot_web_icon_ssr_pct": _optional_benchmark_ssr(
            result, "screenspot_web_icon"
        ),
        "dom_target_ssr_pct": None,
        "dom_target_joint_ssr_pct": None,
        "paper_ssr_pct": paper_ssr,
        "paper_ssr_gap_pp": ssr - paper_ssr,
        "test_domain_paper_ssr_gap_pp": (
            None if domain_ssr is None else domain_ssr - paper_ssr
        ),
        "paper_action_f1_pct": paper_action_f1,
        "paper_action_f1_gap_pp": action_f1 - paper_action_f1,
        "results_json": str(path),
    }

    if dom_benchmark_root is not None:
        predictions_dir = path.parents[1]
        dom_metrics = rescore_with_target_root(
            dom_benchmark_root, predictions_dir, benchmark
        )
        row["dom_target_ssr_pct"] = _percent(dom_metrics["ssr_point_only"])
        row["dom_target_joint_ssr_pct"] = _percent(
            dom_metrics["joint_step_success"]
        )
    return row


def summarize_sweep(
    results_root: Path,
    *,
    run_name: str = "s64-b64-ct095",
    benchmark: str = "mind2web",
    dom_benchmark_root: Path | None = None,
    required_steps: Iterable[int] = (),
    primary_step: int | None = None,
    steps_per_epoch: float | None = None,
    paper_ssr: float = TABLE3_SSR_PERCENT,
    paper_action_f1: float = TABLE3_ACTION_F1_PERCENT,
) -> dict[str, Any]:
    results_root = results_root.expanduser().resolve()
    if steps_per_epoch is not None and steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be positive")
    discovered = discover_results(results_root, run_name)

    required = tuple(required_steps)
    missing = sorted(set(required) - set(discovered))
    if missing:
        raise RuntimeError(f"missing required evaluation steps: {missing}")
    if primary_step is not None and primary_step not in discovered:
        raise RuntimeError(f"primary step {primary_step} has no completed evaluation")

    dom_root = (
        dom_benchmark_root.expanduser().resolve()
        if dom_benchmark_root is not None
        else None
    )
    rows = [
        summarize_result(
            path,
            benchmark=benchmark,
            dom_benchmark_root=dom_root,
            paper_ssr=paper_ssr,
            paper_action_f1=paper_action_f1,
            steps_per_epoch=steps_per_epoch,
        )
        for _, path in sorted(discovered.items())
    ]
    primary = (
        next(row for row in rows if row["step"] == primary_step)
        if primary_step is not None
        else None
    )
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "results_root": str(results_root),
        "run_name": run_name,
        "benchmark": benchmark,
        "paper_reference": {
            "paper": "Towards GUI Agents: Vision-Language Diffusion Models for GUI Grounding",
            "arxiv": "2603.26211",
            "table": 3,
            "row": "Mind2Web-only, cropped, OCR-based target annotation, 10 epochs",
            "ssr_pct": paper_ssr,
            "action_f1_pct": paper_action_f1,
        },
        "comparison_policy": {
            "primary_step": primary_step,
            "intermediate_checkpoints": (
                "training diagnostics only; do not select a checkpoint on the test set"
            ),
            "mind2web_scope": (
                "combined official test_domain, test_task, and test_website splits; "
                "split-specific values are also reported because the paper does not "
                "publish its exact test-split selection"
            ),
            "dom_target_rescore": (
                "the same predictions rescored against original DOM boxes"
                if dom_root is not None
                else None
            ),
        },
        "required_steps": list(required),
        "completed_steps": [row["step"] for row in rows],
        "latest": rows[-1],
        "primary": primary,
        "rows": rows,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty checkpoint sweep")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    required_steps = parse_required_steps(args.require_steps)
    summary = summarize_sweep(
        args.results_root,
        run_name=args.run_name,
        benchmark=args.benchmark,
        dom_benchmark_root=args.dom_benchmark_root,
        required_steps=required_steps,
        primary_step=args.primary_step,
        steps_per_epoch=args.steps_per_epoch,
        paper_ssr=args.paper_ssr,
        paper_action_f1=args.paper_action_f1,
    )
    output_json = (
        args.output_json.expanduser().resolve()
        if args.output_json
        else args.results_root.expanduser().resolve() / "table3_sweep.json"
    )
    output_csv = (
        args.output_csv.expanduser().resolve()
        if args.output_csv
        else args.results_root.expanduser().resolve() / "table3_sweep.csv"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(output_csv, summary["rows"])
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
