#!/usr/bin/env python3
"""Compare the deployable original-16K and YaRN-128K GUI runtimes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from eval.gui_grounding.metrics import score_records
from eval.gui_grounding.score_benchmark import (
    context_bucket,
    joined_records,
    load_predictions,
    load_targets,
    runtime_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    baseline = parser.add_mutually_exclusive_group(required=True)
    baseline.add_argument("--original-dir", type=Path)
    baseline.add_argument(
        "--unscaled-dir",
        dest="original_dir",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--yarn-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmark", default="mind2web_fullpage")
    parser.add_argument("--original-max-model-len", type=int, default=16_384)
    parser.add_argument(
        "--require-original-vs-yarn",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "require a native-resized original 16K arm and a sequential, "
            "uncompressed YaRN arm above the original position limit"
        ),
    )
    return parser.parse_args()


def evaluate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "quality": score_records(rows),
        "runtime": runtime_metrics(rows),
    }


def load_run_config(directory: Path) -> dict[str, Any]:
    paths = sorted(directory.glob("run-config-rank-*.json"))
    if not paths:
        raise FileNotFoundError(f"no run config below {directory}")
    configs = [json.loads(path.read_text()) for path in paths]
    keys = (
        "max_model_len",
        "kv_cache_capacity",
        "rope_scaling",
        "rope_factor",
        "original_max_position_embeddings",
        "full_page_tiles",
        "full_page_position_mode",
        "kv_cache_compression",
    )
    reference = {key: configs[0].get(key) for key in keys}
    for path, config in zip(paths[1:], configs[1:]):
        actual = {key: config.get(key) for key in keys}
        if actual != reference:
            raise RuntimeError(
                f"inconsistent run protocol in {path}: "
                f"{actual} != {reference}"
            )
    return reference


def percent_delta(before: float | None, after: float | None) -> float | None:
    if before is None or after is None or before == 0:
        return None
    return 100.0 * (after - before) / before


def comparison_row(
    name: str,
    original: dict[str, Any],
    yarn: dict[str, Any],
) -> dict[str, Any]:
    original_quality = original["quality"]
    yarn_quality = yarn["quality"]
    original_runtime = original["runtime"]
    yarn_runtime = yarn["runtime"]
    latency_before = original_quality["latency_seconds"]["mean"]
    latency_after = yarn_quality["latency_seconds"]["mean"]
    throughput_before = original_runtime["total_tokens_per_second"]["mean"]
    throughput_after = yarn_runtime["total_tokens_per_second"]["mean"]
    memory_before = original_runtime["peak_memory_allocated_gib"]["mean"]
    memory_after = yarn_runtime["peak_memory_allocated_gib"]["mean"]
    original_max_position = original_runtime["max_generation_position"]["max"]
    yarn_max_position = yarn_runtime["max_generation_position"]["max"]
    return {
        "bucket": name,
        "samples": yarn_quality["num_samples"],
        "original_16k_ssr_pct": 100.0 * original_quality["ssr_point_only"],
        "yarn_ssr_pct": 100.0 * yarn_quality["ssr_point_only"],
        "ssr_delta_pp": 100.0
        * (
            yarn_quality["ssr_point_only"]
            - original_quality["ssr_point_only"]
        ),
        "original_16k_action_f1_pct": 100.0
        * original_quality["action_f1_macro_present"],
        "yarn_action_f1_pct": 100.0
        * yarn_quality["action_f1_macro_present"],
        "action_f1_delta_pp": 100.0
        * (
            yarn_quality["action_f1_macro_present"]
            - original_quality["action_f1_macro_present"]
        ),
        "original_16k_parse_rate_pct": 100.0
        * original_quality["parse_rate"],
        "yarn_parse_rate_pct": 100.0 * yarn_quality["parse_rate"],
        "original_16k_latency_s": latency_before,
        "yarn_latency_s": latency_after,
        "latency_delta_pct": percent_delta(latency_before, latency_after),
        "original_16k_tokens_per_s": throughput_before,
        "yarn_tokens_per_s": throughput_after,
        "throughput_delta_pct": percent_delta(
            throughput_before, throughput_after
        ),
        "original_16k_peak_allocated_gib": memory_before,
        "yarn_peak_allocated_gib": memory_after,
        "peak_allocated_delta_gib": (
            None
            if memory_before is None or memory_after is None
            else memory_after - memory_before
        ),
        "original_16k_max_generation_position": original_max_position,
        "yarn_max_generation_position": yarn_max_position,
        "original_16k_errors": original_runtime["errors"],
        "yarn_errors": yarn_runtime["errors"],
    }


def validate_original_16k(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    original_max_position: int,
) -> dict[str, Any]:
    expected_config = {
        "max_model_len": original_max_position,
        "kv_cache_capacity": original_max_position,
        "rope_scaling": "none",
        "full_page_tiles": False,
        "full_page_position_mode": "native",
        "kv_cache_compression": False,
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected_config.items()
        if config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"original 16K run config mismatch: {mismatches}")
    modes = sorted({str(row.get("position_mode")) for row in rows})
    protocols = sorted(
        {str(row.get("runtime_input_protocol")) for row in rows}
    )
    if modes != ["native"] or protocols != ["native_resize"]:
        raise RuntimeError(
            "original 16K predictions must use native positions and native "
            f"resize: modes={modes}, protocols={protocols}"
        )
    positions = [row.get("max_generation_position") for row in rows]
    dense_lengths = [row.get("dense_prefix_tokens") for row in rows]
    cached_lengths = [row.get("cached_prefix_tokens") for row in rows]
    generated = [row.get("generated_tokens") for row in rows]
    if any(not isinstance(value, int) for value in positions):
        raise RuntimeError("original 16K run is missing generation positions")
    if any(not isinstance(value, int) for value in dense_lengths + cached_lengths):
        raise RuntimeError("original 16K run is missing KV prefix lengths")
    if any(not isinstance(value, int) for value in generated):
        raise RuntimeError("original 16K run is missing generation lengths")
    if any(value >= original_max_position for value in positions):
        raise RuntimeError("original 16K run exceeded its position limit")
    if any(
        dense + output > original_max_position
        for dense, output in zip(dense_lengths, generated)
    ):
        raise RuntimeError("original 16K run exceeded its token capacity")
    compressed = sum(
        dense != cached
        for dense, cached in zip(dense_lengths, cached_lengths)
    )
    if compressed:
        raise RuntimeError(
            f"original 16K run has {compressed}/{len(rows)} compressed prefixes"
        )
    return {
        "position_mode": "native",
        "input_protocol": "native_resize",
        "samples": len(rows),
        "min_runtime_tokens": min(
            dense + output
            for dense, output in zip(dense_lengths, generated)
        ),
        "max_runtime_tokens": max(
            dense + output
            for dense, output in zip(dense_lengths, generated)
        ),
        "max_generation_position": max(positions),
        "compressed_prefixes": compressed,
    }


def validate_true_long_rope(
    name: str,
    rows: list[dict[str, Any]],
    *,
    original_max_position: int,
) -> dict[str, Any]:
    modes = sorted({str(row.get("position_mode")) for row in rows})
    if modes != ["sequential"]:
        raise RuntimeError(
            f"{name} is not a sequential-position run: modes={modes}"
        )
    positions = [row.get("max_generation_position") for row in rows]
    if any(not isinstance(value, int) for value in positions):
        raise RuntimeError(f"{name} is missing integer max generation positions")
    below_or_equal = sum(
        int(value) <= original_max_position for value in positions
    )
    if below_or_equal:
        raise RuntimeError(
            f"{name} has {below_or_equal}/{len(rows)} samples that do not "
            f"exceed position {original_max_position}"
        )
    prefix_lengths = [
        (row.get("dense_prefix_tokens"), row.get("cached_prefix_tokens"))
        for row in rows
    ]
    if any(
        not isinstance(dense, int) or not isinstance(cached, int)
        for dense, cached in prefix_lengths
    ):
        raise RuntimeError(f"{name} is missing integer KV prefix lengths")
    compressed = sum(dense != cached for dense, cached in prefix_lengths)
    if compressed:
        raise RuntimeError(
            f"{name} has {compressed}/{len(rows)} compressed KV prefixes"
        )
    return {
        "position_mode": "sequential",
        "samples": len(rows),
        "min_generation_position": min(positions),
        "max_generation_position": max(positions),
        "compressed_prefixes": compressed,
    }


def validate_yarn_128k_config(config: dict[str, Any]) -> None:
    expected = {
        "max_model_len": 131_072,
        "kv_cache_capacity": 65_536,
        "rope_scaling": "yarn",
        "rope_factor": 8.0,
        "original_max_position_embeddings": 16_384,
        "full_page_tiles": True,
        "full_page_position_mode": "sequential",
        "kv_cache_compression": False,
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"YaRN 128K run config mismatch: {mismatches}")


def main() -> None:
    args = parse_args()
    root = args.benchmark_root.expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    targets = load_targets(root, manifest, args.benchmark, None)
    runs: dict[str, list[dict[str, Any]]] = {}
    protocol_validation: dict[str, Any] = {}
    for name, directory in (
        ("original_16k", args.original_dir),
        ("yarn", args.yarn_dir),
    ):
        resolved_directory = directory.expanduser().resolve()
        predictions = load_predictions(resolved_directory, args.benchmark)
        missing = sorted(set(targets) - set(predictions))
        unexpected = sorted(set(predictions) - set(targets))
        if missing or unexpected:
            raise RuntimeError(
                f"{name} coverage mismatch: missing={len(missing)} "
                f"unexpected={len(unexpected)}"
            )
        runs[name] = joined_records(targets, predictions)
        if args.require_original_vs_yarn:
            config = load_run_config(resolved_directory)
            if name == "original_16k":
                protocol_validation[name] = validate_original_16k(
                    runs[name],
                    config,
                    original_max_position=args.original_max_model_len,
                )
            else:
                validate_yarn_128k_config(config)
                protocol_validation[name] = validate_true_long_rope(
                    name,
                    runs[name],
                    original_max_position=args.original_max_model_len,
                )

    rows: list[dict[str, Any]] = []
    detailed: dict[str, Any] = {}
    buckets = ["overall", "16k_32k", "32k_48k", "48k_64k"]
    for bucket in buckets:
        selected = {}
        for name, records in runs.items():
            selected[name] = (
                records
                if bucket == "overall"
                else [
                    row
                    for row in records
                    if context_bucket(row) == bucket
                ]
            )
        original_metrics = evaluate(selected["original_16k"])
        yarn_metrics = evaluate(selected["yarn"])
        detailed[bucket] = {
            "original_16k": original_metrics,
            "yarn": yarn_metrics,
        }
        rows.append(
            comparison_row(
                bucket, original_metrics, yarn_metrics
            )
        )

    original_rejections = sum(
        int((row.get("sequence_tokens") or {}).get("total", 0))
        > args.original_max_model_len
        for row in targets.values()
    )
    payload = {
        "benchmark": args.benchmark,
        "manifest": str((root / "manifest.json").resolve()),
        "uncompressed_source_vs_original_16k_capacity": {
            "at_or_below": len(targets) - original_rejections,
            "above": original_rejections,
            "total": len(targets),
        },
        "protocol_validation": protocol_validation,
        "comparison": detailed,
        "table": rows,
    }
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
        + "\n"
    )
    with (output / "comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    markdown = [
        "# LLaDA-o D2F original 16K vs YaRN 128K",
        "",
        (
            f"All {original_rejections}/{len(targets)} source sequences above "
            "16K were evaluated by the original arm using checkpoint-native "
            "single-image resize."
        ),
        (
            "Maximum generation RoPE position: "
            "original16K="
            f"{rows[0]['original_16k_max_generation_position']}, "
            f"YaRN={rows[0]['yarn_max_generation_position']}."
        ),
        (
            "Original-16K/YaRN-128K protocol validation: "
            f"{'passed' if protocol_validation else 'not requested'}."
        ),
        "",
        "| Source-length bucket | N | SSR original 16K | SSR YaRN 128K | "
        "Δ SSR | Latency original 16K | Latency YaRN 128K | Δ latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            f"| {row['bucket']} | {row['samples']} | "
            f"{row['original_16k_ssr_pct']:.2f}% | "
            f"{row['yarn_ssr_pct']:.2f}% | "
            f"{row['ssr_delta_pp']:+.2f} pp | "
            f"{row['original_16k_latency_s'] or 0:.3f}s | "
            f"{row['yarn_latency_s'] or 0:.3f}s | "
            f"{row['latency_delta_pct'] or 0:+.2f}% |"
        )
    (output / "comparison.md").write_text("\n".join(markdown) + "\n")
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
