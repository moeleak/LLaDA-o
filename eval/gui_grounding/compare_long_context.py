#!/usr/bin/env python3
"""Compare unscaled and YaRN D2F runs on the full-page Mind2Web set."""

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
    parser.add_argument("--unscaled-dir", type=Path, required=True)
    parser.add_argument("--yarn-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmark", default="mind2web_fullpage")
    parser.add_argument("--original-max-model-len", type=int, default=16_384)
    parser.add_argument(
        "--require-true-long-rope",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "require sequential visual positions above the original limit "
            "and an uncompressed dense KV prefix"
        ),
    )
    return parser.parse_args()


def evaluate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "quality": score_records(rows),
        "runtime": runtime_metrics(rows),
    }


def percent_delta(before: float | None, after: float | None) -> float | None:
    if before is None or after is None or before == 0:
        return None
    return 100.0 * (after - before) / before


def comparison_row(
    name: str,
    unscaled: dict[str, Any],
    yarn: dict[str, Any],
) -> dict[str, Any]:
    unscaled_quality = unscaled["quality"]
    yarn_quality = yarn["quality"]
    unscaled_runtime = unscaled["runtime"]
    yarn_runtime = yarn["runtime"]
    latency_before = unscaled_quality["latency_seconds"]["mean"]
    latency_after = yarn_quality["latency_seconds"]["mean"]
    throughput_before = unscaled_runtime["total_tokens_per_second"]["mean"]
    throughput_after = yarn_runtime["total_tokens_per_second"]["mean"]
    memory_before = unscaled_runtime["peak_memory_allocated_gib"]["mean"]
    memory_after = yarn_runtime["peak_memory_allocated_gib"]["mean"]
    unscaled_max_position = unscaled_runtime["max_generation_position"]["max"]
    yarn_max_position = yarn_runtime["max_generation_position"]["max"]
    return {
        "bucket": name,
        "samples": yarn_quality["num_samples"],
        "unscaled_ssr_pct": 100.0 * unscaled_quality["ssr_point_only"],
        "yarn_ssr_pct": 100.0 * yarn_quality["ssr_point_only"],
        "ssr_delta_pp": 100.0
        * (
            yarn_quality["ssr_point_only"]
            - unscaled_quality["ssr_point_only"]
        ),
        "unscaled_action_f1_pct": 100.0
        * unscaled_quality["action_f1_macro_present"],
        "yarn_action_f1_pct": 100.0
        * yarn_quality["action_f1_macro_present"],
        "action_f1_delta_pp": 100.0
        * (
            yarn_quality["action_f1_macro_present"]
            - unscaled_quality["action_f1_macro_present"]
        ),
        "unscaled_parse_rate_pct": 100.0 * unscaled_quality["parse_rate"],
        "yarn_parse_rate_pct": 100.0 * yarn_quality["parse_rate"],
        "unscaled_latency_s": latency_before,
        "yarn_latency_s": latency_after,
        "latency_delta_pct": percent_delta(latency_before, latency_after),
        "unscaled_tokens_per_s": throughput_before,
        "yarn_tokens_per_s": throughput_after,
        "throughput_delta_pct": percent_delta(
            throughput_before, throughput_after
        ),
        "unscaled_peak_allocated_gib": memory_before,
        "yarn_peak_allocated_gib": memory_after,
        "peak_allocated_delta_gib": (
            None
            if memory_before is None or memory_after is None
            else memory_after - memory_before
        ),
        "unscaled_max_generation_position": unscaled_max_position,
        "yarn_max_generation_position": yarn_max_position,
        "unscaled_errors": unscaled_runtime["errors"],
        "yarn_errors": yarn_runtime["errors"],
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


def main() -> None:
    args = parse_args()
    root = args.benchmark_root.expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    targets = load_targets(root, manifest, args.benchmark, None)
    runs: dict[str, list[dict[str, Any]]] = {}
    protocol_validation: dict[str, Any] = {}
    for name, directory in (
        ("unscaled", args.unscaled_dir),
        ("yarn", args.yarn_dir),
    ):
        predictions = load_predictions(
            directory.expanduser().resolve(), args.benchmark
        )
        missing = sorted(set(targets) - set(predictions))
        unexpected = sorted(set(predictions) - set(targets))
        if missing or unexpected:
            raise RuntimeError(
                f"{name} coverage mismatch: missing={len(missing)} "
                f"unexpected={len(unexpected)}"
            )
        runs[name] = joined_records(targets, predictions)
        if args.require_true_long_rope:
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
        unscaled_metrics = evaluate(selected["unscaled"])
        yarn_metrics = evaluate(selected["yarn"])
        detailed[bucket] = {
            "unscaled": unscaled_metrics,
            "yarn": yarn_metrics,
        }
        rows.append(
            comparison_row(
                bucket, unscaled_metrics, yarn_metrics
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
        "original_16k_capacity": {
            "accepted": len(targets) - original_rejections,
            "rejected": original_rejections,
            "total": len(targets),
        },
        "true_long_rope_validation": protocol_validation,
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
        "# LLaDA-o D2F 16K–64K full-page comparison",
        "",
        (
            f"Original 16K capacity rejected {original_rejections}/"
            f"{len(targets)} prepared samples."
        ),
        (
            "Maximum generation RoPE position: "
            f"unscaled={rows[0]['unscaled_max_generation_position']}, "
            f"YaRN={rows[0]['yarn_max_generation_position']}."
        ),
        (
            "True-long-RoPE protocol validation: "
            f"{'passed' if protocol_validation else 'not requested'}."
        ),
        "",
        "| Bucket | N | SSR unscaled | SSR YaRN | Δ SSR | "
        "Latency unscaled | Latency YaRN | Δ latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            f"| {row['bucket']} | {row['samples']} | "
            f"{row['unscaled_ssr_pct']:.2f}% | "
            f"{row['yarn_ssr_pct']:.2f}% | "
            f"{row['ssr_delta_pp']:+.2f} pp | "
            f"{row['unscaled_latency_s'] or 0:.3f}s | "
            f"{row['yarn_latency_s'] or 0:.3f}s | "
            f"{row['latency_delta_pct'] or 0:+.2f}% |"
        )
    (output / "comparison.md").write_text("\n".join(markdown) + "\n")
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
