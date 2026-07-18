#!/usr/bin/env python3
"""Run one sharded LLaDA-o GUI-grounding benchmark worker."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterator

import torch

from demo_pipeline import LLaDAMultimodalDemo, set_seed
from eval.gui_grounding.metrics import parse_action
from eval.gui_grounding.reproducibility import paired_sample_seed


DEFAULT_BENCHMARKS = "mind2web,screenspot_web_text,screenspot_web_icon,visualwebarena"


def optional_float(value: str) -> float | None:
    if value.lower() in {"none", "null", "off", "fixed"}:
        return None
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("confidence threshold must be in [0,1] or 'none'")
    return parsed


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None else int(raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--benchmarks", default=DEFAULT_BENCHMARKS)
    parser.add_argument("--rank", type=int, default=env_int("SLURM_PROCID", 0))
    parser.add_argument("--world-size", type=int, default=env_int("SLURM_NTASKS", 1))
    parser.add_argument("--limit", type=int, help="maximum rows per benchmark before sharding")
    parser.add_argument("--block-length", type=int, default=64)
    parser.add_argument("--diffusion-steps", type=int, default=64)
    parser.add_argument("--confidence-threshold", type=optional_float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-mem-per-gpu", default="90GiB")
    parser.add_argument("--offload-dir", type=Path, default=Path("/tmp/lladao-gui-eval-offload"))
    parser.add_argument("--flush-every", type=int, default=1)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if args.rank < 0 or args.world_size <= 0 or args.rank >= args.world_size:
        parser.error("rank must satisfy 0 <= rank < world-size")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.block_length <= 0 or args.diffusion_steps <= 0:
        parser.error("block length and diffusion steps must be positive")
    if args.flush_every <= 0:
        parser.error("--flush-every must be positive")
    return args


def selected_benchmarks(args: argparse.Namespace, manifest: dict[str, Any]) -> list[str]:
    requested = [value.strip() for value in args.benchmarks.split(",") if value.strip()]
    available = manifest.get("benchmarks", {})
    missing = [benchmark for benchmark in requested if benchmark not in available]
    if missing:
        print(
            "Skipping unavailable benchmarks: " + ", ".join(missing),
            file=sys.stderr,
            flush=True,
        )
    selected = [benchmark for benchmark in requested if benchmark in available]
    if not selected:
        raise RuntimeError("none of the requested benchmarks is prepared")
    return selected


def iter_samples(
    root: Path,
    manifest: dict[str, Any],
    benchmark: str,
    *,
    rank: int,
    world_size: int,
    limit: int | None,
) -> Iterator[dict[str, Any]]:
    path = root / manifest["benchmarks"][benchmark]["path"]
    with path.open(encoding="utf-8") as handle:
        logical_index = 0
        for line in handle:
            if not line.strip():
                continue
            if limit is not None and logical_index >= limit:
                break
            if logical_index % world_size == rank:
                yield json.loads(line)
            logical_index += 1


def load_completed(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                completed.add(str(json.loads(line)["sample_id"]))
            except (json.JSONDecodeError, KeyError) as exc:
                raise RuntimeError(
                    f"cannot resume malformed {path}:{line_number}: {exc}"
                ) from exc
    return completed


def synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def infer_one(
    model: LLaDAMultimodalDemo,
    root: Path,
    sample: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    inference_seed = paired_sample_seed(sample, args.seed)
    set_seed(inference_seed)
    synchronize_cuda()
    started = time.perf_counter()
    model_result = model.understand(
        root / sample["image"],
        sample["prompt"],
        block_length=args.block_length,
        steps_per_block=args.diffusion_steps,
        max_blocks=1,
        temperature=args.temperature,
        cfg_scale=args.cfg_scale,
        confidence_threshold=args.confidence_threshold,
    )
    synchronize_cuda()
    latency = time.perf_counter() - started
    parsed = parse_action(model_result["text"])
    return {
        "sample_id": sample["sample_id"],
        "benchmark": sample["benchmark"],
        "split": sample["split"],
        "prediction": model_result["text"],
        "raw_prediction": model_result["raw_text"],
        "predicted_action": parsed.action,
        "predicted_bbox_1000": list(parsed.bbox_1000) if parsed.bbox_1000 else None,
        "predicted_value": parsed.value,
        "parse_error": parsed.error,
        "target_action": sample["target_action"],
        "target_bbox_1000": sample["target_bbox_1000"],
        "target_value": sample.get("target_value", ""),
        "latency_seconds": latency,
        "model_elapsed_seconds": model_result["elapsed_seconds"],
        "convergence_steps": model_result["convergence_steps"],
        "valid_tokens": model_result["valid_tokens"],
        "generated_tokens": model_result["total_tokens"],
        "generation_stats": model_result["generation_stats"],
        "inference_seed": inference_seed,
        "error": None,
    }


def error_record(sample: dict[str, Any], exc: BaseException) -> dict[str, Any]:
    return {
        "sample_id": sample["sample_id"],
        "benchmark": sample["benchmark"],
        "split": sample["split"],
        "prediction": "",
        "raw_prediction": "",
        "predicted_action": None,
        "predicted_bbox_1000": None,
        "predicted_value": "",
        "parse_error": "inference_error",
        "target_action": sample["target_action"],
        "target_bbox_1000": sample["target_bbox_1000"],
        "target_value": sample.get("target_value", ""),
        "latency_seconds": None,
        "model_elapsed_seconds": None,
        "convergence_steps": None,
        "valid_tokens": None,
        "generated_tokens": None,
        "generation_stats": None,
        "inference_seed": paired_sample_seed(sample, args.seed),
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(limit=20),
    }


def run_config(args: argparse.Namespace, benchmarks: list[str]) -> dict[str, Any]:
    return {
        "model_path": str(args.model_path.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "benchmark_root": str(args.benchmark_root.resolve()),
        "benchmarks": benchmarks,
        "rank": args.rank,
        "world_size": args.world_size,
        "limit_per_benchmark": args.limit,
        "block_length": args.block_length,
        "generation_length": args.block_length,
        "diffusion_steps": args.diffusion_steps,
        "confidence_threshold": args.confidence_threshold,
        "temperature": args.temperature,
        "cfg_scale": args.cfg_scale,
        "seed": args.seed,
        "sample_seed_policy": "sha256(base_seed, provenance.action_uid || sample_id)",
        "latency_scope": (
            "synchronized image decode, preprocessing, and generation; "
            "model loading and warmup excluded"
        ),
    }


def main() -> None:
    args = parse_args()
    root = args.benchmark_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    benchmarks = selected_benchmarks(args, manifest)
    set_seed(args.seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / f"run-config-rank-{args.rank:05d}.json"
    config_path.write_text(
        json.dumps(run_config(args, benchmarks), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        f"Rank {args.rank}/{args.world_size}: loading {args.checkpoint}",
        flush=True,
    )
    model = LLaDAMultimodalDemo.from_pretrained(
        model_path=args.model_path,
        checkpoint_path=args.checkpoint,
        enable_visual_generation=False,
        max_mem_per_gpu=args.max_mem_per_gpu,
        offload_dir=args.offload_dir / f"rank-{args.rank:05d}",
    )

    warmup_candidates: list[dict[str, Any]] = []
    for benchmark in benchmarks:
        warmup_candidates.extend(
            itertools.islice(
                iter_samples(
                    root,
                    manifest,
                    benchmark,
                    rank=args.rank,
                    world_size=args.world_size,
                    limit=args.limit,
                ),
                args.warmup,
            )
        )
        if len(warmup_candidates) >= args.warmup:
            break
    for sample in warmup_candidates[: args.warmup]:
        print(f"Rank {args.rank}: warmup {sample['sample_id']}", flush=True)
        infer_one(model, root, sample, args)

    total_written = 0
    for benchmark in benchmarks:
        benchmark_dir = output_dir / benchmark
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        output_path = benchmark_dir / f"part-{args.rank:05d}.jsonl"
        if args.no_resume and output_path.exists():
            output_path.unlink()
        completed = load_completed(output_path)
        pending = [
            sample
            for sample in iter_samples(
                root,
                manifest,
                benchmark,
                rank=args.rank,
                world_size=args.world_size,
                limit=args.limit,
            )
            if sample["sample_id"] not in completed
        ]
        print(
            f"Rank {args.rank}: {benchmark}: {len(pending)} pending, "
            f"{len(completed)} already complete",
            flush=True,
        )
        with output_path.open("a", encoding="utf-8", buffering=1) as handle:
            for index, sample in enumerate(pending, start=1):
                try:
                    record = infer_one(model, root, sample, args)
                except Exception as exc:
                    print(
                        f"Rank {args.rank}: failed {sample['sample_id']}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    if args.fail_fast:
                        raise
                    record = error_record(sample, exc)
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                if index % args.flush_every == 0:
                    handle.flush()
                    os.fsync(handle.fileno())
                total_written += 1
                if index == 1 or index % 10 == 0 or index == len(pending):
                    latency = record.get("latency_seconds")
                    latency_text = (
                        f"{latency:.3f}s"
                        if isinstance(latency, (int, float)) and math.isfinite(latency)
                        else "error"
                    )
                    print(
                        f"Rank {args.rank}: {benchmark} {index}/{len(pending)} "
                        f"{record.get('prediction')!r} {latency_text}",
                        flush=True,
                    )

    print(f"Rank {args.rank}: wrote {total_written} predictions", flush=True)


if __name__ == "__main__":
    main()
