#!/usr/bin/env python3
"""Run a sharded llama.cpp LLaDA-o native-image grounding benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any


_GENERATION_RE = re.compile(r"D2F generation_seconds=([0-9.]+)")
_ITERATION_RE = re.compile(r"D2F iteration ([0-9]+):")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--benchmark", default="mind2web_fullpage")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mmproj", type=Path, required=True)
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--ctx-size", type=int, default=16384)
    parser.add_argument("--gpu-layers", type=int, default=999)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    if not 1 <= args.limit <= 100:
        parser.error("--limit must be in [1, 100]")
    if args.num_shards <= 0:
        parser.error("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, num-shards)")
    if args.ctx_size <= 0:
        parser.error("--ctx-size must be positive")
    return args


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row["sample_id"]) for row in load_jsonl(path)}


def sample_id_sha256(samples: list[dict[str, Any]]) -> str:
    payload = "".join(f"{sample['sample_id']}\n" for sample in samples)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_runtime(stderr: str) -> dict[str, Any]:
    iterations = [int(value) for value in _ITERATION_RE.findall(stderr)]
    generation_match = _GENERATION_RE.search(stderr)
    return {
        "convergence_steps": max(iterations) if iterations else None,
        "generation_seconds": (
            float(generation_match.group(1)) if generation_match else None
        ),
    }


def main() -> None:
    args = parse_args()
    repo = args.repo.expanduser().resolve()
    sys.path.insert(0, str(repo))
    from eval.gui_grounding.metrics import parse_action
    from eval.gui_grounding.ocr_fullpage_retrieval import native_instruction
    from eval.gui_grounding.run_llamacpp_fullpage_benchmark import run_process

    benchmark_root = args.benchmark_root.expanduser().resolve()
    manifest = json.loads(
        (benchmark_root / "manifest.json").read_text(encoding="utf-8")
    )
    sample_path = benchmark_root / manifest["benchmarks"][args.benchmark]["path"]
    samples = load_jsonl(sample_path)[: args.limit]
    shard_samples = [
        (index, sample)
        for index, sample in enumerate(samples)
        if index % args.num_shards == args.shard_index
    ]

    binary = args.binary.expanduser().resolve()
    model = args.model.expanduser().resolve()
    mmproj = args.mmproj.expanduser().resolve()
    lora = args.lora.expanduser().resolve()
    for path in (binary, model, mmproj, lora):
        if not path.is_file():
            raise FileNotFoundError(path)

    output_dir = args.output_dir.expanduser().resolve()
    shard_dir = output_dir / args.benchmark
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"part-{args.shard_index:05d}.jsonl"
    log_path = output_dir / f"runner-shard-{args.shard_index:05d}.stderr.log"
    config_path = output_dir / f"run-config-shard-{args.shard_index:05d}.json"

    config = {
        "backend": "llama.cpp-d2f-native",
        "benchmark_root": str(benchmark_root),
        "benchmark": args.benchmark,
        "limit": args.limit,
        "sample_id_sha256": sample_id_sha256(samples),
        "binary": str(binary),
        "model": str(model),
        "mmproj": str(mmproj),
        "lora": str(lora),
        "ctx_size": args.ctx_size,
        "gpu_layers": args.gpu_layers,
        "threads": args.threads,
        "generation_length": 64,
        "block_length": 16,
        "block_add_threshold": 0.1,
        "decoded_token_threshold": 0.95,
        "skip_threshold": 0.9,
        "temperature": 0.0,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "shard_sample_ids": [
            str(sample["sample_id"]) for _, sample in shard_samples
        ],
    }
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != config:
            raise RuntimeError(
                f"refusing to resume with a changed config: {config_path}"
            )
    else:
        config_path.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    completed = load_completed(shard_path)
    mode = "a" if shard_path.exists() else "w"
    with shard_path.open(mode, encoding="utf-8") as output, log_path.open(
        "a", encoding="utf-8"
    ) as log:
        for local_index, (global_index, sample) in enumerate(
            shard_samples, start=1
        ):
            sample_id = str(sample["sample_id"])
            if sample_id in completed:
                print(
                    f"[{local_index}/{len(shard_samples)}] already complete: "
                    f"{sample_id}",
                    flush=True,
                )
                continue

            image = benchmark_root / sample["image"]
            operation = native_instruction(str(sample["prompt"]))
            command = [
                str(binary),
                "--model",
                str(model),
                "--mmproj",
                str(mmproj),
                "--lora",
                str(lora),
                "--image",
                str(image),
                "--prompt",
                operation,
                "--ctx-size",
                str(args.ctx_size),
                "--gpu-layers",
                str(args.gpu_layers),
                "--threads",
                str(args.threads),
            ]

            started = time.perf_counter()
            returncode, stdout, stderr, error, peak_memory_mib = run_process(
                command, args.timeout
            )
            latency = time.perf_counter() - started
            parsed = parse_action(stdout if error is None else "")

            record: dict[str, Any] = {
                "backend": "llama.cpp-d2f-native",
                "benchmark": sample["benchmark"],
                "sample_id": sample_id,
                "sample_index": global_index,
                "split": sample["split"],
                "operation": operation,
                "prediction": stdout,
                "raw_prediction": stdout,
                "parse_error": parsed.error,
                "target_action": sample["target_action"],
                "target_bbox_1000": sample["target_bbox_1000"],
                "target_value": sample.get("target_value", ""),
                "latency_seconds": latency,
                "generated_tokens": 64,
                "error": error,
                "runner_returncode": returncode,
                "peak_gpu_memory_mib": peak_memory_mib,
                "runtime_input_protocol": "native_single_image",
                "source_width": sample.get("image_width"),
                "source_height": sample.get("image_height"),
                **parse_runtime(stderr),
            }
            if parsed.action is not None and parsed.bbox_1000 is not None:
                record["predicted_action"] = parsed.action
                record["predicted_bbox_1000"] = list(parsed.bbox_1000)
                record["predicted_value"] = parsed.value
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()

            log.write(
                f"\n===== sample={global_index + 1}/{len(samples)} "
                f"id={sample_id} latency={latency:.3f}s error={error!r} =====\n"
            )
            log.write(stderr)
            if stderr and not stderr.endswith("\n"):
                log.write("\n")
            log.flush()

            print(
                f"[{local_index}/{len(shard_samples)} "
                f"global={global_index + 1}] {latency:.2f}s "
                f"prediction={stdout!r} error={error}",
                flush=True,
            )
            if error is not None and args.fail_fast:
                raise RuntimeError(error)


if __name__ == "__main__":
    main()
