#!/usr/bin/env python3
"""Run a sharded llama.cpp LLaDA-o full-page grounding benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


_SELECTED_RE = re.compile(
    r"tile_retrieval selected_sources=\[([^\]]*)\] "
    r"overview=(true|false) latency=([0-9.]+)"
)
_TOKEN_RE = re.compile(
    r"tile_retrieval resident_tokens=(\d+) dense_tokens=(\d+) "
    r"image_token_ratio=([0-9.]+) prefix_token_ratio=([0-9.]+) "
    r"yarn_factor=([0-9.]+)"
)
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
    parser.add_argument("--ctx-size", type=int, default=65536)
    parser.add_argument("--gpu-layers", type=int, default=999)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--full-page-tile-size", type=int, default=980)
    parser.add_argument("--tile-retrieval-topk", type=int, default=4)
    parser.add_argument("--tile-retrieval-mask-rounds", type=int, default=2)
    parser.add_argument("--yarn-factor", type=float, default=8.0)
    parser.add_argument("--yarn-orig-ctx", type=int, default=16384)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    if not 1 <= args.limit <= 100:
        parser.error("--limit must be in [1, 100]")
    if args.num_shards <= 0:
        parser.error("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, num-shards)")
    if not 1 <= args.full_page_tile_size <= 980:
        parser.error("--full-page-tile-size must be in [1, 980]")
    if args.tile_retrieval_topk < 0:
        parser.error("--tile-retrieval-topk must be non-negative")
    if args.tile_retrieval_mask_rounds <= 0:
        parser.error("--tile-retrieval-mask-rounds must be positive")
    if args.yarn_factor <= 0 or args.yarn_orig_ctx <= 0:
        parser.error("YaRN parameters must be positive")
    return args


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sample_id_sha256(samples: list[dict[str, Any]]) -> str:
    payload = "".join(f"{sample['sample_id']}\n" for sample in samples)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row["sample_id"]) for row in load_jsonl(path)}


class NvidiaMemorySampler:
    """Best-effort peak device-memory sampling via nvidia-smi."""

    def __init__(self) -> None:
        self.peak_mib: int | None = None
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        try:
            self.device_index = int(visible_devices[0])
        except (IndexError, ValueError):
            self.device_index = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> int | None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        return self.peak_mib

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    timeout=2.0,
                )
                for line in result.stdout.splitlines():
                    fields = [field.strip() for field in line.split(",")]
                    if len(fields) != 2:
                        continue
                    if (
                        self.device_index is not None
                        and int(fields[0]) != self.device_index
                    ):
                        continue
                    used_mib = int(fields[1])
                    self.peak_mib = max(self.peak_mib or 0, used_mib)
            except (FileNotFoundError, OSError, subprocess.SubprocessError, ValueError):
                return
            self._stop.wait(0.5)


def run_process(
    command: list[str],
    timeout: float,
) -> tuple[int | None, str, str, str | None, int | None]:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=os.environ.copy(),
    )
    sampler = NvidiaMemorySampler()
    sampler.start()
    error: str | None = None
    returncode: int | None = None
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        returncode = process.returncode
        if returncode != 0:
            error = f"runner exited with status {returncode}"
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        error = f"runner timed out after {timeout:.1f} seconds"
    peak_memory_mib = sampler.stop()
    return returncode, stdout.strip(), stderr, error, peak_memory_mib


def parse_runtime(stderr: str, dense_source_indices: list[int]) -> dict[str, Any]:
    selected_match = _SELECTED_RE.search(stderr)
    token_match = _TOKEN_RE.search(stderr)
    generation_match = _GENERATION_RE.search(stderr)
    iterations = [int(value) for value in _ITERATION_RE.findall(stderr)]

    if selected_match is None:
        selected_sources = dense_source_indices
        overview = True
        retrieval_seconds = 0.0
    else:
        selected_sources = [
            int(value)
            for value in selected_match.group(1).split(",")
            if value.strip()
        ]
        overview = selected_match.group(2) == "true"
        retrieval_seconds = float(selected_match.group(3))

    runtime: dict[str, Any] = {
        "convergence_steps": max(iterations) if iterations else None,
        "selected_source_tiles": selected_sources,
        "overview_retained": overview,
        "retrieval_seconds": retrieval_seconds,
        "generation_seconds": (
            float(generation_match.group(1)) if generation_match else None
        ),
    }
    if token_match:
        runtime.update(
            {
                "resident_image_tokens": int(token_match.group(1)),
                "dense_image_tokens": int(token_match.group(2)),
                "image_token_ratio": float(token_match.group(3)),
                "prefix_token_ratio": float(token_match.group(4)),
                "runtime_yarn_factor": float(token_match.group(5)),
            }
        )
    return runtime


def main() -> None:
    args = parse_args()
    repo = args.repo.expanduser().resolve()
    sys.path.insert(0, str(repo))
    from eval.gui_grounding.metrics import parse_action
    from eval.gui_grounding.ocr_fullpage_retrieval import native_instruction

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
        "backend": "llama.cpp-d2f-fullpage",
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
        "full_page_tiles": True,
        "full_page_overview": True,
        "full_page_tile_size": args.full_page_tile_size,
        "tile_retrieval_topk": args.tile_retrieval_topk,
        "tile_retrieval_mask_rounds": args.tile_retrieval_mask_rounds,
        "yarn_factor": args.yarn_factor,
        "yarn_orig_ctx": args.yarn_orig_ctx,
        "generation_length": 64,
        "block_length": 16,
        "block_add_threshold": 0.1,
        "decoded_token_threshold": 0.95,
        "skip_threshold": 0.9,
        "temperature": 0.0,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "shard_sample_ids": [str(sample["sample_id"]) for _, sample in shard_samples],
    }
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != config:
            raise RuntimeError(f"refusing to resume with a changed config: {config_path}")
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
        for local_index, (global_index, sample) in enumerate(shard_samples, start=1):
            sample_id = str(sample["sample_id"])
            if sample_id in completed:
                print(
                    f"[{local_index}/{len(shard_samples)}] already complete: {sample_id}",
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
                "--full-page-tiles",
                "--full-page-tile-size",
                str(args.full_page_tile_size),
                "--tile-retrieval-topk",
                str(args.tile_retrieval_topk),
                "--tile-retrieval-mask-rounds",
                str(args.tile_retrieval_mask_rounds),
                "--ctx-size",
                str(args.ctx_size),
                "--yarn-factor",
                str(args.yarn_factor),
                "--yarn-orig-ctx",
                str(args.yarn_orig_ctx),
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
            dense_source_indices = [
                int(tile["index"]) for tile in sample.get("tile_layout", [])
            ]
            runtime = parse_runtime(stderr, dense_source_indices)

            record: dict[str, Any] = {
                "backend": "llama.cpp-d2f-fullpage",
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
                **runtime,
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
                f"[{local_index}/{len(shard_samples)} global={global_index + 1}] "
                f"{latency:.2f}s prediction={stdout!r} error={error}",
                flush=True,
            )
            if error is not None and args.fail_fast:
                raise RuntimeError(error)


if __name__ == "__main__":
    main()
