#!/usr/bin/env python3
"""Create a deterministic, self-contained subset of a prepared GUI benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--benchmark", default="mind2web_fullpage")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(
            f"output root already exists; choose a new path: {output_root}"
        )

    source_manifest_path = source_root / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    details = source_manifest.get("benchmarks", {}).get(args.benchmark)
    if not isinstance(details, dict):
        raise KeyError(f"benchmark is not prepared: {args.benchmark}")
    source_samples_path = source_root / str(details["path"])
    rows = [
        json.loads(line)
        for line in source_samples_path.read_text().splitlines()
        if line.strip()
    ]
    if args.count > len(rows):
        raise ValueError(
            f"requested {args.count} rows from only {len(rows)} candidates"
        )

    selected_indices = sorted(
        random.Random(args.seed).sample(range(len(rows)), args.count)
    )
    selected = [rows[index] for index in selected_indices]
    sample_ids = [str(row["sample_id"]) for row in selected]
    if len(set(sample_ids)) != len(sample_ids):
        raise RuntimeError("selected benchmark contains duplicate sample IDs")

    sample_path = Path("samples") / f"{args.benchmark}.jsonl"
    output_samples_path = output_root / sample_path
    write_jsonl(output_samples_path, selected)
    for relative in sorted({str(row["image"]) for row in selected}):
        link_or_copy(source_root / relative, output_root / relative)

    sample_id_digest = hashlib.sha256(
        ("\n".join(sample_ids) + "\n").encode("utf-8")
    ).hexdigest()
    output_manifest = {
        key: value
        for key, value in source_manifest.items()
        if key not in {"benchmarks", "counters"}
    }
    output_manifest["benchmarks"] = {
        args.benchmark: {
            **details,
            "path": str(sample_path),
            "rows": len(selected),
            "sha256": sha256_file(output_samples_path),
        }
    }
    output_manifest["counters"] = {
        "source_candidates": len(rows),
        "selected": len(selected),
    }
    output_manifest["subset"] = {
        "algorithm": "python_random_sample_indices_then_source_order",
        "benchmark": args.benchmark,
        "count": len(selected),
        "seed": args.seed,
        "selected_sample_ids_sha256": sample_id_digest,
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
    }
    output_manifest.setdefault("protocol_notes", []).append(
        "This root is a deterministic random subset; selection does not use "
        "model predictions, labels, or evaluation scores."
    )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(
            output_manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"selected {len(selected)}/{len(rows)} rows into {output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
