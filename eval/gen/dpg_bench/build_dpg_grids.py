#!/usr/bin/env python3

# Copyright 2025 AntGroup and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

RESAMPLE_BICUBIC = getattr(Image, "Resampling", Image).BICUBIC


def parse_args():
    parser = argparse.ArgumentParser(description="Build 2x2 grid images for DPG-Bench scoring.")
    parser.add_argument("--input-root", type=str, required=True, help="Folder containing per-prompt subfolders.")
    parser.add_argument("--output-root", type=str, required=True, help="Folder to save grid images.")
    parser.add_argument("--resolution", type=int, default=1024, help="Single-image resolution in the grid.")
    parser.add_argument("--pic-num", type=int, default=4, help="How many samples to use from each prompt folder.")
    parser.add_argument("--num-workers", type=int, default=min(os.cpu_count() or 1, 8))
    parser.add_argument("--output-format", type=str, default="png", choices=["png", "jpg", "jpeg"])
    parser.add_argument("--png-compress-level", type=int, default=1, help="Lower is faster and larger.")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true", help="Rebuild grids even if output files already exist.")
    return parser.parse_args()


def find_sample_images(folder: Path):
    sample_dir = folder / "samples"
    if not sample_dir.is_dir():
        return []

    sample_images = []
    for path in sorted(sample_dir.iterdir()):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            continue
        sample_images.append(path)
    return sample_images


def load_tile(sample_path, resolution):
    target_size = (resolution, resolution)
    with Image.open(sample_path) as image:
        image = image.convert("RGB")
        if image.size != target_size:
            image = image.resize(target_size, RESAMPLE_BICUBIC)
        else:
            image = image.copy()
    return image


def build_grid(sample_paths, resolution):
    if len(sample_paths) == 1:
        return load_tile(sample_paths[0], resolution)

    canvas = Image.new("RGB", (resolution * 2, resolution * 2), color=(255, 255, 255))
    positions = [
        (0, 0),
        (resolution, 0),
        (0, resolution),
        (resolution, resolution),
    ]

    for index, sample_path in enumerate(sample_paths[:4]):
        image = load_tile(sample_path, resolution)
        canvas.paste(image, positions[index])
    return canvas


def output_suffix(output_format):
    return ".jpg" if output_format in {"jpg", "jpeg"} else ".png"


def save_grid(grid, save_path, output_format, png_compress_level, jpeg_quality):
    if output_format == "png":
        grid.save(save_path, compress_level=png_compress_level)
    else:
        grid.save(save_path, quality=jpeg_quality, optimize=False)


def build_one(task):
    prompt_dir, output_root, resolution, pic_num, output_format, png_compress_level, jpeg_quality, overwrite = task
    sample_images = find_sample_images(prompt_dir)
    if len(sample_images) < min(pic_num, 1):
        return "skipped", prompt_dir.name

    chosen = sample_images[:pic_num]
    if not chosen:
        return "skipped", prompt_dir.name

    save_path = output_root / f"{prompt_dir.name}{output_suffix(output_format)}"
    if save_path.exists() and not overwrite:
        return "exists", prompt_dir.name

    grid = build_grid(chosen, resolution)
    save_grid(grid, save_path, output_format, png_compress_level, jpeg_quality)
    return "built", prompt_dir.name


def main():
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    prompt_dirs = sorted(path for path in input_root.iterdir() if path.is_dir())
    if not prompt_dirs:
        raise FileNotFoundError(f"No prompt subfolders found under {input_root}")

    tasks = [
        (
            prompt_dir,
            output_root,
            args.resolution,
            args.pic_num,
            args.output_format,
            args.png_compress_level,
            args.jpeg_quality,
            args.overwrite,
        )
        for prompt_dir in prompt_dirs
    ]

    built = 0
    existing = 0
    skipped = 0
    workers = max(args.num_workers, 1)

    if workers == 1:
        for task in tasks:
            status, _ = build_one(task)
            if status == "built":
                built += 1
            elif status == "exists":
                existing += 1
            else:
                skipped += 1
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(build_one, task) for task in tasks]
            for future in as_completed(futures):
                status, _ = future.result()
                if status == "built":
                    built += 1
                elif status == "exists":
                    existing += 1
                else:
                    skipped += 1

    print(f"Built {built} DPG grid images into: {output_root}")
    if existing:
        print(f"Skipped {existing} existing grid images.")
    if skipped:
        print(f"Skipped {skipped} prompt folders due to missing samples.")


if __name__ == "__main__":
    main()
