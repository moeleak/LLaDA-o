import tempfile
import unittest
from pathlib import Path

from PIL import Image

from eval.gui_grounding.compare_long_context import validate_true_long_rope
from eval.gui_grounding.score_benchmark import context_bucket, runtime_metrics
from scripts.data.prepare_gui_grounding_benchmarks import (
    full_page_prompt,
    full_page_tile_layout,
    write_source_image_once,
)


class FullPageContextTest(unittest.TestCase):
    def test_layout_covers_source_without_overlap_or_resize(self) -> None:
        layout = full_page_tile_layout(
            1318,
            5283,
            tile_size=980,
            patch_size=14,
        )
        covered_area = 0
        for tile in layout:
            left, top, right, bottom = tile["box_xyxy"]
            self.assertLessEqual(right - left, 980)
            self.assertLessEqual(bottom - top, 980)
            self.assertEqual(
                tile["patch_tokens"],
                tile["grid_width"] * tile["grid_height"],
            )
            covered_area += (right - left) * (bottom - top)
        self.assertEqual(covered_area, 1318 * 5283)
        for index, left_tile in enumerate(layout):
            left_box = left_tile["box_xyxy"]
            for right_tile in layout[index + 1 :]:
                right_box = right_tile["box_xyxy"]
                overlap_width = max(
                    0,
                    min(left_box[2], right_box[2])
                    - max(left_box[0], right_box[0]),
                )
                overlap_height = max(
                    0,
                    min(left_box[3], right_box[3])
                    - max(left_box[1], right_box[1]),
                )
                self.assertEqual(overlap_width * overlap_height, 0)

    def test_prompt_declares_global_coordinate_frame(self) -> None:
        prompt = full_page_prompt(
            "Click on Quick Tools.",
            width=1318,
            height=5283,
            tile_count=12,
        )
        self.assertIn("1318x5283", prompt)
        self.assertIn("12 images", prompt)
        self.assertIn("complete original screenshot", prompt)

    def test_source_image_bytes_are_not_reencoded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            Image.new("RGB", (3, 5), (1, 2, 3)).save(source)
            raw = source.read_bytes()
            relative = write_source_image_once(root, "raw", raw)
            self.assertEqual((root / relative).read_bytes(), raw)

    def test_context_buckets_use_exact_total_tokens(self) -> None:
        self.assertEqual(
            context_bucket({"sequence_tokens": {"total": 20_000}}),
            "16k_32k",
        )
        self.assertEqual(
            context_bucket({"sequence_tokens": {"total": 40_000}}),
            "32k_48k",
        )
        self.assertEqual(
            context_bucket({"sequence_tokens": {"total": 60_000}}),
            "48k_64k",
        )

    def test_runtime_metrics_include_throughput_and_errors(self) -> None:
        metrics = runtime_metrics(
            [
                {
                    "model_elapsed_seconds": 2.0,
                    "sequence_tokens": {"total": 20_000},
                    "max_prefill_position": 19_935,
                    "max_generation_position": 19_999,
                    "error": None,
                },
                {
                    "model_elapsed_seconds": None,
                    "sequence_tokens": {"total": 30_000},
                    "max_prefill_position": 29_935,
                    "max_generation_position": 29_999,
                    "error": "failed",
                },
            ]
        )
        self.assertEqual(metrics["total_tokens_per_second"]["mean"], 10_000)
        self.assertEqual(metrics["max_prefill_position"]["max"], 29_935)
        self.assertEqual(metrics["max_generation_position"]["max"], 29_999)
        self.assertEqual(metrics["errors"], 1)

    def test_true_long_rope_validation_requires_dense_sequential_prefix(self) -> None:
        rows = [
            {
                "position_mode": "sequential",
                "max_generation_position": 20_000,
                "dense_prefix_tokens": 19_937,
                "cached_prefix_tokens": 19_937,
            }
        ]
        validation = validate_true_long_rope(
            "yarn",
            rows,
            original_max_position=16_384,
        )
        self.assertEqual(validation["min_generation_position"], 20_000)
        self.assertEqual(validation["compressed_prefixes"], 0)

        with self.assertRaisesRegex(RuntimeError, "compressed KV"):
            validate_true_long_rope(
                "yarn",
                [{**rows[0], "cached_prefix_tokens": 15_000}],
                original_max_position=16_384,
            )

        with self.assertRaisesRegex(RuntimeError, "sequential-position"):
            validate_true_long_rope(
                "unscaled",
                [{**rows[0], "position_mode": "native"}],
                original_max_position=16_384,
            )


if __name__ == "__main__":
    unittest.main()
