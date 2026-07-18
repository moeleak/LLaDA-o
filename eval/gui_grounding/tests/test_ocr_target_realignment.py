import math
import argparse
import json
import tempfile
import unittest
from pathlib import Path

from scripts.data.ocr_target_realignment import (
    OcrDetection,
    match_ocr_target,
    normalize_ocr_text,
    replace_action_bbox,
    scale_bbox,
    text_similarity,
    unscale_bbox,
)
from scripts.data.realign_gui_grounding_ocr import finalize


class OcrTextMatchingTest(unittest.TestCase):
    def test_normalization_and_similarity_handle_visible_punctuation(self) -> None:
        self.assertEqual(normalize_ocr_text("ZIP Code™"), "zip codetm")
        self.assertEqual(text_similarity("Track & Field", "TRACK & FIELD"), 1.0)
        self.assertGreater(text_similarity("Downloads folder", "Downloads"), 0.85)
        self.assertLess(text_similarity("Downloads", "Documents"), 0.68)

    def test_links_exact_nearby_text_without_using_predictions(self) -> None:
        detections = [
            OcrDetection("Documents", 0.99, (400, 600, 520, 630)),
            OcrDetection("Downloads", 0.97, (650, 600, 790, 630)),
            OcrDetection("Downloads", 0.70, (20, 20, 160, 50)),
        ]
        match = match_ocr_target(
            target_text="Downloads",
            source_bbox_xyxy=(630, 480, 810, 650),
            detections=detections,
            image_width=1000,
            image_height=1000,
        )

        self.assertTrue(match.accepted)
        self.assertEqual(match.matched_text, "Downloads")
        self.assertEqual(match.candidate_index, 1)
        self.assertEqual(match.bbox_xyxy, (650, 600, 790, 630))

    def test_rejects_fuzzy_text_on_the_other_side_of_screen(self) -> None:
        match = match_ocr_target(
            target_text="Checkout",
            source_bbox_xyxy=(800, 800, 900, 900),
            detections=[OcrDetection("Check out", 0.99, (10, 10, 100, 40))],
            image_width=1000,
            image_height=1000,
        )

        self.assertFalse(match.accepted)
        self.assertEqual(match.reason, "no_credible_nearby_text")

    def test_bbox_scaling_round_trip(self) -> None:
        normalized = scale_bbox((20, 10, 100, 60), width=200, height=100)
        self.assertEqual(normalized, [100, 100, 500, 600])
        restored = unscale_bbox(normalized, width=200, height=100)
        self.assertTrue(all(math.isclose(a, b) for a, b in zip(restored, (20, 10, 100, 60))))

    def test_action_rewrite_preserves_type_value(self) -> None:
        self.assertEqual(
            replace_action_bbox("type_in [1,2,3,4] trail shoes", [10, 20, 30, 40]),
            "type_in [10,20,30,40] trail shoes",
        )
        with self.assertRaises(ValueError):
            replace_action_bbox("click the button", [10, 20, 30, 40])


class OcrBenchmarkFinalizeTest(unittest.TestCase):
    def test_rewrites_paired_protocols_from_prediction_independent_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            work = root / "work"
            output = root / "output"
            (source / "images").mkdir(parents=True)
            (source / "samples").mkdir()
            (work / "detections").mkdir(parents=True)

            manifest = {
                "benchmarks": {
                    "mind2web": {"path": "samples/mind2web.jsonl", "rows": 1},
                    "mind2web_task_history": {
                        "path": "samples/mind2web_task_history.jsonl",
                        "rows": 1,
                    },
                },
                "protocol_notes": [],
            }
            (source / "manifest.json").write_text(json.dumps(manifest))
            common = {
                "image": "images/example.jpg",
                "image_width": 100,
                "image_height": 100,
                "split": "test",
                "target_action": "lclick",
                "target_bbox_1000": [100, 100, 300, 300],
                "provenance": {
                    "action_uid": "action-1",
                    "target_description": "Downloads",
                },
            }
            target = {
                **common,
                "sample_id": "mind2web:test:action-1",
                "benchmark": "mind2web",
                "prompt": "Click on Downloads.",
            }
            history = {
                **common,
                "sample_id": "mind2web_task_history:test:action-1",
                "benchmark": "mind2web_task_history",
                "prompt": "Complete a task.",
            }
            (source / "samples/mind2web.jsonl").write_text(json.dumps(target) + "\n")
            (source / "samples/mind2web_task_history.jsonl").write_text(
                json.dumps(history) + "\n"
            )
            detection = {
                "sample_id": target["sample_id"],
                "action_uid": "action-1",
                "target_bbox_ocr_1000": [120, 220, 280, 260],
                "match": {
                    "accepted": True,
                    "reason": "matched_nearby_ocr_text",
                    "matched_text": "Downloads",
                    "text_similarity": 1.0,
                    "ocr_confidence": 0.99,
                    "edge_distance_normalized": 0.0,
                    "source_iou": 0.2,
                },
                "error": None,
            }
            (work / "detections/part-00000.jsonl").write_text(
                json.dumps(detection) + "\n"
            )

            finalize(
                argparse.Namespace(
                    benchmark_root=source,
                    output_root=output,
                    work_dir=work,
                    benchmark="mind2web",
                    force=False,
                )
            )

            for name in ("mind2web", "mind2web_task_history"):
                row = json.loads((output / f"samples/{name}.jsonl").read_text())
                self.assertEqual(row["target_bbox_1000"], [120, 220, 280, 260])
                self.assertEqual(row["target_bbox_dom_1000"], [100, 100, 300, 300])
                self.assertTrue(row["provenance"]["ocr_realignment"]["prediction_independent"])
            self.assertTrue((output / "images").is_symlink())


if __name__ == "__main__":
    unittest.main()
