import json
import tempfile
import unittest
from pathlib import Path

from eval.gui_grounding.summarize_table3_sweep import (
    parse_required_steps,
    summarize_sweep,
    write_csv,
)


def _metrics(ssr: float, action_f1: float = 0.99) -> dict:
    return {
        "num_samples": 1,
        "ssr_point_only": ssr,
        "joint_step_success": ssr,
        "action_f1_macro_present": action_f1,
        "action_accuracy": action_f1,
        "parse_rate": 1.0,
        "convergence_steps": {"mean": 12.0},
        "latency_seconds": {"mean": 0.8},
    }


class Table3SweepSummaryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.results_root = self.root / "results"
        self.run_dir = self.results_root / "step-0000500" / "fixed"
        scores = self.run_dir / "scores"
        scores.mkdir(parents=True)

        result = {
            "benchmarks": {
                "mind2web": _metrics(0.80),
                "screenspot_web_text": _metrics(0.70),
                "screenspot_web_icon": _metrics(0.50),
            },
            "subgroups": {
                "mind2web": {
                    "test_domain": _metrics(0.82),
                    "test_task": _metrics(0.78),
                    "test_website": _metrics(0.77),
                }
            },
            "coverage": {
                "mind2web": {
                    "targets": 1,
                    "predictions": 1,
                    "joined": 1,
                    "missing": 0,
                }
            },
        }
        (scores / "results.json").write_text(json.dumps(result), encoding="utf-8")

        predictions = self.run_dir / "mind2web"
        predictions.mkdir()
        prediction = {
            "sample_id": "sample-1",
            "predicted_action": "lclick",
            "predicted_bbox_1000": [100, 100, 200, 200],
        }
        (predictions / "part-00000.jsonl").write_text(
            json.dumps(prediction) + "\n", encoding="utf-8"
        )

        self.dom_root = self.root / "dom"
        samples = self.dom_root / "samples"
        samples.mkdir(parents=True)
        target = {
            "sample_id": "sample-1",
            "target_action": "lclick",
            "target_bbox_1000": [120, 120, 180, 180],
            "split": "test_domain",
        }
        (samples / "mind2web.jsonl").write_text(
            json.dumps(target) + "\n", encoding="utf-8"
        )
        manifest = {
            "benchmarks": {"mind2web": {"path": "samples/mind2web.jsonl"}}
        }
        (self.dom_root / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_summarizes_fixed_metrics_and_dom_rescore(self) -> None:
        summary = summarize_sweep(
            self.results_root,
            run_name="fixed",
            dom_benchmark_root=self.dom_root,
            required_steps=(500,),
            primary_step=500,
            steps_per_epoch=100.0,
        )

        row = summary["rows"][0]
        self.assertEqual(summary["completed_steps"], [500])
        self.assertEqual(summary["primary"], row)
        self.assertEqual(row["estimated_epochs"], 5.0)
        self.assertEqual(row["mind2web_ssr_pct"], 80.0)
        self.assertEqual(row["test_domain_ssr_pct"], 82.0)
        self.assertAlmostEqual(row["paper_ssr_gap_pp"], -3.31)
        self.assertAlmostEqual(row["test_domain_paper_ssr_gap_pp"], -1.31)
        self.assertEqual(row["screenspot_web_text_ssr_pct"], 70.0)
        self.assertEqual(row["dom_target_ssr_pct"], 100.0)

        csv_path = self.root / "summary.csv"
        write_csv(csv_path, summary["rows"])
        self.assertIn("mind2web_ssr_pct", csv_path.read_text(encoding="utf-8"))

    def test_rejects_missing_required_or_primary_steps(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "missing required"):
            summarize_sweep(
                self.results_root, run_name="fixed", required_steps=(750,)
            )
        with self.assertRaisesRegex(RuntimeError, "primary step"):
            summarize_sweep(self.results_root, run_name="fixed", primary_step=750)

    def test_parses_required_steps(self) -> None:
        self.assertEqual(parse_required_steps("250, 500,750"), (250, 500, 750))
        with self.assertRaisesRegex(ValueError, "duplicates"):
            parse_required_steps("250,250")


if __name__ == "__main__":
    unittest.main()
