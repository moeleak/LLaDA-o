import math
import unittest

from eval.gui_grounding.metrics import parse_action, point_in_box, score_records


class ParseActionTest(unittest.TestCase):
    def test_parses_action_box_and_type_value(self):
        parsed = parse_action(
            "<|startoftext|>type_in [10, 20, 110, 220] hello world<|endoftext|>"
        )

        self.assertTrue(parsed.valid)
        self.assertEqual(parsed.action, "type_in")
        self.assertEqual(parsed.bbox_1000, (10.0, 20.0, 110.0, 220.0))
        self.assertEqual(parsed.value, "hello world")

    def test_rejects_degenerate_and_out_of_range_boxes(self):
        self.assertEqual(
            parse_action("lclick [10,20,5,30]").error,
            "degenerate_bbox",
        )
        self.assertEqual(
            parse_action("hover [0,0,1001,30]").error,
            "bbox_out_of_range",
        )

    def test_rejects_unstructured_text(self):
        parsed = parse_action("click the button near the top right")

        self.assertFalse(parsed.valid)
        self.assertEqual(parsed.error, "action_or_bbox_not_found")

    def test_point_in_box_boundary_policy_is_explicit(self):
        self.assertTrue(point_in_box((0, 10), (0, 0, 20, 20)))
        self.assertFalse(point_in_box((0, 10), (0, 0, 20, 20), inclusive=False))


class ScoreRecordsTest(unittest.TestCase):
    def test_reports_point_only_and_joint_success_separately(self):
        metrics = score_records(
            [
                {
                    "target_action": "lclick",
                    "target_bbox_1000": [0, 0, 100, 100],
                    "prediction": "lclick [10,10,20,20]",
                    "latency_seconds": 1.0,
                    "convergence_steps": 10,
                },
                {
                    "target_action": "hover",
                    "target_bbox_1000": [200, 200, 400, 400],
                    "prediction": "lclick [250,250,300,300]",
                    "latency_seconds": 3.0,
                    "convergence_steps": 20,
                },
                {
                    "target_action": "type_in",
                    "target_bbox_1000": [0, 0, 100, 100],
                    "prediction": "not an action",
                },
            ]
        )

        self.assertEqual(metrics["num_samples"], 3)
        self.assertTrue(math.isclose(metrics["ssr_point_only"], 2 / 3))
        self.assertTrue(math.isclose(metrics["joint_step_success"], 1 / 3))
        self.assertTrue(math.isclose(metrics["action_accuracy"], 1 / 3))
        self.assertTrue(math.isclose(metrics["action_f1_macro_all"], 2 / 9))
        self.assertEqual(metrics["parse_errors"], {"action_or_bbox_not_found": 1})
        self.assertEqual(metrics["latency_seconds"]["mean"], 2.0)
        self.assertEqual(metrics["convergence_steps"]["mean"], 15.0)

    def test_click_only_macro_variants_are_explicit(self):
        metrics = score_records(
            [
                {
                    "target_action": "lclick",
                    "target_bbox_1000": [0, 0, 100, 100],
                    "prediction": "lclick [10,10,20,20]",
                },
                {
                    "target_action": "lclick",
                    "target_bbox_1000": [0, 0, 100, 100],
                    "prediction": "hover [10,10,20,20]",
                },
            ]
        )

        self.assertTrue(math.isclose(metrics["action_f1_macro_present"], 2 / 3))
        self.assertTrue(math.isclose(metrics["action_f1_macro_all"], 2 / 9))


if __name__ == "__main__":
    unittest.main()
