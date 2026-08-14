import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.data.build_context_grounding_benchmarks import build_pair
from scripts.data.harden_residual_mobile_grounding import (
    build_context_prompt,
    load_hierarchy,
    normalize_ui_label,
    select_hard_negative,
)


class ResidualMobileGroundingHardeningTest(unittest.TestCase):
    def test_prefers_same_screen_similar_clickable_negative(self):
        elements = [
            {
                "text": "Settings",
                "class_name": "android.widget.TextView",
                "package_name": "com.android.settings",
                "is_visible": True,
                "is_enabled": True,
                "is_clickable": True,
                "bbox_pixels": {
                    "x_min": 80,
                    "y_min": 200,
                    "x_max": 360,
                    "y_max": 420,
                },
            },
            {
                "content_description": "Settings",
                "class_name": "android.widget.ImageButton",
                "package_name": "io.github.moeleak.lladaagent",
                "is_visible": True,
                "is_enabled": True,
                "is_clickable": True,
                "bbox_pixels": {
                    "x_min": 900,
                    "y_min": 2100,
                    "x_max": 1060,
                    "y_max": 2320,
                },
            },
            {
                "text": "Unrelated",
                "class_name": "android.widget.Button",
                "package_name": "example",
                "is_visible": True,
                "is_enabled": True,
                "is_clickable": True,
                "bbox_pixels": {
                    "x_min": 500,
                    "y_min": 900,
                    "x_max": 700,
                    "y_max": 1100,
                },
            },
        ]

        selected = select_hard_negative(
            elements=elements,
            target="Settings application",
            target_bbox=[74, 83, 333, 175],
            target_role="app icon",
            width=1080,
            height=2400,
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected["label"], "Settings")
        self.assertEqual(
            selected["package"], "io.github.moeleak.lladaagent"
        )

    def test_excludes_target_and_non_actionable_nodes(self):
        elements = [
            {
                "text": "Save",
                "is_visible": True,
                "is_enabled": True,
                "is_clickable": True,
                "bbox_pixels": {
                    "x_min": 100,
                    "y_min": 100,
                    "x_max": 300,
                    "y_max": 300,
                },
            },
            {
                "text": "Save",
                "is_visible": True,
                "is_enabled": True,
                "is_clickable": False,
                "bbox_pixels": {
                    "x_min": 700,
                    "y_min": 700,
                    "x_max": 900,
                    "y_max": 900,
                },
            },
        ]

        self.assertIsNone(
            select_hard_negative(
                elements=elements,
                target="Save",
                target_bbox=[100, 100, 300, 300],
                target_role="button",
                width=1000,
                height=1000,
            )
        )

    def test_context_prompt_redacts_typed_text_and_marks_hint_untrusted(self):
        prompt = build_context_prompt(
            task="Open Settings and select Sound & vibration",
            task_app="Settings",
            task_package="com.android.settings",
            packages=["io.github.moeleak.lladaagent"],
            history=[
                {"action": "type", "text": "secret"},
                {"action": "click", "target": "Continue"},
            ],
            target_hint="gear icon in the bottom right",
        )

        self.assertIn("Planner target hint (may be imprecise)", prompt)
        self.assertIn("type <redacted text>", prompt)
        self.assertNotIn("secret", prompt)
        self.assertIn("com.android.settings", prompt)

    def test_context_prompt_omits_stale_task_app_metadata(self):
        prompt = build_context_prompt(
            task="Create a new alarm in the Clock app for 8:15 AM",
            task_app="audio recorder",
            task_package="com.dimowner.audiorecorder",
            packages=["com.google.android.deskclock", "com.android.systemui"],
            history=[],
            target_hint="OK",
        )

        self.assertNotIn("audio recorder", prompt)
        self.assertNotIn("com.dimowner.audiorecorder", prompt)
        self.assertIn("com.google.android.deskclock", prompt)

    def test_resource_name_becomes_human_readable_label(self):
        self.assertEqual(
            normalize_ui_label({"resource_name": "pkg:id/sound_settings_button"}),
            "sound settings button",
        )

    def test_missing_hierarchy_falls_back_without_inventing_candidates(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self.assertIsNone(load_hierarchy(root, "missing", 3, {}))

    def test_held_out_pair_changes_only_context_hint(self):
        sample = {
            "benchmark": "mobile_test",
            "sample_id": "mobile:settings:0",
            "source_sample_id": "settings:0",
            "prompt": "Click on Settings application.",
            "native_prompt": "Click on Settings application.",
            "target_bbox_1000": [74, 83, 333, 175],
            "image": "images/example.png",
        }
        planner = {
            "id": "settings:0",
            "task": "Open Android Settings and choose Sound & vibration",
            "app": "settings",
            "app_package": "com.android.settings",
            "history": [{"action": "open", "app": "LLaDA-Agent"}],
            "planner_action": {"action": "click", "target": "Settings application"},
            "ground_truth": {"target_label": {"source_role": "app icon"}},
        }
        hierarchy = {
            "logical_screen_size": [1080, 2400],
            "ui_elements": [
                {
                    "text": "Settings",
                    "package_name": "com.android.settings",
                    "is_visible": True,
                    "is_enabled": True,
                    "is_clickable": True,
                    "bbox_pixels": {
                        "x_min": 80,
                        "y_min": 200,
                        "x_max": 360,
                        "y_max": 420,
                    },
                },
                {
                    "content_description": "Settings",
                    "package_name": "io.github.moeleak.lladaagent",
                    "is_visible": True,
                    "is_enabled": True,
                    "is_clickable": True,
                    "bbox_pixels": {
                        "x_min": 900,
                        "y_min": 2100,
                        "x_max": 1060,
                        "y_max": 2320,
                    },
                },
            ],
        }

        pair = build_pair(
            sample=sample,
            planner=planner,
            hierarchy=hierarchy,
            clean_benchmark="mobile_test_context_clean",
            hard_benchmark="mobile_test_context_hard_hint",
        )

        self.assertIsNotNone(pair)
        clean, hard = pair
        self.assertEqual(clean["sample_id"], hard["sample_id"])
        self.assertEqual(clean["target_bbox_1000"], hard["target_bbox_1000"])
        self.assertFalse(clean["hint_is_hard_negative"])
        self.assertTrue(hard["hint_is_hard_negative"])
        self.assertIn("Settings application", clean["prompt"])
        self.assertIn("Settings in the bottom right", hard["prompt"])
        self.assertIn("Open Android Settings", hard["prompt"])
        self.assertNotEqual(clean["prompt"], hard["prompt"])


if __name__ == "__main__":
    unittest.main()
