import unittest

from scripts.data.harden_residual_mobile_grounding import (
    build_context_prompt,
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

    def test_resource_name_becomes_human_readable_label(self):
        self.assertEqual(
            normalize_ui_label({"resource_name": "pkg:id/sound_settings_button"}),
            "sound settings button",
        )


if __name__ == "__main__":
    unittest.main()
