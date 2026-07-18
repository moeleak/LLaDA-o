import unittest

from eval.gui_grounding.reproducibility import paired_sample_seed
from scripts.data.gui_grounding_protocol import (
    TARGET_GROUNDING,
    TASK_HISTORY,
    canonical_action,
    mind2web_crop_plan,
    mind2web_prompt,
    parse_target_action,
)


class Mind2WebProtocolTest(unittest.TestCase):
    def test_builds_paper_style_target_explicit_click_prompt(self) -> None:
        target = parse_target_action(
            "[link] Track & Field -> CLICK",
            {"op": "CLICK", "original_op": "CLICK", "value": ""},
        )

        self.assertEqual(target.action, "lclick")
        self.assertEqual(target.role, "link")
        self.assertEqual(target.description, "Track & Field")
        self.assertEqual(
            mind2web_prompt(TARGET_GROUNDING, target),
            "Click on Track & Field.",
        )

    def test_type_and_select_keep_their_values(self) -> None:
        type_target = parse_target_action(
            "[textbox] Search -> TYPE: trail shoes",
            {"op": "TYPE", "value": "trail shoes"},
        )
        select_target = parse_target_action(
            "[combobox] Sort by -> SELECT: Price",
            {"op": "SELECT", "value": "Price"},
        )

        self.assertEqual(type_target.action, "type_in")
        self.assertEqual(
            mind2web_prompt(TARGET_GROUNDING, type_target),
            'Type "trail shoes" into Search.',
        )
        self.assertEqual(select_target.action, "type_in")
        self.assertEqual(
            mind2web_prompt(TARGET_GROUNDING, select_target),
            'Select "Price" from Sort by.',
        )

    def test_structured_operation_is_authoritative(self) -> None:
        target = parse_target_action(
            "[button] Continue -> CLICK",
            {"op": "CLICK", "original_op": "HOVER"},
        )

        self.assertEqual(target.operation, "HOVER")
        self.assertEqual(target.action, "hover")
        self.assertEqual(
            mind2web_prompt(TARGET_GROUNDING, target),
            "Hover over Continue.",
        )

    def test_legacy_task_history_is_an_explicitly_separate_protocol(self) -> None:
        target = parse_target_action("[button] Submit -> CLICK", {"op": "CLICK"})
        prompt = mind2web_prompt(
            TASK_HISTORY,
            target,
            confirmed_task="Buy a book",
            action_reprs=["[link] Books -> CLICK", "[button] Submit -> CLICK"],
            target_action_index=1,
        )

        self.assertEqual(
            prompt,
            "Complete the following web task by predicting the next GUI action.\n"
            "Task: Buy a book\n"
            "Previous actions:\n"
            "- [link] Books -> CLICK",
        )
        self.assertNotIn("Submit", prompt)

    def test_missing_description_uses_candidate_fallback(self) -> None:
        target = parse_target_action(
            "-> CLICK",
            {"op": "CLICK"},
            fallback_description="Checkout",
        )

        self.assertEqual(target.description, "Checkout")
        self.assertEqual(canonical_action("ENTER"), "lclick")

    def test_crop_plan_is_balanced_deterministic_and_unique(self) -> None:
        source_ids = ["a", "b", "c"]
        plan = mind2web_crop_plan(source_ids, target_count=8, seed=42)

        self.assertEqual(plan, mind2web_crop_plan(source_ids[::-1], 8, seed=42))
        self.assertEqual(len(plan), 8)
        self.assertEqual(len(set(plan)), 8)
        variants = {sample_id: [] for sample_id in source_ids}
        for sample_id, variant in plan:
            variants[sample_id].append(variant)
        self.assertEqual(sorted(len(values) for values in variants.values()), [2, 3, 3])
        self.assertTrue(
            all(values == list(range(len(values))) for values in variants.values())
        )

    def test_crop_plan_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            mind2web_crop_plan([], 1, seed=42)
        with self.assertRaises(ValueError):
            mind2web_crop_plan(["duplicate", "duplicate"], 2, seed=42)

    def test_paired_protocols_receive_same_order_independent_seed(self) -> None:
        target_prompt_sample = {
            "sample_id": "mind2web:test:target",
            "provenance": {"action_uid": "shared-action"},
        }
        history_prompt_sample = {
            "sample_id": "mind2web_task_history:test:history",
            "provenance": {"action_uid": "shared-action"},
        }

        self.assertEqual(
            paired_sample_seed(target_prompt_sample, 42),
            paired_sample_seed(history_prompt_sample, 42),
        )
        self.assertNotEqual(
            paired_sample_seed(target_prompt_sample, 42),
            paired_sample_seed(target_prompt_sample, 43),
        )


if __name__ == "__main__":
    unittest.main()
