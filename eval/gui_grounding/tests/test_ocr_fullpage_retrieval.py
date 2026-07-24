import unittest

from eval.gui_grounding.ocr_fullpage_retrieval import (
    globalize_detection,
    instruction_target,
    select_text_match,
)
from scripts.data.ocr_target_realignment import OcrDetection


class OcrFullpageRetrievalTest(unittest.TestCase):
    def test_instruction_target_uses_only_visible_prompt(self) -> None:
        wrapper = (
            "The following 12 images are non-overlapping tiles from one "
            "1318x5283 webpage screenshot, ordered left-to-right and then "
            "top-to-bottom. Treat them as one complete page. "
        )
        suffix = (
            " Return the action and bounding box with coordinates normalized "
            "to the complete original screenshot in [0,1000]."
        )
        self.assertEqual(
            instruction_target(wrapper + "Click on Quick Tools." + suffix),
            ("lclick", "Quick Tools", ""),
        )
        self.assertEqual(
            instruction_target(
                wrapper
                + 'Type "60505" into *City and State or ZIP Code™.'
                + suffix
            ),
            ("type_in", "*City and State or ZIP Code™", "60505"),
        )

    def test_globalize_detection_adds_tile_offset(self) -> None:
        detection = globalize_detection(
            (
                [[10, 20], [30, 20], [30, 40], [10, 40]],
                "Quick Tools",
                0.9,
            ),
            (980, 1960, 1318, 2940),
        )
        self.assertEqual(
            detection.bbox_xyxy,
            (990.0, 1980.0, 1010.0, 2000.0),
        )

    def test_text_match_prefers_exact_prompt_target(self) -> None:
        match, score = select_text_match(
            "Quick Tools",
            [
                OcrDetection("online tools", 0.99, (1, 1, 10, 10)),
                OcrDetection("Quick Tools", 0.90, (20, 20, 40, 40)),
            ],
        )
        self.assertIsNotNone(match)
        self.assertEqual(match.text, "Quick Tools")
        self.assertGreater(score, 0.9)


if __name__ == "__main__":
    unittest.main()
