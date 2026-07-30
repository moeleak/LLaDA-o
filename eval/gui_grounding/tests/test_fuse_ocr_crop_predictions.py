import unittest

from eval.gui_grounding.fuse_ocr_crop_predictions import (
    crop_bbox_to_source,
    prefer_crop_model,
    use_crop_prediction,
)


class FuseOcrCropPredictionsTest(unittest.TestCase):
    def test_maps_crop_bbox_back_to_full_page(self) -> None:
        self.assertEqual(
            crop_bbox_to_source(
                (100, 200, 600, 800),
                crop_box=(400, 800, 1_400, 1_800),
                source_width=2_000,
                source_height=4_000,
            ),
            [250, 250, 500, 400],
        )

    def test_crop_model_is_limited_to_clickable_control_labels(self) -> None:
        self.assertTrue(prefer_crop_model("lclick", "Search By Breed"))
        self.assertTrue(prefer_crop_model("lclick", "Select Location"))
        self.assertFalse(prefer_crop_model("lclick", "News"))
        self.assertFalse(prefer_crop_model("type_in", "Search By Breed"))

    def test_explicit_fusion_policies(self) -> None:
        self.assertTrue(use_crop_prediction("crop", "hover", "News"))
        self.assertFalse(use_crop_prediction("ocr", "lclick", "Search By Breed"))
        self.assertTrue(
            use_crop_prediction("selective", "lclick", "Search By Breed")
        )
        with self.assertRaisesRegex(ValueError, "unsupported fusion policy"):
            use_crop_prediction("unknown", "lclick", "News")


if __name__ == "__main__":
    unittest.main()
