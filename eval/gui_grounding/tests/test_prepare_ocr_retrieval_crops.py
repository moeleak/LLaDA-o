import unittest

from eval.gui_grounding.prepare_ocr_retrieval_crops import (
    bbox_to_crop_coordinates,
    retrieval_crop_box,
)


class PrepareOcrRetrievalCropsTest(unittest.TestCase):
    def test_crop_stays_inside_source_and_keeps_more_context_below(self) -> None:
        self.assertEqual(
            retrieval_crop_box(
                (1_000, 4_000, 1_100, 4_040),
                image_width=1_318,
                image_height=5_283,
                crop_size=980,
                target_anchor_y=0.35,
            ),
            (338, 3677, 1318, 4657),
        )

    def test_target_coordinates_transform_without_clamping(self) -> None:
        self.assertEqual(
            bbox_to_crop_coordinates(
                (250, 250, 500, 500),
                source_width=2_000,
                source_height=4_000,
                crop_box=(400, 800, 1_400, 1_800),
            ),
            [100, 200, 600, 1200],
        )


if __name__ == "__main__":
    unittest.main()
