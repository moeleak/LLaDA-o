from __future__ import annotations

from scripts.data.realign_gui_grounding_ocr import read_spatially_relevant_text


class FakeReader:
    def __init__(self) -> None:
        self.recognized_horizontal = None
        self.recognized_free = None

    def detect(self, image, **kwargs):
        del image, kwargs
        return (
            [[
                [110, 180, 110, 140],
                [900, 980, 900, 940],
            ]],
            [[
                [[130, 130], [190, 130], [190, 160], [130, 160]],
                [[850, 850], [910, 850], [910, 900], [850, 900]],
            ]],
        )

    def recognize(self, image, *, horizontal_list, free_list, **kwargs):
        del image, kwargs
        self.recognized_horizontal = horizontal_list
        self.recognized_free = free_list
        return [[[[110, 110], [180, 110], [180, 140], [110, 140]], "near", 0.9]]


def test_spatial_prefilter_removes_only_impossible_candidates() -> None:
    reader = FakeReader()
    result = read_spatially_relevant_text(
        reader,
        object(),
        source_bbox_xyxy=(100, 100, 120, 120),
        image_width=1000,
        image_height=1000,
    )

    assert result[0][1] == "near"
    assert reader.recognized_horizontal == [[110, 180, 110, 140]]
    assert reader.recognized_free == [
        [[130, 130], [190, 130], [190, 160], [130, 160]]
    ]
