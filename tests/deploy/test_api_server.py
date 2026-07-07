import pytest

from src.deploy.api_server import leaf_display_name, xyxy_to_fractional_xywh


@pytest.mark.parametrize("product_name,expected", [
    ("Fruit/Apple/Royal-Gala", "Royal Gala"),
    ("Vegetables/Carrots", "Carrots"),
    ("Ready-To-Eat/Instant-Noodles", "Instant Noodles"),
    ("Snacks/Chocolate-Bar", "Chocolate Bar"),
])
def test_leaf_display_name_derives_human_readable_label(product_name, expected):
    assert leaf_display_name(product_name) == expected


def test_xyxy_to_fractional_xywh_converts_pixel_box_to_fractional_top_left_xywh():
    result = xyxy_to_fractional_xywh([50.0, 100.0, 150.0, 200.0], img_width=200, img_height=400)

    assert result == pytest.approx((0.25, 0.25, 0.5, 0.25))
