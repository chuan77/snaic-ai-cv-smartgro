import pytest

from src.data.gallery import get_base_price


@pytest.mark.parametrize("class_str,expected_price", [
    ("Packages/Milk/Arla-Standard-Milk", 3.20),
    ("Packages/Juice/Bravo-Apple-Juice", 3.20),
    ("Ready-To-Eat/Instant-Noodles", 3.00),
    ("Snacks/Chocolate-Bar", 6.20),
    ("Fruit/Apple/Royal-Gala", 1.75),
    ("Vegetables/Carrots", 1.75),
])
def test_get_base_price_maps_top_level_category_to_expected_price(class_str, expected_price):
    assert get_base_price(class_str) == expected_price
