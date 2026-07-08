import pytest

from src.data.gallery import get_base_price, get_item_price


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


def test_get_item_price_varies_within_same_category():
    assert get_item_price("Snacks/Chocolate-Bar") == 5.99
    assert get_item_price("Snacks/Cookies") == 6.49


def test_get_item_price_is_deterministic_across_calls():
    first = get_item_price("Packages/Milk/Arla-Standard-Milk")
    second = get_item_price("Packages/Milk/Arla-Standard-Milk")
    assert first == second == 3.49


@pytest.mark.parametrize("class_str,expected_price", [
    ("Snacks/Chocolate-Bar", 5.99),
    ("Snacks/Cookies", 6.49),
    ("Packages/Milk/Arla-Standard-Milk", 3.49),
    ("Ready-To-Eat/Instant-Noodles", 3.49),
    ("Fruit/Apple/Royal-Gala", 1.99),
    ("Vegetables/Carrots", 1.99),
])
def test_get_item_price_ends_in_charm_pricing(class_str, expected_price):
    price = get_item_price(class_str)
    assert price == expected_price
    cents = round((price % 1) * 100)
    assert cents in (49, 99)


def test_get_item_price_stays_near_category_base():
    for class_str in ("Snacks/Chocolate-Bar", "Snacks/Cookies"):
        base = get_base_price(class_str)
        price = get_item_price(class_str)
        assert base * 0.75 <= price <= base * 1.25
