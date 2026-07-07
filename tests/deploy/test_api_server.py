import pytest
from fastapi.testclient import TestClient

from src.deploy.api_server import app, get_catalog, leaf_display_name, load_catalog, xyxy_to_fractional_xywh


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


def test_load_catalog_reads_csv_and_derives_display_name(tmp_path):
    csv_path = tmp_path / "catalog_prices.csv"
    csv_path.write_text(
        "class_id,product_name,price_usd\n"
        "0,Fruit/Apple/Royal-Gala,1.75\n"
        "1,Snacks/Chocolate-Bar,6.2\n"
    )

    catalog = load_catalog(csv_path)

    assert catalog == [
        {"sku": "Fruit/Apple/Royal-Gala", "name": "Royal Gala", "priceUsd": 1.75},
        {"sku": "Snacks/Chocolate-Bar", "name": "Chocolate Bar", "priceUsd": 6.2},
    ]


def test_catalog_endpoint_returns_the_overridden_catalog():
    app.dependency_overrides[get_catalog] = lambda: [
        {"sku": "Fruit/Apple/Royal-Gala", "name": "Royal Gala", "priceUsd": 1.75}
    ]
    client = TestClient(app)

    response = client.get("/catalog")

    assert response.status_code == 200
    assert response.json() == [{"sku": "Fruit/Apple/Royal-Gala", "name": "Royal Gala", "priceUsd": 1.75}]
    app.dependency_overrides.clear()
