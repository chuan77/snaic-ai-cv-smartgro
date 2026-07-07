import io
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.deploy.api_server import (
    app,
    build_detections,
    get_catalog,
    get_catalog_path,
    get_cors_origins,
    get_detector,
    leaf_display_name,
    load_catalog,
    xyxy_to_fractional_xywh,
)


@pytest.mark.parametrize("product_name,expected", [
    ("Fruit/Apple/Royal-Gala", "Royal Gala"),
    ("Vegetables/Carrots", "Carrots"),
    ("Ready-To-Eat/Instant-Noodles", "Instant Noodles"),
    ("Snacks/Chocolate-Bar", "Chocolate Bar"),
])
def test_leaf_display_name_derives_human_readable_label(product_name, expected):
    assert leaf_display_name(product_name) == expected


def test_get_cors_origins_defaults_to_localhost_5173(monkeypatch):
    monkeypatch.delenv("SMARTCART_CORS_ORIGINS", raising=False)

    assert get_cors_origins() == ["http://localhost:5173"]


def test_get_cors_origins_splits_comma_separated_values_and_strips_whitespace(monkeypatch):
    monkeypatch.setenv("SMARTCART_CORS_ORIGINS", "http://localhost:5173, http://127.0.0.1:5173 ,http://example.com")

    assert get_cors_origins() == [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://example.com",
    ]


def test_get_catalog_path_defaults_to_artifacts_catalog_prices_csv(monkeypatch):
    monkeypatch.delenv("SMARTCART_CATALOG_PATH", raising=False)

    assert get_catalog_path() == Path("./artifacts/catalog_prices.csv")


def test_get_catalog_path_reads_from_environment(monkeypatch, tmp_path):
    custom_path = tmp_path / "custom_catalog.csv"
    monkeypatch.setenv("SMARTCART_CATALOG_PATH", str(custom_path))

    assert get_catalog_path() == custom_path


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


def test_build_detections_converts_raw_detections_to_response_shape():
    raw = [{"class_name": "Snacks/Chocolate-Bar", "confidence": 0.9, "bbox": [50.0, 100.0, 150.0, 200.0]}]

    result = build_detections(raw, img_width=200, img_height=400)

    assert len(result) == 1
    assert result[0]["label"] == "Snacks/Chocolate-Bar"
    assert result[0]["confidence"] == 0.9
    assert result[0]["bbox"] == pytest.approx([0.25, 0.25, 0.5, 0.25])
    assert isinstance(result[0]["id"], str) and result[0]["id"]


def test_build_detections_returns_empty_list_for_no_detections():
    assert build_detections([], img_width=200, img_height=400) == []


def test_predict_endpoint_returns_converted_detections_from_mocked_detector():
    mock_detector = MagicMock()
    mock_detector.detect.return_value = [
        {"class_name": "Snacks/Chocolate-Bar", "confidence": 0.9, "bbox": [50.0, 100.0, 150.0, 200.0]}
    ]
    app.dependency_overrides[get_detector] = lambda: mock_detector
    client = TestClient(app)

    image_bytes = io.BytesIO()
    Image.new("RGB", (200, 400)).save(image_bytes, format="JPEG")
    image_bytes.seek(0)

    response = client.post("/predict", files={"file": ("test.jpg", image_bytes, "image/jpeg")})

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["label"] == "Snacks/Chocolate-Bar"
    assert body[0]["bbox"] == pytest.approx([0.25, 0.25, 0.5, 0.25])
    app.dependency_overrides.clear()


def test_predict_endpoint_returns_empty_list_when_no_detections():
    mock_detector = MagicMock()
    mock_detector.detect.return_value = []
    app.dependency_overrides[get_detector] = lambda: mock_detector
    client = TestClient(app)

    image_bytes = io.BytesIO()
    Image.new("RGB", (100, 100)).save(image_bytes, format="JPEG")
    image_bytes.seek(0)

    response = client.post("/predict", files={"file": ("test.jpg", image_bytes, "image/jpeg")})

    assert response.status_code == 200
    assert response.json() == []
    app.dependency_overrides.clear()
