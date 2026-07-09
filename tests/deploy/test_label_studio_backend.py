import io
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from PIL import Image

from src.deploy.api_server import app
from src.deploy.detector import get_detector
from src.deploy.label_studio_backend import (
    build_ls_prediction,
    detections_to_ls_regions,
    fetch_task_image,
    get_label_studio_auth_headers,
    xyxy_to_percentage_xywh,
)


def _png_bytes(width=64, height=32):
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color="red").save(buf, format="PNG")
    return buf.getvalue()


def test_xyxy_to_percentage_xywh_converts_pixel_box_to_percentage():
    box = [10.0, 20.0, 110.0, 70.0]

    assert xyxy_to_percentage_xywh(box, img_width=200, img_height=100) == (5.0, 20.0, 50.0, 50.0)


def test_detections_to_ls_regions_builds_rectanglelabels_shape():
    detections = [{"class_name": "Fruit/Apple/Royal-Gala", "confidence": 0.87, "bbox": [10.0, 20.0, 110.0, 70.0]}]

    regions = detections_to_ls_regions(detections, img_width=200, img_height=100, from_name="label", to_name="image")

    assert len(regions) == 1
    region = regions[0]
    assert region["from_name"] == "label"
    assert region["to_name"] == "image"
    assert region["type"] == "rectanglelabels"
    assert region["original_width"] == 200
    assert region["original_height"] == 100
    assert region["image_rotation"] == 0
    assert region["score"] == 0.87
    assert region["value"] == {
        "x": 5.0,
        "y": 20.0,
        "width": 50.0,
        "height": 50.0,
        "rotation": 0,
        "rectanglelabels": ["Fruit/Apple/Royal-Gala"],
    }
    assert isinstance(region["id"], str) and region["id"]


def test_detections_to_ls_regions_empty_returns_empty_list():
    assert detections_to_ls_regions([], img_width=200, img_height=100, from_name="label", to_name="image") == []


def test_build_ls_prediction_sets_model_version_and_mean_score():
    detections = [
        {"class_name": "Vegetables/Carrots", "confidence": 0.6, "bbox": [0.0, 0.0, 10.0, 10.0]},
        {"class_name": "Vegetables/Carrots", "confidence": 0.8, "bbox": [0.0, 0.0, 10.0, 10.0]},
    ]

    prediction = build_ls_prediction(
        detections, img_width=100, img_height=100, from_name="label", to_name="image", model_version="v1"
    )

    assert prediction["model_version"] == "v1"
    assert prediction["score"] == 0.7
    assert len(prediction["result"]) == 2


def test_build_ls_prediction_zero_detections_score_zero():
    prediction = build_ls_prediction(
        [], img_width=100, img_height=100, from_name="label", to_name="image", model_version="v1"
    )

    assert prediction["score"] == 0.0
    assert prediction["result"] == []


def _mock_token_refresh(monkeypatch, access_token="fresh-access-token"):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {"access": access_token}
        return response

    monkeypatch.setattr("src.deploy.label_studio_backend.httpx.post", fake_post)
    return captured


def test_get_label_studio_auth_headers_exchanges_refresh_token(monkeypatch):
    monkeypatch.setenv("LABEL_STUDIO_URL", "http://ls-host:8080")
    monkeypatch.setenv("LABEL_STUDIO_API_KEY", "secret-refresh-token")
    captured = _mock_token_refresh(monkeypatch)

    headers = get_label_studio_auth_headers()

    assert captured["url"] == "http://ls-host:8080/api/token/refresh"
    assert captured["json"] == {"refresh": "secret-refresh-token"}
    assert headers == {"Authorization": "Bearer fresh-access-token"}


def test_get_label_studio_auth_headers_empty_when_no_key(monkeypatch):
    monkeypatch.setenv("LABEL_STUDIO_API_KEY", "")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("should not call Label Studio when no key is configured")

    monkeypatch.setattr("src.deploy.label_studio_backend.httpx.post", fail_if_called)

    assert get_label_studio_auth_headers() == {}


def test_fetch_task_image_prefixes_relative_url_and_sends_auth_header(monkeypatch):
    monkeypatch.setenv("LABEL_STUDIO_URL", "http://ls-host:8080")
    monkeypatch.setenv("LABEL_STUDIO_API_KEY", "secret-refresh-token")
    _mock_token_refresh(monkeypatch)
    captured = {}

    def fake_get(url, headers=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        response = MagicMock()
        response.content = _png_bytes()
        response.raise_for_status = MagicMock()
        return response

    monkeypatch.setattr("src.deploy.label_studio_backend.httpx.get", fake_get)

    image = fetch_task_image("/data/upload/1/image.jpg")

    assert captured["url"] == "http://ls-host:8080/data/upload/1/image.jpg"
    assert captured["headers"] == {"Authorization": "Bearer fresh-access-token"}
    assert image.size == (64, 32)


def test_fetch_task_image_omits_auth_header_when_no_key(monkeypatch):
    monkeypatch.setenv("LABEL_STUDIO_URL", "http://ls-host:8080")
    monkeypatch.setenv("LABEL_STUDIO_API_KEY", "")
    captured = {}

    def fake_get(url, headers=None, timeout=None):
        captured["headers"] = headers
        response = MagicMock()
        response.content = _png_bytes()
        response.raise_for_status = MagicMock()
        return response

    monkeypatch.setattr("src.deploy.label_studio_backend.httpx.get", fake_get)

    fetch_task_image("http://elsewhere.example.com/image.jpg")

    assert captured["headers"] == {}


def test_health_returns_up_status():
    client = TestClient(app)

    response = client.get("/ls/health")

    assert response.status_code == 200
    assert response.json()["status"] == "UP"


def test_setup_returns_model_version(monkeypatch):
    monkeypatch.setenv("SMARTCART_MODEL_VERSION", "test-v2")
    client = TestClient(app)

    response = client.post("/ls/setup", json={"project": "1", "schema": "<View/>"})

    assert response.status_code == 200
    assert response.json() == {"model_version": "test-v2"}


def test_predict_route_returns_results_with_converted_boxes(monkeypatch):
    mock_detector = MagicMock()
    mock_detector.detect.return_value = [
        {"class_name": "Snacks/Chocolate-Bar", "confidence": 0.9, "bbox": [0.0, 0.0, 32.0, 16.0]}
    ]
    app.dependency_overrides[get_detector] = lambda: mock_detector
    monkeypatch.setattr(
        "src.deploy.label_studio_backend.fetch_task_image",
        lambda data_value: Image.new("RGB", (64, 32)),
    )
    client = TestClient(app)

    try:
        response = client.post("/ls/predict", json={"tasks": [{"id": 1, "data": {"image": "http://x/img.jpg"}}]})
    finally:
        app.dependency_overrides.pop(get_detector, None)

    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 1
    region = body["results"][0]["result"][0]
    assert region["value"]["width"] == 50.0
    assert region["from_name"] == "label"
    assert region["to_name"] == "image"


def test_predict_route_handles_fetch_failure_per_task(monkeypatch):
    mock_detector = MagicMock()
    app.dependency_overrides[get_detector] = lambda: mock_detector

    def raise_error(data_value):
        raise RuntimeError("fetch failed")

    monkeypatch.setattr("src.deploy.label_studio_backend.fetch_task_image", raise_error)
    client = TestClient(app)

    try:
        response = client.post("/ls/predict", json={"tasks": [{"id": 1, "data": {"image": "http://x/img.jpg"}}]})
    finally:
        app.dependency_overrides.pop(get_detector, None)

    assert response.status_code == 200
    assert response.json()["results"] == [{"model_version": response.json()["results"][0]["model_version"], "score": 0.0, "result": []}]
    mock_detector.detect.assert_not_called()
