import json

from PIL import Image

from src.deploy.active_learning_capture import capture_frame, is_uncertain, maybe_capture


def test_is_uncertain_true_when_no_detections():
    assert is_uncertain([], threshold=0.5) is True


def test_is_uncertain_true_when_any_below_threshold():
    detections = [{"confidence": 0.9}, {"confidence": 0.3}]

    assert is_uncertain(detections, threshold=0.5) is True


def test_is_uncertain_false_when_all_above_threshold():
    detections = [{"confidence": 0.9}, {"confidence": 0.6}]

    assert is_uncertain(detections, threshold=0.5) is False


def test_capture_frame_writes_image_and_sidecar_when_uncertain(tmp_path):
    image = Image.new("RGB", (64, 32), color="blue")
    detections = [{"class_name": "Vegetables/Carrots", "confidence": 0.2, "bbox": [1.0, 2.0, 3.0, 4.0]}]

    result_path = capture_frame(image, detections, threshold=0.5, staging_dir=tmp_path)

    assert result_path is not None
    assert result_path.exists()
    assert result_path.suffix == ".jpg"
    sidecar_path = result_path.with_suffix(".json")
    assert sidecar_path.exists()
    sidecar = json.loads(sidecar_path.read_text())
    assert sidecar["image_file"] == result_path.name
    assert sidecar["image_width"] == 64
    assert sidecar["image_height"] == 32
    assert sidecar["num_detections"] == 1
    assert sidecar["min_confidence"] == 0.2
    assert sidecar["detections"] == detections
    assert "captured_at" in sidecar


def test_capture_frame_returns_none_when_confident(tmp_path):
    image = Image.new("RGB", (64, 32), color="blue")
    detections = [{"class_name": "Vegetables/Carrots", "confidence": 0.9, "bbox": [1.0, 2.0, 3.0, 4.0]}]

    result_path = capture_frame(image, detections, threshold=0.5, staging_dir=tmp_path)

    assert result_path is None
    assert list(tmp_path.iterdir()) == []


def test_maybe_capture_swallows_errors(monkeypatch):
    monkeypatch.setenv("SMARTCART_CAPTURE_ENABLED", "true")
    monkeypatch.setattr(
        "src.deploy.active_learning_capture.capture_frame",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    maybe_capture(Image.new("RGB", (1, 1)), [])


def test_maybe_capture_noop_when_disabled(monkeypatch):
    monkeypatch.setenv("SMARTCART_CAPTURE_ENABLED", "false")
    called = []
    monkeypatch.setattr(
        "src.deploy.active_learning_capture.capture_frame",
        lambda *a, **k: called.append(True),
    )

    maybe_capture(Image.new("RGB", (1, 1)), [])

    assert called == []
