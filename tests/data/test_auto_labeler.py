import json

from src.data.auto_labeler import (
    classify_capture,
    get_autolabel_min_conf,
    mark_consumed,
    pending_sidecars,
)


def test_get_autolabel_min_conf_default_and_override(monkeypatch):
    monkeypatch.delenv("SMARTCART_AUTOLABEL_MIN_CONF", raising=False)
    assert get_autolabel_min_conf() == 0.35

    monkeypatch.setenv("SMARTCART_AUTOLABEL_MIN_CONF", "0.4")
    assert get_autolabel_min_conf() == 0.4


def _write_sidecar(staging_dir, name, detections):
    staging_dir.mkdir(parents=True, exist_ok=True)
    (staging_dir / f"{name}.jpg").write_bytes(b"fake-jpg")
    sidecar = {
        "captured_at": "2026-07-11T00:00:00",
        "image_file": f"{name}.jpg",
        "image_width": 64,
        "image_height": 32,
        "num_detections": len(detections),
        "min_confidence": min((d["confidence"] for d in detections), default=None),
        "detections": detections,
    }
    (staging_dir / f"{name}.json").write_text(json.dumps(sidecar))
    return staging_dir / f"{name}.json"


def test_pending_sidecars_excludes_consumed_subdir(tmp_path):
    _write_sidecar(tmp_path, "one", [])
    consumed_dir = tmp_path / "consumed"
    _write_sidecar(consumed_dir, "already_done", [])

    pending = pending_sidecars(tmp_path)

    assert pending == [tmp_path / "one.json"]


def test_mark_consumed_moves_sidecar_but_leaves_image(tmp_path):
    sidecar_path = _write_sidecar(tmp_path, "one", [])

    mark_consumed(sidecar_path)

    assert not sidecar_path.exists()
    assert (tmp_path / "one.jpg").exists()
    assert (tmp_path / "consumed" / "one.json").exists()


def test_classify_capture_zero_or_low_when_no_detections():
    assert classify_capture([], min_conf=0.35, max_conf=0.5) == "zero_or_low"


def test_classify_capture_zero_or_low_when_any_detection_below_min_conf():
    detections = [{"confidence": 0.45}, {"confidence": 0.1}]

    assert classify_capture(detections, min_conf=0.35, max_conf=0.5) == "zero_or_low"


def test_classify_capture_mid_band_when_all_detections_in_band():
    detections = [{"confidence": 0.4}, {"confidence": 0.45}]

    assert classify_capture(detections, min_conf=0.35, max_conf=0.5) == "mid_band"


from unittest.mock import patch

from PIL import Image

from src.data.auto_labeler import auto_import_mid_band_detection


def test_auto_import_mid_band_detection_imports_when_vlm_agrees(tmp_path):
    image = Image.new("RGB", (100, 100), color=(255, 0, 0))
    detection = {"class_name": "Fruit/Apple/Royal-Gala", "confidence": 0.4, "bbox": [10, 10, 60, 60]}
    class_names = ["Fruit/Apple/Royal-Gala", "Snacks/Chocolate-Bar/Cadbury"]

    with patch("src.data.auto_labeler.ask_vlm", return_value="This looks like a Royal-Gala apple"):
        result = auto_import_mid_band_detection(image, detection, class_names, tmp_path, capture_id="cap1")

    expected_path = tmp_path / "Fruit/Apple/Royal-Gala" / "cap1.jpg"
    assert result == expected_path
    assert expected_path.exists()


def test_auto_import_mid_band_detection_discards_when_vlm_disagrees(tmp_path):
    image = Image.new("RGB", (100, 100))
    detection = {"class_name": "Fruit/Apple/Royal-Gala", "confidence": 0.4, "bbox": [10, 10, 60, 60]}
    class_names = ["Fruit/Apple/Royal-Gala", "Snacks/Chocolate-Bar/Cadbury"]

    with patch("src.data.auto_labeler.ask_vlm", return_value="This is a Cadbury chocolate bar"):
        result = auto_import_mid_band_detection(image, detection, class_names, tmp_path, capture_id="cap1")

    assert result is None
    assert not (tmp_path / "Fruit/Apple/Royal-Gala").exists()


def test_auto_import_mid_band_detection_discards_when_vlm_unavailable(tmp_path):
    image = Image.new("RGB", (100, 100))
    detection = {"class_name": "Fruit/Apple/Royal-Gala", "confidence": 0.4, "bbox": [10, 10, 60, 60]}

    with patch("src.data.auto_labeler.ask_vlm", return_value=None):
        result = auto_import_mid_band_detection(
            image, detection, ["Fruit/Apple/Royal-Gala"], tmp_path, capture_id="cap1"
        )

    assert result is None


def test_auto_import_mid_band_detection_discards_when_vlm_matches_unknown_category(tmp_path):
    image = Image.new("RGB", (100, 100))
    detection = {"class_name": "Fruit/Apple/Royal-Gala", "confidence": 0.4, "bbox": [10, 10, 60, 60]}

    with patch("src.data.auto_labeler.ask_vlm", return_value="I have no idea what this is"):
        result = auto_import_mid_band_detection(
            image, detection, ["Fruit/Apple/Royal-Gala"], tmp_path, capture_id="cap1"
        )

    assert result is None
