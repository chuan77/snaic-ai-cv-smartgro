from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch

from src.models.yolo_detector import YoloDetector


class FakeBox:
    """Mimics one row of ultralytics.engine.results.Boxes."""

    def __init__(self, cls_id, conf, xyxy):
        self.cls = [cls_id]
        self.conf = [conf]
        self.xyxy = [xyxy]


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


def test_constructor_loads_yolo_with_str_weights_path():
    with patch("src.models.yolo_detector.YOLO") as mock_yolo:
        YoloDetector(Path("fake.pt"))

    mock_yolo.assert_called_once_with("fake.pt")


def test_constructor_selects_mps_device_when_available():
    with patch("src.models.yolo_detector.YOLO"), \
         patch("src.models.yolo_detector.torch.backends.mps.is_available", return_value=True):
        detector = YoloDetector(Path("fake.pt"))

    assert detector.device == torch.device("mps")


def test_constructor_selects_cpu_device_when_mps_unavailable():
    with patch("src.models.yolo_detector.YOLO"), \
         patch("src.models.yolo_detector.torch.backends.mps.is_available", return_value=False):
        detector = YoloDetector(Path("fake.pt"))

    assert detector.device == torch.device("cpu")


def test_detect_converts_rgb_to_bgr_before_predict():
    frame_rgb = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
    expected_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    with patch("src.models.yolo_detector.YOLO") as mock_yolo:
        mock_model = mock_yolo.return_value
        mock_model.predict.return_value = [FakeResult([])]
        detector = YoloDetector(Path("fake.pt"))
        detector.detect(frame_rgb)

    called_source = mock_model.predict.call_args.kwargs["source"]
    np.testing.assert_array_equal(called_source, expected_bgr)


def test_detect_passes_given_conf_threshold_to_predict():
    frame_rgb = np.zeros((2, 2, 3), dtype=np.uint8)

    with patch("src.models.yolo_detector.YOLO") as mock_yolo:
        mock_model = mock_yolo.return_value
        mock_model.predict.return_value = [FakeResult([])]
        detector = YoloDetector(Path("fake.pt"))
        detector.detect(frame_rgb, conf=0.6)

    assert mock_model.predict.call_args.kwargs["conf"] == 0.6


def test_detect_defaults_conf_to_point_25():
    frame_rgb = np.zeros((2, 2, 3), dtype=np.uint8)

    with patch("src.models.yolo_detector.YOLO") as mock_yolo:
        mock_model = mock_yolo.return_value
        mock_model.predict.return_value = [FakeResult([])]
        detector = YoloDetector(Path("fake.pt"))
        detector.detect(frame_rgb)

    assert mock_model.predict.call_args.kwargs["conf"] == 0.25


def test_detect_passes_selected_device_to_predict():
    frame_rgb = np.zeros((2, 2, 3), dtype=np.uint8)

    with patch("src.models.yolo_detector.YOLO") as mock_yolo, \
         patch("src.models.yolo_detector.torch.backends.mps.is_available", return_value=False):
        mock_model = mock_yolo.return_value
        mock_model.predict.return_value = [FakeResult([])]
        detector = YoloDetector(Path("fake.pt"))
        detector.detect(frame_rgb)

    assert mock_model.predict.call_args.kwargs["device"] == str(detector.device)


def test_detect_returns_empty_list_when_no_boxes():
    frame_rgb = np.zeros((2, 2, 3), dtype=np.uint8)

    with patch("src.models.yolo_detector.YOLO") as mock_yolo:
        mock_model = mock_yolo.return_value
        mock_model.predict.return_value = [FakeResult([])]
        detector = YoloDetector(Path("fake.pt"))
        result = detector.detect(frame_rgb)

    assert result == []


def test_detect_returns_one_dict_per_box_with_correct_fields():
    frame_rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    box = FakeBox(cls_id=1, conf=0.87, xyxy=[10.0, 20.0, 30.0, 40.0])

    with patch("src.models.yolo_detector.YOLO") as mock_yolo:
        mock_model = mock_yolo.return_value
        mock_model.names = {0: "Apple", 1: "Banana"}
        mock_model.predict.return_value = [FakeResult([box])]
        detector = YoloDetector(Path("fake.pt"))
        result = detector.detect(frame_rgb)

    assert len(result) == 1
    assert result[0]["class_name"] == "Banana"
    assert result[0]["confidence"] == pytest.approx(0.87)
    assert result[0]["bbox"] == [10.0, 20.0, 30.0, 40.0]


def test_detect_returns_unknown_for_unmapped_class_id():
    frame_rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    box = FakeBox(cls_id=99, conf=0.5, xyxy=[0.0, 0.0, 1.0, 1.0])

    with patch("src.models.yolo_detector.YOLO") as mock_yolo:
        mock_model = mock_yolo.return_value
        mock_model.names = {0: "Apple", 1: "Banana"}
        mock_model.predict.return_value = [FakeResult([box])]
        detector = YoloDetector(Path("fake.pt"))
        result = detector.detect(frame_rgb)

    assert result[0]["class_name"] == "Unknown"


@pytest.mark.slow
def test_detect_smoke_with_real_weights():
    detector = YoloDetector(Path("yolo11n.pt"))
    frame = np.zeros((640, 640, 3), dtype=np.uint8)

    result = detector.detect(frame)

    assert isinstance(result, list)
