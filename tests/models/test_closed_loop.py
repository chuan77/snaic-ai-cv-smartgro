from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.models.closed_loop import compute_iou, match_ground_truth, run_closed_loop_evaluation


@pytest.mark.parametrize("box_a,box_b,expected", [
    ((0, 0, 10, 10), (0, 0, 10, 10), 1.0),
    ((0, 0, 10, 10), (20, 20, 30, 30), 0.0),
    ((0, 0, 10, 10), (5, 5, 15, 15), 25 / 175),
])
def test_compute_iou(box_a, box_b, expected):
    assert compute_iou(box_a, box_b) == pytest.approx(expected)


def test_match_ground_truth_returns_label_of_best_overlapping_box():
    gt_boxes = [((0, 0, 10, 10), "apple"), ((100, 100, 110, 110), "banana")]

    result = match_ground_truth((1, 1, 9, 9), gt_boxes, iou_threshold=0.3)

    assert result == "apple"


def test_match_ground_truth_returns_none_when_no_box_clears_threshold():
    gt_boxes = [((0, 0, 10, 10), "apple")]

    result = match_ground_truth((50, 50, 60, 60), gt_boxes, iou_threshold=0.3)

    assert result is None


def test_run_closed_loop_evaluation_scores_against_ground_truth_not_yolo_label(tmp_path):
    image_path = tmp_path / "scene_001.jpg"
    Image.new("RGB", (100, 100)).save(image_path)

    ground_truth = pd.DataFrame([
        {"image_path": str(image_path), "bbox": (10, 10, 30, 30), "fine": "apple"},
        {"image_path": str(image_path), "bbox": (60, 60, 80, 80), "fine": "banana"},
    ])

    detector = MagicMock()
    detector.detect.return_value = [
        {"class_name": "wrong-yolo-guess", "confidence": 0.9, "bbox": [10.0, 10.0, 30.0, 30.0]},
        {"class_name": "banana", "confidence": 0.9, "bbox": [60.0, 60.0, 80.0, 80.0]},
        {"class_name": "apple", "confidence": 0.9, "bbox": [0.0, 0.0, 5.0, 5.0]},
    ]

    cls_to_i = {"apple": 0, "banana": 1}
    head = MagicMock()
    head.predict.side_effect = [np.array([0]), np.array([1])]

    with patch("src.models.closed_loop.extract_features", return_value=np.zeros((1, 4))):
        result = run_closed_loop_evaluation(
            detector=detector, head=head, backbone=MagicMock(),
            ground_truth=ground_truth, cls_to_i=cls_to_i,
        )

    assert result["n_evaluated"] == 2
    assert result["n_unmatched"] == 1
    assert result["accuracy"] == 1.0


def test_run_closed_loop_evaluation_counts_incorrect_predictions(tmp_path):
    image_path = tmp_path / "scene_002.jpg"
    Image.new("RGB", (100, 100)).save(image_path)

    ground_truth = pd.DataFrame([
        {"image_path": str(image_path), "bbox": (10, 10, 30, 30), "fine": "apple"},
    ])
    detector = MagicMock()
    detector.detect.return_value = [
        {"class_name": "apple", "confidence": 0.9, "bbox": [10.0, 10.0, 30.0, 30.0]},
    ]
    cls_to_i = {"apple": 0, "banana": 1}
    head = MagicMock()
    head.predict.return_value = np.array([1])

    with patch("src.models.closed_loop.extract_features", return_value=np.zeros((1, 4))):
        result = run_closed_loop_evaluation(
            detector=detector, head=head, backbone=MagicMock(),
            ground_truth=ground_truth, cls_to_i=cls_to_i,
        )

    assert result["n_evaluated"] == 1
    assert result["accuracy"] == 0.0
    assert result["records"][0]["pred_label"] == "banana"
    assert result["records"][0]["gt_label"] == "apple"
