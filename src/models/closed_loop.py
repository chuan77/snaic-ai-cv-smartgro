"""Closes the loop: runs Day 2's real trained YOLO detector over real images, classifies each
detected crop with the Day 3 LinearHead, and scores predictions against ground truth (not
against YOLO's own class guess — that would be circular and hide silent misreads)."""
import numpy as np
import pandas as pd
from PIL import Image

from src.data.annotation_import import crop_with_padding
from src.models.dino_extractor import build_transform, extract_features


def compute_iou(box_a: tuple, box_b: tuple) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    return inter_area / union if union > 0 else 0.0


def match_ground_truth(
    detection_bbox: tuple, gt_boxes: list[tuple[tuple, str]], iou_threshold: float = 0.3
) -> str | None:
    best_label = None
    best_iou = iou_threshold
    for box, label in gt_boxes:
        iou = compute_iou(detection_bbox, box)
        if iou > best_iou:
            best_iou = iou
            best_label = label
    return best_label


def run_closed_loop_evaluation(
    detector,
    head,
    backbone,
    ground_truth: pd.DataFrame,
    cls_to_i: dict[str, int],
    conf: float = 0.25,
    iou_threshold: float = 0.3,
) -> dict:
    idx_to_cls = {idx: name for name, idx in cls_to_i.items()}
    transform = build_transform()
    records = []
    n_unmatched = 0

    for image_path, group in ground_truth.groupby("image_path"):
        gt_boxes = list(zip(group["bbox"], group["fine"]))

        with Image.open(image_path).convert("RGB") as img:
            frame_rgb = np.array(img)
            detections = detector.detect(frame_rgb, conf=conf)

            for det in detections:
                bbox = tuple(int(v) for v in det["bbox"])
                gt_label = match_ground_truth(bbox, gt_boxes, iou_threshold)
                if gt_label is None:
                    n_unmatched += 1
                    continue

                crop = crop_with_padding(img, bbox)
                crop_tensor = transform(crop).unsqueeze(0)
                features = extract_features(backbone, [crop_tensor])
                pred_label = idx_to_cls[int(head.predict(features)[0])]
                records.append({
                    "image_path": str(image_path),
                    "gt_label": gt_label,
                    "pred_label": pred_label,
                    "correct": pred_label == gt_label,
                })

    n_evaluated = len(records)
    accuracy = (sum(r["correct"] for r in records) / n_evaluated) if n_evaluated else 0.0
    return {
        "accuracy": accuracy,
        "n_evaluated": n_evaluated,
        "n_unmatched": n_unmatched,
        "records": records,
    }
