"""Standalone YOLO detection wrapper. Not yet wired into any entrypoint — src/deploy/register.py
currently runs YOLO inference with its own inline predict/parse logic."""
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


class YoloDetector:
    """Wraps a YOLO model to run detection on a single RGB frame."""

    def __init__(self, weights_path: Path):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = YOLO(str(weights_path))

    def detect(self, frame_rgb: np.ndarray, conf: float = 0.25) -> list[dict]:
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        results = self.model.predict(source=frame_bgr, conf=conf, device=str(self.device), verbose=False)
        result = results[0]
        return [
            {
                "class_name": self.model.names.get(int(box.cls[0]), "Unknown"),
                "confidence": float(box.conf[0]),
                "bbox": [float(v) for v in box.xyxy[0]],
            }
            for box in result.boxes
        ]
