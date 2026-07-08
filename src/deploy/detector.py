"""Shared YOLO detector singleton, used by both the checkout API and the Label Studio ML backend."""
import os
from functools import lru_cache
from pathlib import Path

from src.models.yolo_detector import YoloDetector

DEFAULT_WEIGHTS_PATH = Path("./runs/detect/train/weights/best.pt")


@lru_cache
def get_detector() -> YoloDetector:
    weights_path = Path(os.environ.get("SMARTCART_WEIGHTS_PATH", str(DEFAULT_WEIGHTS_PATH)))
    return YoloDetector(weights_path)
