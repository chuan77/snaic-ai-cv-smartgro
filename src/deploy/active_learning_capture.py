"""Captures production frames the detector is uncertain about, staging them for
human review in Label Studio. Never allowed to affect the live /predict response."""
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path

from PIL import Image

logger = logging.getLogger("ActiveLearningCapture")


def get_capture_dir() -> Path:
    return Path(os.environ.get("SMARTCART_CAPTURE_DIR", "./artifacts/active_learning_staging"))


def get_capture_threshold() -> float:
    return float(os.environ.get("SMARTCART_CAPTURE_CONF_THRESHOLD", "0.5"))


def get_capture_enabled() -> bool:
    return os.environ.get("SMARTCART_CAPTURE_ENABLED", "true").strip().lower() in ("1", "true", "yes")


def is_uncertain(detections: list[dict], threshold: float) -> bool:
    if not detections:
        return True
    return any(d["confidence"] < threshold for d in detections)


def capture_frame(
    image: Image.Image, detections: list[dict], threshold: float, staging_dir: Path
) -> Path | None:
    if not is_uncertain(detections, threshold):
        return None

    staging_dir.mkdir(parents=True, exist_ok=True)
    name = uuid.uuid4().hex
    image_path = staging_dir / f"{name}.jpg"
    image.save(image_path, format="JPEG")

    confidences = [d["confidence"] for d in detections]
    sidecar = {
        "captured_at": datetime.now().isoformat(),
        "image_file": image_path.name,
        "image_width": image.width,
        "image_height": image.height,
        "num_detections": len(detections),
        "min_confidence": min(confidences) if confidences else None,
        "detections": detections,
    }
    sidecar_path = staging_dir / f"{name}.json"
    sidecar_path.write_text(json.dumps(sidecar))

    return image_path


def maybe_capture(image: Image.Image, detections: list[dict]) -> None:
    if not get_capture_enabled():
        return
    try:
        capture_frame(image, detections, get_capture_threshold(), get_capture_dir())
    except Exception:
        logger.warning("Active learning capture failed; continuing without it.", exc_info=True)
