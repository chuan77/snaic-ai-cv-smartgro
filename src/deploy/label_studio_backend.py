"""Label Studio ML Backend integration: serves the checkout YOLO detector's predictions
as pre-annotations for tasks in a Label Studio project."""
import io
import os
import uuid

import httpx
import numpy as np
from fastapi import APIRouter, Body, Depends
from PIL import Image

from src.deploy.detector import get_detector
from src.models.yolo_detector import YoloDetector

router = APIRouter()


def get_label_studio_url() -> str:
    return os.environ.get("LABEL_STUDIO_URL", "http://localhost:8080").rstrip("/")


def get_label_studio_api_key() -> str:
    return os.environ.get("LABEL_STUDIO_API_KEY", "")


def get_ls_from_name() -> str:
    return os.environ.get("SMARTCART_LS_FROM_NAME", "label")


def get_ls_to_name() -> str:
    return os.environ.get("SMARTCART_LS_TO_NAME", "image")


def get_ls_data_key() -> str:
    return os.environ.get("SMARTCART_LS_DATA_KEY", "image")


def get_model_version() -> str:
    return os.environ.get("SMARTCART_MODEL_VERSION", "smartcart-yolo11n-v1")


def xyxy_to_percentage_xywh(
    box: list[float], img_width: int, img_height: int
) -> tuple[float, float, float, float]:
    """Converts a pixel xyxy box to a percentage (0-100) top-left-origin (x, y, w, h) box."""
    x_min, y_min, x_max, y_max = box
    return (
        x_min / img_width * 100.0,
        y_min / img_height * 100.0,
        (x_max - x_min) / img_width * 100.0,
        (y_max - y_min) / img_height * 100.0,
    )


def detections_to_ls_regions(
    detections: list[dict], img_width: int, img_height: int, from_name: str, to_name: str
) -> list[dict]:
    regions = []
    for d in detections:
        x, y, w, h = xyxy_to_percentage_xywh(d["bbox"], img_width, img_height)
        regions.append(
            {
                "id": str(uuid.uuid4()),
                "from_name": from_name,
                "to_name": to_name,
                "type": "rectanglelabels",
                "original_width": img_width,
                "original_height": img_height,
                "image_rotation": 0,
                "value": {
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "rotation": 0,
                    "rectanglelabels": [d["class_name"]],
                },
                "score": d["confidence"],
            }
        )
    return regions


def build_ls_prediction(
    detections: list[dict],
    img_width: int,
    img_height: int,
    from_name: str,
    to_name: str,
    model_version: str,
) -> dict:
    regions = detections_to_ls_regions(detections, img_width, img_height, from_name, to_name)
    mean_score = sum(d["confidence"] for d in detections) / len(detections) if detections else 0.0
    return {"model_version": model_version, "score": mean_score, "result": regions}


def get_label_studio_auth_headers() -> dict[str, str]:
    """Label Studio's personal access tokens (as issued by its 'New Auth Token' dialog)
    are refresh tokens, not usable directly — each call must exchange one for a
    short-lived access token via /api/token/refresh first."""
    refresh_token = get_label_studio_api_key()
    if not refresh_token:
        return {}

    response = httpx.post(
        f"{get_label_studio_url()}/api/token/refresh", json={"refresh": refresh_token}, timeout=10.0
    )
    response.raise_for_status()
    return {"Authorization": f"Bearer {response.json()['access']}"}


def fetch_task_image(data_value: str) -> Image.Image:
    if data_value.startswith("http://") or data_value.startswith("https://"):
        url = data_value
    else:
        url = f"{get_label_studio_url()}{data_value if data_value.startswith('/') else '/' + data_value}"

    response = httpx.get(url, headers=get_label_studio_auth_headers(), timeout=10.0)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


@router.get("/health")
def health() -> dict:
    return {"status": "UP", "model_version": get_model_version()}


@router.post("/setup")
def setup(payload: dict = Body(default={})) -> dict:
    return {"model_version": get_model_version()}


@router.post("/predict")
def predict(payload: dict = Body(...), detector: YoloDetector = Depends(get_detector)) -> dict:
    from_name = get_ls_from_name()
    to_name = get_ls_to_name()
    data_key = get_ls_data_key()
    model_version = get_model_version()

    results = []
    for task in payload.get("tasks", []):
        data_value = task.get("data", {}).get(data_key)
        try:
            image = fetch_task_image(data_value)
            frame_rgb = np.array(image)
            detections = detector.detect(frame_rgb)
            results.append(
                build_ls_prediction(detections, image.width, image.height, from_name, to_name, model_version)
            )
        except Exception:
            results.append({"model_version": model_version, "score": 0.0, "result": []})

    return {"results": results}
