"""FastAPI backend serving real YOLO detections and catalog data to the frontend."""
import os
from functools import lru_cache
from pathlib import Path

import uuid

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from src.models.yolo_detector import YoloDetector

# Loaded here, not in main_api_server.py: uvicorn's --reload worker imports this
# module directly and never runs main_api_server.py's __main__ block, so loading
# .env only in the entrypoint would leave the reload worker on defaults.
load_dotenv()

DEFAULT_WEIGHTS_PATH = Path("./runs/detect/train/weights/best.pt")
DEFAULT_CATALOG_PATH = Path("./artifacts/catalog_prices.csv")
DEFAULT_CORS_ORIGINS = "http://localhost:5173"


def get_cors_origins() -> list[str]:
    raw = os.environ.get("SMARTCART_CORS_ORIGINS", DEFAULT_CORS_ORIGINS)
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def get_catalog_path() -> Path:
    return Path(os.environ.get("SMARTCART_CATALOG_PATH", str(DEFAULT_CATALOG_PATH)))


def parse_bool_env(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes")


def get_server_config() -> dict[str, str | int | bool]:
    return {
        "host": os.environ.get("SMARTCART_HOST", "0.0.0.0"),
        "port": int(os.environ.get("SMARTCART_PORT", "8000")),
        "reload": parse_bool_env(os.environ.get("SMARTCART_RELOAD", "true")),
    }


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)


def leaf_display_name(product_name: str) -> str:
    """Derives a human-readable label from a dataset-relative class path, e.g.
    'Fruit/Apple/Royal-Gala' -> 'Royal Gala'."""
    leaf = product_name.split("/")[-1]
    return leaf.replace("-", " ").title()


def xyxy_to_fractional_xywh(
    box: list[float], img_width: int, img_height: int
) -> tuple[float, float, float, float]:
    """Converts a pixel xyxy box to a fractional top-left-origin (x, y, w, h) box."""
    x_min, y_min, x_max, y_max = box
    return (
        x_min / img_width,
        y_min / img_height,
        (x_max - x_min) / img_width,
        (y_max - y_min) / img_height,
    )


def load_catalog(csv_path: Path) -> list[dict]:
    df = pd.read_csv(csv_path)
    return [
        {"sku": row.product_name, "name": leaf_display_name(row.product_name), "priceUsd": float(row.price_usd)}
        for row in df.itertuples()
    ]


@lru_cache
def get_catalog() -> list[dict]:
    return load_catalog(get_catalog_path())


@app.get("/catalog")
def catalog_endpoint(catalog: list[dict] = Depends(get_catalog)) -> list[dict]:
    return catalog


@lru_cache
def get_detector() -> YoloDetector:
    weights_path = Path(os.environ.get("SMARTCART_WEIGHTS_PATH", str(DEFAULT_WEIGHTS_PATH)))
    return YoloDetector(weights_path)


def build_detections(detections: list[dict], img_width: int, img_height: int) -> list[dict]:
    return [
        {
            "id": str(uuid.uuid4()),
            "label": d["class_name"],
            "confidence": d["confidence"],
            "bbox": list(xyxy_to_fractional_xywh(d["bbox"], img_width, img_height)),
        }
        for d in detections
    ]


@app.post("/predict")
def predict_endpoint(file: UploadFile, detector: YoloDetector = Depends(get_detector)) -> list[dict]:
    image = Image.open(file.file).convert("RGB")
    frame_rgb = np.array(image)
    detections = detector.detect(frame_rgb)
    return build_detections(detections, image.width, image.height)
