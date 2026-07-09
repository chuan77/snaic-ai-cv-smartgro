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
from fastapi.staticfiles import StaticFiles
from PIL import Image

from src.deploy.active_learning_capture import get_capture_dir, maybe_capture
from src.deploy.detector import get_detector, get_variant_resolver
from src.deploy.label_studio_backend import router as ls_router
from src.models.variant_resolver import VariantResolver
from src.models.yolo_detector import YoloDetector

# Loaded here, not in main_api_server.py: uvicorn's --reload worker imports this
# module directly and never runs main_api_server.py's __main__ block, so loading
# .env only in the entrypoint would leave the reload worker on defaults.
load_dotenv()

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
app.include_router(ls_router, prefix="/ls")
get_capture_dir().mkdir(parents=True, exist_ok=True)
# Wide-open CORS on this mount specifically: Label Studio loads $image URLs into a canvas
# from whatever origin it's served on, which may not match SMARTCART_CORS_ORIGINS (that
# list is for the checkout frontend). Nested here, it only affects /staging — /predict and
# /catalog still enforce the app-level CORSMiddleware's restricted origin list above.
app.mount(
    "/staging",
    CORSMiddleware(StaticFiles(directory=str(get_capture_dir())), allow_origins=["*"], allow_methods=["GET"]),
    name="staging",
)


@app.get("/health")
def health_endpoint() -> dict[str, str]:
    return {"status": "ok"}


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


def resolve_variants(detections: list[dict], image: Image.Image, resolver: VariantResolver) -> list[dict]:
    """Refines each coarse detection's class_name to its specific catalog variant, by cropping
    the detection's box out of `image` and matching it against the DINOv2 gallery."""
    resolved = []
    for d in detections:
        crop = image.crop(tuple(d["bbox"]))
        resolved.append({**d, "class_name": resolver.resolve(d["class_name"], crop)})
    return resolved


@app.post("/predict")
def predict_endpoint(
    file: UploadFile,
    detector: YoloDetector = Depends(get_detector),
    resolver: VariantResolver = Depends(get_variant_resolver),
) -> list[dict]:
    image = Image.open(file.file).convert("RGB")
    frame_rgb = np.array(image)
    detections = detector.detect(frame_rgb)
    detections = resolve_variants(detections, image, resolver)
    maybe_capture(image, detections)
    return build_detections(detections, image.width, image.height)
