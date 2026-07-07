"""FastAPI backend serving real YOLO detections and catalog data to the frontend."""
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

DEFAULT_WEIGHTS_PATH = Path("./runs/detect/train/weights/best.pt")
DEFAULT_CATALOG_PATH = Path("./artifacts/catalog_prices.csv")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
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
    return load_catalog(DEFAULT_CATALOG_PATH)


@app.get("/catalog")
def catalog_endpoint(catalog: list[dict] = Depends(get_catalog)) -> list[dict]:
    return catalog
