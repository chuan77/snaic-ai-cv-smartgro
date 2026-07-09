"""Shared YOLO detector and variant resolver singletons, used by both the checkout API and the
Label Studio ML backend."""
import os
from functools import lru_cache
from pathlib import Path

from src.models.variant_resolver import VariantResolver
from src.models.yolo_detector import YoloDetector

DEFAULT_WEIGHTS_PATH = Path("./runs/detect/train/weights/best.pt")
DEFAULT_GALLERY_INDEX_PATH = Path("./artifacts/gallery_index.npy")
DEFAULT_GALLERY_META_PATH = Path("./artifacts/gallery_meta.csv")


@lru_cache
def get_detector() -> YoloDetector:
    weights_path = Path(os.environ.get("SMARTCART_WEIGHTS_PATH", str(DEFAULT_WEIGHTS_PATH)))
    return YoloDetector(weights_path)


def get_gallery_index_path() -> Path:
    return Path(os.environ.get("SMARTCART_GALLERY_INDEX_PATH", str(DEFAULT_GALLERY_INDEX_PATH)))


def get_gallery_meta_path() -> Path:
    return Path(os.environ.get("SMARTCART_GALLERY_META_PATH", str(DEFAULT_GALLERY_META_PATH)))


@lru_cache
def get_variant_resolver() -> VariantResolver:
    return VariantResolver(get_gallery_index_path(), get_gallery_meta_path())
