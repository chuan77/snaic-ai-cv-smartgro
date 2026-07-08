"""DINOv2 embedding gallery construction.

`GroceryDatasetIndexer` + `UnifiedProductGallery` are the pair used live by
main_day1_catalog.py. `EmbeddingGallery` and `DeterministicGalleryBuilder` are an alternate,
self-discovering embedding pipeline (backed by `HardenedProductCatalog`) that is not currently
called from any entrypoint.
"""
import os
import logging
from pathlib import Path
import hashlib
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as T
from typing import Dict, List, Any, Tuple
from src.data.catalog import HardenedProductCatalog

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Day1DataPipeline")


def get_base_price(class_str: str) -> float:
    """Maps a leaf class's dataset-relative path to its default catalog price by top-level category."""
    category_prices = {
        "Packages": 3.20,
        "Ready-To-Eat": 3.00,
        "Snacks": 6.20,
    }
    top_level = class_str.split("/")[0]
    return category_prices.get(top_level, 1.75)


def get_item_price(class_str: str) -> float:
    """Derives a stable, per-item charm price by jittering the category base ±15%."""
    base = get_base_price(class_str)
    seed = int(hashlib.sha256(class_str.encode()).hexdigest(), 16)
    jittered = base * random.Random(seed).uniform(0.85, 1.15)
    price = round(jittered * 10) / 10 - 0.01
    return round(price, 2)


class UnifiedProductGallery:
    """Extracts DINOv2 visual features to construct the reference item registry."""
    def __init__(self, class_map: Dict[str, int], output_dir: Path):
        self.class_map = class_map
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"Target hardware acceleration platform: {self.device}")
        
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(self.device).eval()
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def compile_gallery(self, src_root: Path, max_samples: int = 5):
        vectors: List[np.ndarray] = []
        metadata: List[Dict[str, Any]] = []
        catalog_records: List[Dict[str, Any]] = []

        for class_str, class_id in self.class_map.items():
            class_dir = src_root / class_str
            valid_files = sorted([f for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in GroceryDatasetIndexer.IMAGE_EXTS])
            selected_files = valid_files[:min(len(valid_files), max_samples)]
            
            # Determine per-item price with randomized jitter around the category base
            item_price = get_item_price(class_str)
            catalog_records.append({"class_id": class_id, "product_name": class_str, "price_usd": item_price})
            
            logger.info(f"Indexing Class [{class_id:03d}] -> {class_str} ({len(selected_files)} items)")
            
            for img_path in selected_files:
                try:
                    with Image.open(img_path).convert('RGB') as img:
                        tensor = self.transform(img).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            embedding = self.backbone(tensor).squeeze().cpu().numpy()
                        vectors.append(embedding)
                        metadata.append({"class_id": class_id, "product_name": class_str, "file_name": img_path.name})
                except Exception as e:
                    logger.error(f"Error parsing feature vector at {img_path}: {e}")

        # Persist baseline gallery bundles securely
        pd.DataFrame(catalog_records).to_csv(self.output_dir / "catalog_prices.csv", index=False)
        pd.DataFrame(metadata).to_csv(self.output_dir / "gallery_meta.csv", index=False)
        np.save(self.output_dir / "gallery_index.npy", np.array(vectors))
        logger.info("✨ Day 1 product memory artifacts successfully written.")

class GroceryDatasetIndexer:
    """Recursively walks Marcus Klasson's dataset to discover fine-grained leaf categories."""
    IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.webp'}

    def __init__(self, dataset_root: Path):
        self.root = dataset_root

    def build_class_map(self) -> Dict[str, int]:
        leaf_dirs = set()
        # Discover directories that contain actual image assets
        for p in self.root.rglob('*'):
            if p.is_dir() and any(f.suffix.lower() in self.IMAGE_EXTS for f in p.iterdir() if f.is_file()):
                # Standardize relative path as class identifier (e.g., "Fruit/Apple/Golden-Delicious")
                leaf_dirs.add(p.relative_to(self.root).as_posix())
        
        # Enforce strict deterministic sorting to maintain label identity across runtimes
        sorted_classes = sorted(list(leaf_dirs))
        return {class_str: idx for idx, class_str in enumerate(sorted_classes)}
    
class EmbeddingGallery:
    """Handles serialization of high-dimensional visual memory vectors."""
    def __init__(self, index_path: str = "gallery_index.npy", meta_path: str = "gallery_meta.csv"):
        self._index_path = index_path
        self._meta_path = meta_path

    def serialize(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        np.save(self._index_path, embeddings)
        pd.DataFrame(metadata).to_csv(self._meta_path, index=False)

class DeterministicGalleryBuilder:
    """Extracts invariant visual feature vectors while excluding OS artifacts like .DS_Store."""
    VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp'}

    def __init__(self, src_dir: Path, idx_out: Path, meta_out: Path):
        self.src_dir = src_dir
        self.idx_out = idx_out
        self.meta_out = meta_out
        
        # Verify Apple Silicon hardware capabilities
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"Using hardware acceleration device backend: {self.device}")
        
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(self.device).eval()
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def execute_ingestion(self, samples_per_class: int = 5) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        if not self.src_dir.exists():
            raise FileNotFoundError(f"Missing baseline dataset pathway: {self.src_dir}")

        # Enforce deterministic alphabetical sorting across all deployment environments
        class_dirs = sorted([d for d in self.src_dir.iterdir() if d.is_dir()])
        
        vectors: List[np.ndarray] = []
        metadata: List[Dict[str, Any]] = []
        catalog = HardenedProductCatalog(self.idx_out.parent / "catalog_prices.csv")

        for class_idx, class_dir in enumerate(class_dirs):
            class_name = class_dir.name
            catalog.register_product(class_idx, class_name)

            # Strict file filtering to bypass hidden OS system entries (.DS_Store)
            valid_files = sorted([f for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in self.VALID_EXTENSIONS])
            selected_files = valid_files[:min(len(valid_files), samples_per_class)]

            logger.info(f"Processing Class [{class_idx:02d}] '{class_name}' -> Ingesting {len(selected_files)} assets.")

            for img_path in selected_files:
                try:
                    with Image.open(img_path).convert('RGB') as img:
                        tensor = self.transform(img).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            feat_vector = self.dinov2(tensor).squeeze().cpu().numpy()
                        
                        vectors.append(feat_vector)
                        metadata.append({"class_id": class_idx, "product_name": class_name, "file_name": img_path.name})
                except Exception as e:
                    logger.error(f"Failed parsing asset matrix at '{img_path.name}': {str(e)}")

        catalog.serialize()
        np.save(self.idx_out, np.array(vectors))
        pd.DataFrame(metadata).to_csv(self.meta_out, index=False)
        logger.info("✨ Day 1 operations successfully finalized.")
        return np.array(vectors), metadata