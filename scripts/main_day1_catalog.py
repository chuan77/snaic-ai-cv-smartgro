# main_day1_catalog.py
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as T
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("Day1Gallery")

class GroceryDatasetIndexer:
    """Recursively walks Marcus Klasson's dataset to discover fine-grained leaf categories."""
    IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.webp'}

    def __init__(self, dataset_root: Path):
        self.root = dataset_root

    def build_class_map(self) -> Dict[str, int]:
        leaf_dirs = set()
        for p in self.root.rglob('*'):
            if p.is_dir() and any(f.suffix.lower() in self.IMAGE_EXTS for f in p.iterdir() if f.is_file()):
                leaf_dirs.add(p.relative_to(self.root).as_posix())
        sorted_classes = sorted(list(leaf_dirs))
        return {class_str: idx for idx, class_str in enumerate(sorted_classes)}

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
            
            base_price = 3.20 if "Packages" in class_str else 1.75
            catalog_records.append({"class_id": class_id, "product_name": class_str, "price_usd": base_price})
            
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

        pd.DataFrame(catalog_records).to_csv(self.output_dir / "catalog_prices.csv", index=False)
        pd.DataFrame(metadata).to_csv(self.output_dir / "gallery_meta.csv", index=False)
        np.save(self.output_dir / "gallery_index.npy", np.array(vectors))
        logger.info("✨ Day 1 product memory artifacts successfully written.")

if __name__ == "__main__":
    root_path = Path("./dataset/GroceryStoreDataset/dataset/train")
    if not root_path.exists():
        logger.error(f"Dataset root source missing at {root_path}. Please clone the repository first.")
    else:
        indexer = GroceryDatasetIndexer(root_path)
        cmap = indexer.build_class_map()
        gallery = UnifiedProductGallery(cmap, Path("./artifacts"))
        gallery.compile_gallery(root_path)
