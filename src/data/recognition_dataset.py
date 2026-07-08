"""Full (uncapped) grocery dataset enumeration and synthetic ground-truth loading for the
Day 3 recognition pipeline. Unlike src/data/gallery.py's UnifiedProductGallery.compile_gallery
(capped at max_samples per class for the embedding gallery), this walks every image in every
leaf class for training/validating a classifier."""
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

from src.data.annotation_import import parse_yolo_bbox_line
from src.data.gallery import GroceryDatasetIndexer


def build_recognition_dataframe(
    dataset_root: Path, class_map: dict[str, int] | None = None
) -> pd.DataFrame:
    """One row per image under every leaf class dir. Columns: 'crop_path', 'fine'."""
    if class_map is None:
        class_map = GroceryDatasetIndexer(dataset_root).build_class_map()

    rows = []
    for class_str in class_map:
        class_dir = dataset_root / class_str
        for f in sorted(class_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in GroceryDatasetIndexer.IMAGE_EXTS:
                rows.append({"crop_path": str(f), "fine": class_str})

    return pd.DataFrame(rows, columns=["crop_path", "fine"])


def load_synthetic_val_ground_truth(
    data_yaml_path: Path, images_dir: Path, labels_dir: Path
) -> pd.DataFrame:
    """One row per YOLO ground-truth box. Columns: 'image_path', 'bbox' (pixel xyxy), 'fine'."""
    names = yaml.safe_load(data_yaml_path.read_text())["names"]

    rows = []
    for label_path in sorted(labels_dir.glob("*.txt")):
        image_path = next(
            (candidate for ext in (".jpg", ".jpeg", ".png", ".webp")
             if (candidate := images_dir / f"{label_path.stem}{ext}").exists()),
            None,
        )
        if image_path is None:
            continue

        with Image.open(image_path) as img:
            width, height = img.size

        for line in label_path.read_text().strip().splitlines():
            class_id = int(line.split()[0])
            bbox = parse_yolo_bbox_line(line, width, height)
            rows.append({"image_path": str(image_path), "bbox": bbox, "fine": names[class_id]})

    return pd.DataFrame(rows, columns=["image_path", "bbox", "fine"])
