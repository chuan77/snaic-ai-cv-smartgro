# main_day2_train.py
import logging
import random
import cv2
import numpy as np
import yaml
from pathlib import Path
from ultralytics import YOLO
from typing import Dict
from main_day1_catalog import GroceryDatasetIndexer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("Day2Synthesis")

class ProgrammaticCheckoutSynthesizer:
    """Composites individual classification photos onto procedural backgrounds to simulate crowded registers."""
    def __init__(self, src_root: Path, dest_root: Path, class_map: Dict[str, int]):
        self.src_root = src_root
        self.dest_root = dest_root
        self.class_map = class_map
        self.image_exts = {'.png', '.jpg', '.jpeg'}

    def _render_canvas(self, width: int = 640, height: int = 640) -> np.ndarray:
        base_color = np.array([random.randint(215, 225), random.randint(220, 230), random.randint(220, 230)], dtype=np.uint8)
        canvas = np.full((height, width, 3), base_color, dtype=np.uint8)
        noise = np.random.normal(0, 2.5, (height, width, 3)).astype(np.int16)
        return np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    def generate_split(self, split_name: str, total_scenes: int = 200):
        img_out = self.dest_root / split_name / "images"
        lbl_out = self.dest_root / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        all_products = []
        for class_str, class_id in self.class_map.items():
            class_dir = self.src_root / class_str
            if class_dir.exists():
                files = [f for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in self.image_exts]
                for f in files:
                    all_products.append((class_id, f))

        for scene_idx in range(total_scenes):
            canvas = self._render_canvas(640, 640)
            labels = []
            num_items = random.randint(2, 5)
            
            for _ in range(num_items):
                if not all_products: continue
                class_id, img_path = random.choice(all_products)
                item_img = cv2.imread(str(img_path))
                if item_img is None: continue

                scale = random.uniform(0.20, 0.33)
                h, w = item_img.shape[:2]
                new_h, new_w = int(640 * scale), int(640 * scale * (w / h))
                item_resized = cv2.resize(item_img, (new_w, new_h))

                x_max, y_max = 640 - new_w - 15, 640 - new_h - 15
                if x_max <= 15 or y_max <= 15: continue
                x_min, y_min = random.randint(15, x_max), random.randint(15, y_max)

                canvas[y_min:y_min+new_h, x_min:x_min+new_w] = item_resized

                x_center = (x_min + (new_w / 2.0)) / 640.0
                y_center = (y_min + (new_h / 2.0)) / 640.0
                norm_w = new_w / 640.0
                norm_h = new_h / 640.0
                labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

            base_name = f"synth_checkout_{split_name}_{scene_idx:05d}"
            cv2.imwrite(str(img_out / f"{base_name}.jpg"), canvas)
            with open(lbl_out / f"{base_name}.txt", "w") as lf:
                lf.write("
".join(labels))

        logger.info(f"📋 Generation completed: {total_scenes} scenes rendered for split '{split_name}'.")

    def write_yaml_config(self) -> Path:
        config = {
            'train': str((self.dest_root / 'train' / 'images').absolute()),
            'val': str((self.dest_root / 'val' / 'images').absolute()),
            'nc': len(self.class_map),
            'names': {idx: name for name, idx in self.class_map.items()}
        }
        yaml_path = self.dest_root / "data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        return yaml_path

if __name__ == "__main__":
    root_path = Path("./dataset/GroceryStoreDataset/dataset/train")
    if not root_path.exists():
         print("Dataset path missing.")
    else:
         indexer = GroceryDatasetIndexer(root_path)
         cmap = indexer.build_class_map()
         synth = ProgrammaticCheckoutSynthesizer(root_path, Path("./synthetic_dataset"), cmap)
         synth.generate_split("train", total_scenes=100)
         synth.generate_split("val", total_scenes=20)
         ypath = synth.write_yaml_config()
         
         model = YOLO("yolo11n.pt")
         model.train(data=str(ypath), epochs=5, imgsz=640, device="mps" if torch.backends.mps.is_available() else "cpu", workers=0)
