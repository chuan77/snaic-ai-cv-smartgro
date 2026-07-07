import os
from pathlib import Path

def setup_smartcart_architecture():
    # 1. Define target directory mapping structure
    base_dir = Path(".")
    
    directories = [
        base_dir / "dataset",
        base_dir / "artifacts",
        base_dir / "synthetic_dataset" / "train" / "images",
        base_dir / "synthetic_dataset" / "train" / "labels",
        base_dir / "synthetic_dataset" / "val" / "images",
        base_dir / "synthetic_dataset" / "val" / "labels",
        base_dir / "runs" / "detect",
        base_dir / "scripts",
    ]
    
    print("🚀 Initializing SmartCart AI project environment setup...")
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created folder path: {directory}")

    # 2. Write standard requirements.txt file
    requirements_content = """# Core Framework Dependencies
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.1.0
gradio>=4.0.0
opencv-python>=4.7.0
numpy>=1.24.0
pandas>=2.0.0
pyyaml>=6.0
pillow>=9.5.0
"""
    with open(base_dir / "requirements.txt", "w") as req_file:
        req_file.write(requirements_content.strip() + "\n")
    print("📝 Written: requirements.txt")

    # 3. Define script files and content mappings
    # Day 1 Script Content
    day1_code = """# main_day1_catalog.py
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
    \"\"\"Recursively walks Marcus Klasson's dataset to discover fine-grained leaf categories.\"\"\"
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
    \"\"\"Extracts DINOv2 visual features to construct the reference item registry.\"\"\"
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
"""

    # Day 2 Script Content
    day2_code = """# main_day2_train.py
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
    \"\"\"Composites individual classification photos onto procedural backgrounds to simulate crowded registers.\"\"\"
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
                lf.write("\n".join(labels))

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
"""

    # Day 3 Script Content
    day3_code = """# main_day3_evaluate.py
import logging
from pathlib import Path
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("Day3Audit")

class CheckoutModelAuditor:
    \"\"\"Runs data audits over compiled validation sets to pinpoint classification error trends.\"\"\"
    def __init__(self, check_weights: Path):
        self.model = YOLO(str(check_weights))

    def perform_validation_audit(self, data_yaml: Path):
        logger.info("Executing network validation verification loops...")
        metrics = self.model.val(data=str(data_yaml), device="mps")
        logger.info(f"Evaluation mAP50-95 Accuracy score: {metrics.box.map:.4f}")
        logger.info(f"Evaluation mAP50 Accuracy score: {metrics.box.map50:.4f}")

if __name__ == "__main__":
    # auditor = CheckoutModelAuditor(Path("./runs/detect/train/weights/best.pt"))
    # auditor.perform_validation_audit(Path("./synthetic_dataset/data.yaml"))
    pass
"""

    # Day 4 Script Content
    day4_code = """# main_day4_optimize.py
import cv2
import numpy as np
import random
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("Day4Augment")

class EnvironmentalStressAugmentor:
    \"\"\"Applies illumination stress to simulate overhead retail environment shadows.\"\"\"
    @staticmethod
    def apply_shadow_overlay(image: np.ndarray, alpha: float = 0.75, beta: int = -35) -> np.ndarray:
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    def optimize_target_classes(self, img_dir: Path, lbl_dir: Path, weak_class_id: int, duplication_rate: int = 2):
        valid_exts = {'.jpg', '.jpeg', '.png'}
        images = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in valid_exts])
        augmented_count = 0
        
        for img_path in images:
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists(): continue
            
            with open(lbl_path, "r") as f:
                lines = f.readlines()
            
            has_weak_target = any(int(line.split()[0]) == weak_class_id for line in lines if line.strip())
            
            if has_weak_target:
                img = cv2.imread(str(img_path))
                if img is None: continue
                
                for step in range(duplication_rate):
                    stressed_img = self.apply_shadow_overlay(img, alpha=random.uniform(0.65, 0.85), beta=random.randint(-45, -20))
                    augmented_stem = f"aug_stress_{step}_{img_path.stem}"
                    
                    cv2.imwrite(str(img_dir / f"{augmented_stem}.jpg"), stressed_img)
                    with open(lbl_dir / f"{augmented_stem}.txt", "w") as lf:
                        lf.writelines(lines)
                augmented_count += 1
        logger.info(f"✨ Custom augmentations complete. Added {augmented_count * duplication_rate} variant fields.")

if __name__ == "__main__":
    pass
"""

    # Day 5 Script Content
    day5_code = """# main_day5_deploy.py
import logging
from pathlib import Path
import gradio as gr
import pandas as pd
from ultralytics import YOLO
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("Day5Deployment")

class AutonomousPOSRegister:
    \"\"\"Compiles transaction balances and displays tracking predictions within an interactive dashboard.\"\"\"
    def __init__(self, model_weights: Path, catalog_prices_csv: Path):
        self.model = YOLO(str(model_weights))
        if catalog_prices_csv.exists():
            df = pd.read_csv(catalog_prices_csv)
            self.price_sheet = dict(zip(df['product_name'], df['price_usd']))
            logger.info("Price reference catalog loaded successfully.")
        else:
            logger.warning("Catalog pricing reference absent. Defaulting to base rate ($1.75).")
            self.price_sheet = {}

    def compile_transaction_bill(self, frame_rgb: np.ndarray) -> tuple:
        if frame_rgb is None:
            return None, "### ⚠️ System Warning\\nNo input feed recognized."

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        results = self.model.predict(source=frame_bgr, conf=0.25, verbose=False)
        result = results[0]
        
        detected_names = [self.model.names.get(int(box.cls[0]), "Unknown") for box in result.boxes]
        confidences = [float(box.conf[0]) for box in result.boxes]
        annotated_frame = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)

        if not detected_names:
            return annotated_frame, "## 🧾 Automated Checkout Statement\\n\\n### 🛒 Basket Status: Empty\\nPlace items on the tray."

        df = pd.DataFrame({"Item": detected_names, "Confidence": confidences})
        summary = df.groupby("Item").agg(Qty=("Item", "size"), AvgConf=("Confidence", "mean")).to_dict(orient="index")

        receipt_md = "## 🧾 Automated Checkout Statement\\n\\n"
        receipt_md += "| Product Class | Quantity | Confidence | Unit Price | Subtotal |\\n| :--- | :---: | :---: | :---: | :---: |\\n"
        
        total_balance = 0.00
        for name, metrics in summary.items():
            unit_price = self.price_sheet.get(name, 1.75)
            subtotal = metrics["Qty"] * unit_price
            total_balance += subtotal
            receipt_md += f"| **{name}** | {metrics['Qty']} | {metrics['AvgConf']:.1%} | ${unit_price:.2f} | ${subtotal:.2f} |\\n"

        receipt_md += f"\\n### 💵 Total Balance Due: ${total_balance:.2f}\\n"
        return annotated_frame, receipt_md

    def launch_interface(self):
        with gr.Blocks(theme=gr.themes.Soft()) as demo:
            gr.Markdown("# 🛒 SmartCart AI Assistant Dashboard")
            with gr.Row():
                with gr.Column():
                    input_img = gr.Image(type="numpy", label="Basket Intake Camera Feed")
                    checkout_btn = gr.Button("Calculate Checkout Transaction", variant="primary")
                with gr.Column():
                    output_img = gr.Image(label="YOLO11 Target Tracking Verification")
                    output_receipt = gr.Markdown(label="Compiled Digital Statement Balance")
            
            checkout_btn.click(
                fn=self.compile_transaction_bill, 
                inputs=input_img, 
                outputs=[output_img, output_receipt]
            )
        demo.launch()

if __name__ == "__main__":
    # app = AutonomousPOSRegister(Path("yolo11n.pt"), Path("./artifacts/catalog_prices.csv"))
    # app.launch_interface()
    pass
"""

    script_mappings = {
        "main_day1_catalog.py": day1_code,
        "main_day2_train.py": day2_code,
        "main_day3_evaluate.py": day3_code,
        "main_day4_optimize.py": day4_code,
        "main_day5_deploy.py": day5_code
    }

    # 4. Generate scripts safely inside the /scripts subfolder
    scripts_folder = base_dir / "scripts"
    
    # Touch __init__.py to allow internal lookups if handled as a package module
    (scripts_folder / "__init__.py").touch()

    for filename, code in script_mappings.items():
        file_path = scripts_folder / filename
        with open(file_path, "w") as f:
            f.write(code.strip() + "\n")
        print(f"📄 Generated script: {file_path}")

    print("\n✅ SmartCart initialization successfully completed! Move into your project folder and begin by executing Day 1.")

if __name__ == "__main__":
    setup_smartcart_architecture()