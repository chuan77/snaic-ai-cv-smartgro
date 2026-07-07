# 🛒 SmartCart AI Checkout Assistant

An autonomous retail checkout system utilizing **Programmatic Scene Synthesis** to transform unannotated classification imagery into a robust object detection and billing pipeline.

## 🚀 Quick Start
1. **Initialize Project:** Run `python boostrap_project.py` to generate the directory structure (also (re)writes the standalone `scripts/` copies — see Architecture below).
2. **Install Dependencies:**
   `pip install -r requirements.txt` (or `uv sync`)
3. **Clone the dataset:** clone [Marcus Klasson's GroceryStoreDataset](https://github.com/marcusklasson/GroceryStoreDataset) into `dataset/GroceryStoreDataset`.
4. **Run Pipeline:**
   Execute the day scripts in sequence from the root:
   ```bash
   python main_day1_catalog.py
   python main_day2_train.py
   python main_day3_fine_tune.py
   python main_day4_optimize.py
   python main_day5_deploy.py
   ```

## 🏗️ Project Architecture
The project is modularized into `src/` (core logic) and root-level `main_dayN_*.py` scripts (orchestration) — each day's script is a thin entrypoint that imports its logic from `src/`.

A second, self-contained copy of the same logic lives under `scripts/main_dayN_*.py`; those files embed their own classes with no `src/` dependency and are regenerated wholesale by `boostrap_project.py`. Develop against the root scripts + `src/`; treat `scripts/` as a frozen snapshot.

- **Data Engineering:** Uses synthetic composition to solve the bounding-box annotation gap.
- **Vision Backbone:** Employs frozen DINOv2 embeddings for high-fidelity product indexing.
- **Detector:** Utilizes YOLOv11 optimized for MPS/Apple Silicon.

## 🛠️ Folder Structure
- `src/data/` — dataset indexing, DINOv2 embedding gallery, scene synthesis.
- `src/models/` — YOLO evaluation (auditor), stress augmentation (optimizer), DINOv2 feature extraction.
- `src/deploy/` — the Gradio checkout register.
- `src/pipeline/` — reserved for a unified training pipeline (not yet implemented).
- `dataset/`: Raw GroceryStoreDataset source.
- `synthetic_dataset/`: Procedurally generated training/val scenes.
- `artifacts/`: Day 1 outputs (product catalog, embedding gallery).
- `runs/`: Training logs and model weights.