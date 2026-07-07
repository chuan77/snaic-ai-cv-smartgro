# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

SmartCart AI Checkout Assistant: an autonomous retail checkout system built as a 5-stage pipeline. It turns unannotated grocery classification photos (no bounding boxes) into a working YOLO object detector by programmatically compositing product crops onto synthetic backgrounds, then serves live detections through a Gradio checkout dashboard that prices items from a catalog.

All ML code targets Apple Silicon: device selection is `torch.device("mps" if torch.backends.mps.is_available() else "cpu")` throughout — assume this pattern when adding new inference/training code.

## Setup and running

```bash
uv sync                          # install deps from pyproject.toml / uv.lock (Python >=3.14)
# or: pip install -r requirements.txt

python main_day1_catalog.py      # build product gallery: DINOv2 embeddings + catalog_prices.csv
python main_day2_train.py        # synthesize checkout scenes + train YOLO11 detector
python main_day3_fine_tune.py    # validate trained weights against synthetic_dataset/data.yaml
python main_day4_optimize.py     # augment weak classes with shadow/illumination stress
python main_day5_deploy.py       # launch the Gradio checkout dashboard
```

Day 1 and Day 2 expect `dataset/GroceryStoreDataset` to exist locally (Marcus Klasson's grocery classification dataset, cloned as its own git repo under `dataset/`) — Day 1 logs an error and no-ops if the path is missing rather than raising.

There is no test suite, linter, or CI configuration in this repo currently.

## Architecture: two parallel implementations

The codebase has the pipeline logic duplicated in two places — know which one you're editing:

- **Root-level `main_dayN_*.py`** — the live entrypoints. Each is a thin script that imports its logic from `src/` (e.g. `main_day1_catalog.py` imports `UnifiedProductGallery`/`GroceryDatasetIndexer` from `src.data.gallery`; `main_day5_deploy.py` imports `AutonomousPOSRegister` from `src.deploy.register`). This is where day-to-day changes should go.
- **`scripts/main_dayN_*.py`** — self-contained, single-file copies with no dependency on `src/`. Each file embeds its own full copy of the day's classes. These are regenerated verbatim (as string templates) by `boostrap_project.py`'s `setup_smartcart_architecture()` — **running that script overwrites everything in `scripts/`**. Treat them as a frozen/legacy snapshot, not a place to develop.

When asked to modify pipeline behavior, edit `src/` (consumed by the root entrypoints) unless the user specifically points at `scripts/`.

`src/` layout:
- `src/data/` — `catalog.py` (`HardenedProductCatalog`: price registry actually used by `DeterministicGalleryBuilder`; `ProductCatalog`: a simpler alternate, currently unused by any entrypoint), `gallery.py` (`GroceryDatasetIndexer` + `UnifiedProductGallery` — the pair used live by `main_day1_catalog.py`; `EmbeddingGallery` + `DeterministicGalleryBuilder` — an alternate self-discovering embedding pipeline not currently called from any entrypoint), `synthesizer.py` (`ProgrammaticCheckoutSynthesizer`, used live by `main_day2_train.py`).
- `src/models/` — `auditor.py` (`CheckoutModelAuditor`, used live by `main_day3_fine_tune.py`), `optimizer.py` (`EnvironmentalStressAugmentor`, used live by `main_day4_optimize.py`), `dino_extractor.py` (`DinoFeatureExtractor`, a standalone DINOv2 wrapper not yet wired into `gallery.py`, which still extracts embeddings via its own inline copy of the same backbone/transform), `yolo_detector.py` (`YoloDetector`, unimplemented stub).
- `src/deploy/` — `register.py` (`AutonomousPOSRegister`, used live by `main_day5_deploy.py`), `web_dashboard.py` (`SmartCartPresentationApp`, unimplemented stub).
- `src/pipeline/` — `data_augmentor.py` (`AdvancedDataAugmentor`, unimplemented stub), `training_pipeline.py` (`ModelTrainingPipeline`, unimplemented stub).

Each `src/<pkg>/__init__.py` re-exports one or two symbols but is not what the entrypoints actually import from (they import the submodule directly, e.g. `src.data.gallery`) — don't assume the `__init__.py` export list reflects what's live.

## Pipeline stages (Day 1–5)

1. **Catalog** (`main_day1_catalog.py`): walk `dataset/GroceryStoreDataset/dataset/train` to build a leaf-class map (fine-grained categories, e.g. `Fruit/Apple/Royal-Gala`), extract frozen DINOv2 (`dinov2_vits14`) embeddings per sample, and write `artifacts/{catalog_prices.csv, gallery_meta.csv, gallery_index.npy}`.
2. **Synthesize + train** (`main_day2_train.py`): composite product crops from the same dataset onto procedurally generated backgrounds (`ProgrammaticCheckoutSynthesizer`) to fabricate bounding-box-labeled scenes under `synthetic_dataset/{train,val}` (YOLO txt format) plus a `data.yaml`, then train `yolo11n.pt` on it (`runs/detect/train`).
3. **Evaluate** (`main_day3_fine_tune.py`): run `YOLO.val()` against `synthetic_dataset/data.yaml` and report mAP50 / mAP50-95. (Named "fine_tune" but currently only evaluates — no fine-tuning step exists yet.)
4. **Optimize** (`main_day4_optimize.py`): for a given weak/underrepresented class id, duplicate its scenes with shadow/illumination stress augmentation to improve robustness.
5. **Deploy** (`main_day5_deploy.py`): `AutonomousPOSRegister` wraps a trained YOLO model + `catalog_prices.csv` behind a Gradio UI — runs detection on an input frame, tallies detected items by class, prices them from the catalog (default $1.75 if unpriced), and renders a markdown receipt.

## Data directories

- `dataset/GroceryStoreDataset/` — the raw, unannotated classification-image source (its own git repo).
- `synthetic_dataset/` — generated YOLO-format training data (images + labels + `data.yaml`), not source of truth — regenerate via Day 2.
- `artifacts/` (or root `catalog_prices.csv` / `gallery_index.npy` / `gallery_meta.csv`, depending on which script wrote them) — Day 1 outputs: product catalog and DINOv2 embedding gallery.
- `runs/detect/train/` — Ultralytics YOLO training outputs (weights, logs).
