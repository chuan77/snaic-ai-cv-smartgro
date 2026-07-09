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

```bash
uv run pytest -v          # 161 tests across tests/{data,models,deploy}, 1 deselected @pytest.mark.slow
cd frontend && npm run build && npm run lint
```

There is no linter or CI configuration for the Python side; the test suite above is the only automated check.

## Architecture: two parallel implementations

The codebase has the pipeline logic duplicated in two places — know which one you're editing:

- **Root-level `main_dayN_*.py`** — the live entrypoints. Each is a thin script that imports its logic from `src/` (e.g. `main_day1_catalog.py` imports `UnifiedProductGallery`/`GroceryDatasetIndexer` from `src.data.gallery`; `main_day5_deploy.py` imports `AutonomousPOSRegister` from `src.deploy.register`). This is where day-to-day changes should go.
- **`scripts/main_dayN_*.py`** — self-contained, single-file copies with no dependency on `src/`. Each file embeds its own full copy of the day's classes. These are regenerated verbatim (as string templates) by `boostrap_project.py`'s `setup_smartcart_architecture()` — **running that script overwrites everything in `scripts/`**. Treat them as a frozen/legacy snapshot, not a place to develop.

When asked to modify pipeline behavior, edit `src/` (consumed by the root entrypoints) unless the user specifically points at `scripts/`.

`src/` layout:
- `src/data/` — `catalog.py` (`HardenedProductCatalog`: price registry actually used by `DeterministicGalleryBuilder`; `ProductCatalog`: a simpler alternate, currently unused by any entrypoint), `gallery.py` (`GroceryDatasetIndexer` + `UnifiedProductGallery` — the pair used live by `main_day1_catalog.py`; `EmbeddingGallery` + `DeterministicGalleryBuilder` — an alternate self-discovering embedding pipeline not currently called from any entrypoint), `synthesizer.py` (`ProgrammaticCheckoutSynthesizer`, used live by `main_day2_train.py`), `annotation_import.py` (`import_label_studio_export`/`classify_category` — crops a Label Studio YOLO export and routes each box into the `GroceryStoreDataset` leaf-class tree by longest-matching keyword, behind both `import_annotations.py` and the active-learning pull path), `label_studio_export_pull.py` (`pull_and_import_from_label_studio` — exports a Label Studio project via its API and feeds it through `annotation_import.py`, behind `retrain_from_label_studio.py`).
- `src/models/` — `auditor.py` (`CheckoutModelAuditor`, used live by `main_day3_fine_tune.py`; `perform_validation_audit` also returns `(map50, map50_95)`), `optimizer.py` (`EnvironmentalStressAugmentor`, used live by `main_day4_optimize.py`), `dino_extractor.py` (`DinoFeatureExtractor`, a standalone DINOv2 wrapper not yet wired into `gallery.py`, which still extracts embeddings via its own inline copy of the same backbone/transform), `yolo_detector.py` (`YoloDetector`, the real detector wrapper used live by `src/deploy/detector.py`), `variant_resolver.py` (`VariantResolver` — refines a coarse YOLO detection to its specific catalog variant via DINOv2 nearest-neighbor against the Day 1 gallery, scoped to that detection's coarse-class prefix; used live by `/predict`).
- `src/deploy/` — `api_server.py` (the FastAPI app behind `main_api_server.py`: `GET /health`, `GET /catalog`, `POST /predict`; mounts the Label Studio ML backend router at `/ls` and a `/staging` static mount for active-learning capture review), `detector.py` (`get_detector`/`get_variant_resolver` — `lru_cache`d singletons shared by the checkout API and the Label Studio backend), `label_studio_backend.py` (Label Studio ML Backend routes: `GET /ls/health`, `POST /ls/setup`, `POST /ls/predict`; also owns `get_label_studio_auth_headers()` — see "Active learning / Label Studio integration" below), `active_learning_capture.py` (`maybe_capture` — stages a `/predict` frame + sidecar JSON under `SMARTCART_CAPTURE_DIR` whenever any detection is below `SMARTCART_CAPTURE_CONF_THRESHOLD` or there are zero detections), `label_studio_push.py` (`push_staging_dir`, behind `push_captures_to_label_studio.py` — imports staged captures into a Label Studio project as pre-annotated tasks), `register.py` (`AutonomousPOSRegister`, used live by `main_day5_deploy.py`), `web_dashboard.py` (`SmartCartPresentationApp`, unimplemented stub).
- `src/pipeline/` — `data_augmentor.py` (`AdvancedDataAugmentor`, unimplemented stub), `training_pipeline.py`'s `ModelTrainingPipeline` (implemented — orchestrates rebuild-catalog → resynthesize → retrain → audit for `retrain_from_label_studio.py`; always writes to a fresh `runs/detect/retrain_<timestamp>/` and `synthetic_dataset_retrain/`, never to `runs/detect/train/` or the checkout API's served weights — promotion is manual).

Not part of this project's actual deliverables, deliberately excluded from the pipeline narrative above even though the code is real and live: `src/data/recognition_dataset.py`, `src/models/{recognizer.py,recognition_auditor.py,closed_loop.py}`, `main_day3_recognize.py` (a course-assignment "Day 3 recognition" module — linear head on frozen DINOv2 features). Don't reintroduce it into README/CLAUDE.md pipeline narratives or diagrams unless the user asks for it specifically.

Each `src/<pkg>/__init__.py` re-exports one or two symbols but is not what the entrypoints actually import from (they import the submodule directly, e.g. `src.data.gallery`) — don't assume the `__init__.py` export list reflects what's live.

## Pipeline stages (Day 1–5)

1. **Catalog** (`main_day1_catalog.py`): walk `dataset/GroceryStoreDataset/dataset/train` to build a leaf-class map (fine-grained categories, e.g. `Fruit/Apple/Royal-Gala`), extract frozen DINOv2 (`dinov2_vits14`) embeddings per sample, and write `artifacts/{catalog_prices.csv, gallery_meta.csv, gallery_index.npy}`.
2. **Synthesize + train** (`main_day2_train.py`): composite product crops from the same dataset onto procedurally generated backgrounds (`ProgrammaticCheckoutSynthesizer`) to fabricate bounding-box-labeled scenes under `synthetic_dataset/{train,val}` (YOLO txt format) plus a `data.yaml`, then train `yolo11n.pt` on it (`runs/detect/train`).
3. **Evaluate** (`main_day3_fine_tune.py`): run `YOLO.val()` against `synthetic_dataset/data.yaml` and report mAP50 / mAP50-95. (Named "fine_tune" but currently only evaluates — no fine-tuning step exists yet.)
4. **Optimize** (`main_day4_optimize.py`): for a given weak/underrepresented class id, duplicate its scenes with shadow/illumination stress augmentation to improve robustness.
5. **Deploy** (`main_day5_deploy.py`): `AutonomousPOSRegister` wraps a trained YOLO model + `catalog_prices.csv` behind a Gradio UI — runs detection on an input frame, tallies detected items by class, prices them from the catalog (default $1.75 if unpriced), and renders a markdown receipt.

## Alternate serving surface: FastAPI + React

Besides the Gradio dashboard above, `main_api_server.py` runs a FastAPI backend (`src/deploy/api_server.py`) behind the `frontend/` React/Vite checkout UI. Both are `.env`-configured (`SMARTCART_*` / `VITE_*`, see `.env.example` / `frontend/.env.example`); `.env` is only loaded via `load_dotenv()` inside `api_server.py` itself (so uvicorn's `--reload` worker sees it) and inside the two standalone active-learning scripts below — any *new* standalone script that reads `SMARTCART_*`/`LABEL_STUDIO_*` env vars needs its own explicit `load_dotenv()` call; importing `src/` modules does not trigger it.

## Active learning / Label Studio integration

Closes the loop from live checkout traffic back into a retrained model — see README's "🎯 Active Learning with Label Studio" section for the operator-facing walkthrough and diagram. Implementation notes relevant to changing this code:

- **Label Studio issues refresh tokens, not usable API keys.** Its "New Auth Token" / Personal Access Token dialog gives a JWT with `token_type: "refresh"` — sending it directly as `Authorization: Token <key>` 401s. `get_label_studio_auth_headers()` (`src/deploy/label_studio_backend.py`) exchanges it for a short-lived access token via `POST {LABEL_STUDIO_URL}/api/token/refresh` on every call, then returns `Authorization: Bearer <access>`. All three Label-Studio-facing call sites (`fetch_task_image`, `push_tasks`, `export_project`) go through this helper — don't reintroduce a direct `Token` header.
- **`mark_pushed()` must never move the image file**, only its sidecar JSON (`src/deploy/label_studio_push.py`). Once a capture is pushed, the Label Studio task holds a permanent URL to the image at its original path under `SMARTCART_CAPTURE_DIR` — relocating the file (as an earlier version of this function did) 404s that task forever. The sidecar's absence from the staging root is the only idempotency signal `push_staging_dir` needs; images accumulate at the staging root permanently by design.
- **`/staging`'s CORS policy is nested, not inherited.** Label Studio's own origin loading `$image` into a canvas won't generally match `SMARTCART_CORS_ORIGINS` (that list is for the checkout frontend). `api_server.py` wraps just the `/staging` `StaticFiles` mount in its own `CORSMiddleware(allow_origins=["*"])`, nested inside the app-level restrictive one — `/predict`/`/catalog` still enforce `SMARTCART_CORS_ORIGINS` normally.
- **Testing gotcha:** `label_studio_backend.py`, `label_studio_push.py`, and `label_studio_export_pull.py` all do `import httpx` — that's the *same* module object across files, so `monkeypatch.setattr("mod_a.httpx.post", ...)` and `monkeypatch.setattr("mod_b.httpx.post", ...)` both patch the identical `httpx.post` attribute; the second call silently wins. Tests that need to distinguish a token-refresh call from the "real" call (e.g. `test_push_tasks_posts_to_import_endpoint_with_auth`) use one fake function that branches on URL, not two separate patches.
- `SMARTCART_STAGING_PUBLIC_URL` must match wherever the backend is actually reachable (host/port), including the `/staging` suffix — it's used verbatim as `f"{base}/{image_file}"`, not derived from `SMARTCART_PORT`.

## Data directories

- `dataset/GroceryStoreDataset/` — the raw, unannotated classification-image source (its own git repo).
- `synthetic_dataset/` — generated YOLO-format training data (images + labels + `data.yaml`), not source of truth — regenerate via Day 2.
- `artifacts/` (or root `catalog_prices.csv` / `gallery_index.npy` / `gallery_meta.csv`, depending on which script wrote them) — Day 1 outputs: product catalog and DINOv2 embedding gallery.
- `runs/detect/train/` — Ultralytics YOLO training outputs (weights, logs).
