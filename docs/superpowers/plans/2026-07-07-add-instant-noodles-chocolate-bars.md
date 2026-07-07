# Add Instant Noodles & Chocolate Bars Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two new detector classes — Instant Noodles and Chocolate Bars — to the SmartCart pipeline, sourced from the user's own Label-Studio-annotated photos, and produce a retrained YOLO detector covering all 5 classes.

**Architecture:** A new import script crops Label-Studio-annotated photos down to their product bounding box and drops them into the existing `GroceryStoreDataset/dataset/train/` tree using the same "leaf directory full of images" convention Day 1's indexer already expects. The existing Day 1–3 scripts pick the new classes up with two small edits: a category-aware pricing rule in `gallery.py`, and an expanded class map in `main_day2_train.py`. Because this codebase has no incremental/continual-learning path for YOLO (class count is baked into trained weights), "fine-tuning" the detector means a full retrain from `yolo11n.pt` with the expanded 5-class map — the same mechanism Day 2 already uses for the current 3 classes.

**Tech Stack:** Python (>=3.14), Pillow (PIL) for image cropping, pytest for unit tests, Ultralytics YOLO for training/eval — all already in `pyproject.toml`.

## Global Constraints

- Python >=3.14 (per `pyproject.toml`); use modern type hints (`tuple[int, int, int, int]`, `list[Path]`), not `typing.Tuple`/`typing.List`.
- Device selection for any ML code follows `torch.device("mps" if torch.backends.mps.is_available() else "cpu")` (per `CLAUDE.md`) — not needed in this plan's new code since cropping is CPU-only PIL work, but do not deviate if touched.
- Edit `src/` and root `main_dayN_*.py` files only — never touch `scripts/` (regenerated verbatim by `boostrap_project.py`, treated as a frozen legacy snapshot per `CLAUDE.md`).
- Tests live under `tests/`, run via `uv run pytest <path> -v`; `pyproject.toml` already sets `testpaths = ["tests"]` and defaults to skipping the `slow` marker (`addopts = "-m 'not slow'"`).
- Match existing test style: plain pytest functions/parametrize, `unittest.mock.patch` for anything touching real I/O-heavy libs (see `tests/models/test_yolo_detector.py`) — no new test framework or fixtures library.
- `dataset/GroceryStoreDataset/` is its own nested git repo (per `CLAUDE.md`) — new staging/raw-photo directories must live outside it (i.e. under `dataset/` but not under `dataset/GroceryStoreDataset/`), only the final cropped images go inside it.
- Day 1 (`GroceryDatasetIndexer.build_class_map`, `src/data/gallery.py:80-90`) discovers a "class" as any directory (at any depth) with image files (`.png/.jpg/.jpeg/.webp`) directly inside it — no code change needed there for discovery itself.

---

### Task 1: Stage raw-photo directories + Label Studio annotation instructions

**Files:**
- Create: `dataset/raw_photos/Instant-Noodles/.gitkeep`
- Create: `dataset/raw_photos/Chocolate-Bar/.gitkeep`
- Create: `dataset/raw_photos/README.md`

**Interfaces:**
- Produces: the two staging directory paths that Task 4's `import_annotations.py` reads its Label Studio exports from — `dataset/raw_photos/Instant-Noodles/labelstudio_export/` and `dataset/raw_photos/Chocolate-Bar/labelstudio_export/`.

- [ ] **Step 1: Create the staging directories**

```bash
mkdir -p dataset/raw_photos/Instant-Noodles
mkdir -p dataset/raw_photos/Chocolate-Bar
touch dataset/raw_photos/Instant-Noodles/.gitkeep
touch dataset/raw_photos/Chocolate-Bar/.gitkeep
```

- [ ] **Step 2: Write the annotation instructions**

Create `dataset/raw_photos/README.md`:

```markdown
# Raw photo staging + Label Studio annotation

This directory holds product photos *before* they're cropped and merged into
`dataset/GroceryStoreDataset/dataset/train/`. It is intentionally outside the
`GroceryStoreDataset/` folder, which is its own git repo.

## 1. Drop in raw photos

- `dataset/raw_photos/Instant-Noodles/` — one photo per instant-noodle item,
  single product per photo, any background.
- `dataset/raw_photos/Chocolate-Bar/` — one photo per chocolate-bar item, same rule.

## 2. Annotate in Label Studio

1. Install/run Label Studio (`pip install label-studio && label-studio start`,
   or your existing instance).
2. Create one project per category (or one project with two labels — either
   works, since each category is imported separately in Task 4).
3. Import the photos from the matching `dataset/raw_photos/<Category>/` folder.
4. Labeling config: a single rectangle label per photo, drawn tightly around
   the product. Exactly one bounding box per image — the import script in
   this repo only reads the first box in each label file.
5. Export the finished annotations using Label Studio's **YOLO** export
   format. This produces an `images/` folder, a `labels/` folder (one `.txt`
   per image, `class_id cx cy w h` normalized), and a `classes.txt`.
6. Unzip/place the export here:
   - `dataset/raw_photos/Instant-Noodles/labelstudio_export/{images,labels}/`
   - `dataset/raw_photos/Chocolate-Bar/labelstudio_export/{images,labels}/`

## 3. Import

Run `python import_annotations.py` (see repo root) to crop each annotated
photo to its bounding box and copy it into
`dataset/GroceryStoreDataset/dataset/train/Ready-To-Eat/Instant-Noodles/` and
`dataset/GroceryStoreDataset/dataset/train/Snacks/Chocolate-Bar/` respectively.
```

- [ ] **Step 3: Commit**

```bash
git add dataset/raw_photos/
git commit -m "docs: add raw-photo staging dirs and Label Studio annotation instructions"
```

---

### Task 2: Implement bbox-parsing and cropping helpers (TDD)

**Files:**
- Create: `src/data/annotation_import.py`
- Create: `tests/data/__init__.py`
- Create: `tests/data/test_annotation_import.py`

**Interfaces:**
- Produces: `parse_yolo_bbox_line(line: str, img_width: int, img_height: int) -> tuple[int, int, int, int]` (returns pixel `(x_min, y_min, x_max, y_max)`, clamped to `[0, img_width] x [0, img_height]`).
- Produces: `crop_with_padding(image: PIL.Image.Image, box: tuple[int, int, int, int], padding_ratio: float = 0.05) -> PIL.Image.Image`.

- [ ] **Step 1: Create the test package marker**

```bash
mkdir -p tests/data
touch tests/data/__init__.py
```

- [ ] **Step 2: Write the failing tests**

Create `tests/data/test_annotation_import.py`:

```python
from PIL import Image

from src.data.annotation_import import parse_yolo_bbox_line, crop_with_padding


def test_parse_yolo_bbox_line_converts_normalized_center_to_pixel_xyxy():
    box = parse_yolo_bbox_line("0 0.5 0.5 0.4 0.2", img_width=100, img_height=100)

    assert box == (30, 40, 70, 60)


def test_parse_yolo_bbox_line_clamps_to_image_bounds():
    box = parse_yolo_bbox_line("0 0.05 0.05 0.5 0.5", img_width=100, img_height=100)

    assert box == (0, 0, 30, 30)


def test_crop_with_padding_expands_box_by_padding_ratio():
    image = Image.new("RGB", (100, 100))

    cropped = crop_with_padding(image, box=(40, 40, 60, 60), padding_ratio=0.5)

    assert cropped.size == (40, 40)


def test_crop_with_padding_clamps_to_image_bounds():
    image = Image.new("RGB", (100, 100))

    cropped = crop_with_padding(image, box=(0, 0, 10, 10), padding_ratio=1.0)

    assert cropped.size == (20, 20)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/data/test_annotation_import.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data.annotation_import'`

- [ ] **Step 4: Implement the helpers**

Create `src/data/annotation_import.py`:

```python
"""Imports Label Studio YOLO-format exports into the GroceryStoreDataset leaf-class tree."""
from pathlib import Path

from PIL import Image


def parse_yolo_bbox_line(line: str, img_width: int, img_height: int) -> tuple[int, int, int, int]:
    """Converts one YOLO-format label line ('class_id cx cy w h', normalized) to pixel xyxy."""
    _, cx, cy, w, h = line.split()
    cx, cy, w, h = float(cx), float(cy), float(w), float(h)
    x_min = int((cx - w / 2) * img_width)
    y_min = int((cy - h / 2) * img_height)
    x_max = int((cx + w / 2) * img_width)
    y_max = int((cy + h / 2) * img_height)
    return (
        max(0, x_min),
        max(0, y_min),
        min(img_width, x_max),
        min(img_height, y_max),
    )


def crop_with_padding(
    image: Image.Image, box: tuple[int, int, int, int], padding_ratio: float = 0.05
) -> Image.Image:
    """Crops `image` to `box`, expanded by `padding_ratio` of the box's own width/height."""
    x_min, y_min, x_max, y_max = box
    pad_x = int((x_max - x_min) * padding_ratio)
    pad_y = int((y_max - y_min) * padding_ratio)
    padded_box = (
        max(0, x_min - pad_x),
        max(0, y_min - pad_y),
        min(image.width, x_max + pad_x),
        min(image.height, y_max + pad_y),
    )
    return image.crop(padded_box)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/data/test_annotation_import.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add src/data/annotation_import.py tests/data/__init__.py tests/data/test_annotation_import.py
git commit -m "feat: add YOLO bbox parsing and padded cropping helpers"
```

---

### Task 3: Implement the Label Studio export driver (TDD)

**Files:**
- Modify: `src/data/annotation_import.py`
- Modify: `tests/data/test_annotation_import.py`

**Interfaces:**
- Consumes: `parse_yolo_bbox_line`, `crop_with_padding` from Task 2 (same file).
- Produces: `import_label_studio_export(export_dir: Path, dest_dir: Path) -> list[Path]` — reads `export_dir/images/` + `export_dir/labels/`, writes one cropped image per label file into `dest_dir` (creating it if needed), returns the list of written paths.

- [ ] **Step 1: Write the failing tests**

Append to `tests/data/test_annotation_import.py`:

```python
from src.data.annotation_import import import_label_studio_export


def test_import_label_studio_export_writes_cropped_images_and_returns_paths(tmp_path):
    export_dir = tmp_path / "export"
    images_dir = export_dir / "images"
    labels_dir = export_dir / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    Image.new("RGB", (100, 100), color=(255, 0, 0)).save(images_dir / "sample_001.jpg")
    (labels_dir / "sample_001.txt").write_text("0 0.5 0.5 0.4 0.4\n")
    dest_dir = tmp_path / "dest"

    written = import_label_studio_export(export_dir, dest_dir)

    assert written == [dest_dir / "sample_001.jpg"]
    with Image.open(dest_dir / "sample_001.jpg") as cropped:
        assert cropped.size == (44, 44)


def test_import_label_studio_export_skips_label_files_without_matching_image(tmp_path):
    export_dir = tmp_path / "export"
    (export_dir / "images").mkdir(parents=True)
    labels_dir = export_dir / "labels"
    labels_dir.mkdir(parents=True)
    (labels_dir / "orphan.txt").write_text("0 0.5 0.5 0.4 0.4\n")

    written = import_label_studio_export(export_dir, tmp_path / "dest")

    assert written == []


def test_import_label_studio_export_skips_empty_label_files(tmp_path):
    export_dir = tmp_path / "export"
    images_dir = export_dir / "images"
    labels_dir = export_dir / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    Image.new("RGB", (50, 50)).save(images_dir / "empty_001.jpg")
    (labels_dir / "empty_001.txt").write_text("")

    written = import_label_studio_export(export_dir, tmp_path / "dest")

    assert written == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/data/test_annotation_import.py -v`
Expected: the 3 new tests FAIL with `ImportError: cannot import name 'import_label_studio_export'`

- [ ] **Step 3: Implement the driver function**

Append to `src/data/annotation_import.py`:

```python
def import_label_studio_export(export_dir: Path, dest_dir: Path) -> list[Path]:
    """Reads a Label Studio YOLO-format export and writes one cropped photo per
    label file into dest_dir, matching images to labels by filename stem."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    images_dir = export_dir / "images"
    labels_dir = export_dir / "labels"
    written: list[Path] = []

    for label_path in sorted(labels_dir.glob("*.txt")):
        lines = label_path.read_text().strip().splitlines()
        if not lines:
            continue

        image_path = next(
            (candidate for ext in (".jpg", ".jpeg", ".png", ".webp")
             if (candidate := images_dir / f"{label_path.stem}{ext}").exists()),
            None,
        )
        if image_path is None:
            continue

        with Image.open(image_path).convert("RGB") as img:
            box = parse_yolo_bbox_line(lines[0], img.width, img.height)
            cropped = crop_with_padding(img, box)
            dest_path = dest_dir / image_path.name
            cropped.save(dest_path)
            written.append(dest_path)

    return written
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/data/test_annotation_import.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/data/annotation_import.py tests/data/test_annotation_import.py
git commit -m "feat: add Label Studio YOLO export import driver"
```

---

### Task 4: Add the `import_annotations.py` entrypoint

**Files:**
- Create: `import_annotations.py`

**Interfaces:**
- Consumes: `import_label_studio_export(export_dir: Path, dest_dir: Path) -> list[Path]` from Task 3.

- [ ] **Step 1: Write the entrypoint**

Create `import_annotations.py` (repo root, matching the existing `main_dayN_*.py` thin-entrypoint convention):

```python
"""Imports Label Studio-annotated product photos into the GroceryStoreDataset tree
so Day 1's GroceryDatasetIndexer discovers them as new classes."""
from pathlib import Path

from src.data.annotation_import import import_label_studio_export

if __name__ == "__main__":
    noodle_written = import_label_studio_export(
        Path("./dataset/raw_photos/Instant-Noodles/labelstudio_export"),
        Path("./dataset/GroceryStoreDataset/dataset/train/Ready-To-Eat/Instant-Noodles"),
    )
    choc_written = import_label_studio_export(
        Path("./dataset/raw_photos/Chocolate-Bar/labelstudio_export"),
        Path("./dataset/GroceryStoreDataset/dataset/train/Snacks/Chocolate-Bar"),
    )
    print(f"Imported {len(noodle_written)} Instant-Noodles crops, {len(choc_written)} Chocolate-Bar crops.")
```

- [ ] **Step 2: Commit**

```bash
git add import_annotations.py
git commit -m "feat: add import_annotations.py entrypoint for new product categories"
```

*(This script is exercised for real in Task 7, once real annotated photos exist — there's nothing to unit-test here beyond what Tasks 2–3 already cover.)*

---

### Task 5: Category-aware pricing rule (TDD)

**Files:**
- Modify: `src/data/gallery.py`
- Create: `tests/data/test_gallery.py`

**Interfaces:**
- Produces: `get_base_price(class_str: str) -> float` in `src/data/gallery.py`, used by `UnifiedProductGallery.compile_gallery`.

- [ ] **Step 1: Write the failing tests**

Create `tests/data/test_gallery.py`:

```python
import pytest

from src.data.gallery import get_base_price


@pytest.mark.parametrize("class_str,expected_price", [
    ("Packages/Milk/Arla-Standard-Milk", 3.20),
    ("Packages/Juice/Bravo-Apple-Juice", 3.20),
    ("Ready-To-Eat/Instant-Noodles", 3.00),
    ("Snacks/Chocolate-Bar", 6.20),
    ("Fruit/Apple/Royal-Gala", 1.75),
    ("Vegetables/Carrots", 1.75),
])
def test_get_base_price_maps_top_level_category_to_expected_price(class_str, expected_price):
    assert get_base_price(class_str) == expected_price
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/data/test_gallery.py -v`
Expected: FAIL with `ImportError: cannot import name 'get_base_price'`

- [ ] **Step 3: Implement the pricing function and wire it in**

Add to `src/data/gallery.py`, above the `UnifiedProductGallery` class:

```python
def get_base_price(class_str: str) -> float:
    """Maps a leaf class's dataset-relative path to its default catalog price by top-level category."""
    category_prices = {
        "Packages": 3.20,
        "Ready-To-Eat": 3.00,
        "Snacks": 6.20,
    }
    top_level = class_str.split("/")[0]
    return category_prices.get(top_level, 1.75)
```

Replace `src/data/gallery.py:51`:

```python
            base_price = 3.20 if "Packages" in class_str else 1.75
```

with:

```python
            base_price = get_base_price(class_str)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/data/test_gallery.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/data/gallery.py tests/data/test_gallery.py
git commit -m "feat: add category-aware pricing rule for Ready-To-Eat and Snacks"
```

---

### Task 6: Extend the Day 2 detector class map and fix the retrain run-dir path

**Files:**
- Modify: `main_day2_train.py`

**Interfaces:**
- Consumes: the leaf-class paths `Ready-To-Eat/Instant-Noodles` and `Snacks/Chocolate-Bar` that Task 4's import writes into `dataset/GroceryStoreDataset/dataset/train/`.

- [ ] **Step 1: Extend the class map**

In `main_day2_train.py`, replace:

```python
    cmap = {"Fruit/Apple/Royal-Gala": 0, "Vegetables/Carrots": 1, "Packages/Milk/Arla-Standard-Milk": 2}
```

with:

```python
    cmap = {
        "Fruit/Apple/Royal-Gala": 0,
        "Vegetables/Carrots": 1,
        "Packages/Milk/Arla-Standard-Milk": 2,
        "Ready-To-Eat/Instant-Noodles": 3,
        "Snacks/Chocolate-Bar": 4,
    }
```

- [ ] **Step 2: Make the retrain overwrite the same run directory**

Ultralytics' `model.train()` auto-increments the run folder (`train2`, `train3`, ...) unless told otherwise, but `main_day3_fine_tune.py` hardcodes `runs/detect/train/weights/best.pt`. Replace:

```python
    model.train(data=str(ypath), epochs=20, imgsz=640, device="mps", workers=2)
```

with:

```python
    model.train(data=str(ypath), epochs=20, imgsz=640, device="mps", workers=2, name="train", exist_ok=True)
```

- [ ] **Step 3: Commit**

```bash
git add main_day2_train.py
git commit -m "feat: expand detector to 5 classes and pin retrain output to runs/detect/train"
```

*(No unit test here — `main_day2_train.py` is a thin, untested entrypoint script like the rest of the `main_dayN_*.py` files. It's exercised end-to-end in Task 7.)*

---

### Task 7: End-to-end regeneration and validation

**Files:** none (execution/verification only — depends on Tasks 1–6 plus real annotated photos)

**Interfaces:** none — this task drives the existing `main_day1_catalog.py`, `main_day2_train.py`, `main_day3_fine_tune.py`, `main_day5_deploy.py` entrypoints end-to-end.

**Blocking dependency:** this task cannot run until the user has completed the manual Label Studio annotation step from Task 1 and placed the exports at `dataset/raw_photos/Instant-Noodles/labelstudio_export/` and `dataset/raw_photos/Chocolate-Bar/labelstudio_export/`.

- [ ] **Step 1: Import the annotated crops**

Run: `python import_annotations.py`
Expected: prints `Imported N Instant-Noodles crops, M Chocolate-Bar crops.` with N, M > 0. Spot-check a few files:

```bash
ls dataset/GroceryStoreDataset/dataset/train/Ready-To-Eat/Instant-Noodles/
ls dataset/GroceryStoreDataset/dataset/train/Snacks/Chocolate-Bar/
```

- [ ] **Step 2: Rebuild the Day 1 catalog/gallery**

Run: `python main_day1_catalog.py`
Expected: no errors; then verify the new classes and prices landed:

```bash
grep -E "Ready-To-Eat/Instant-Noodles|Snacks/Chocolate-Bar" artifacts/catalog_prices.csv
```

Expected output: one row per new class, with `3.0` and `6.2` respectively in the `price_usd` column.

- [ ] **Step 3: Regenerate the synthetic dataset and retrain the detector**

Run: `python main_day2_train.py`
Expected: no errors; training runs for 20 epochs on `device=mps`. Verify the class list:

```bash
cat synthetic_dataset/data.yaml
```

Expected: `nc: 5` and `names` including `3: Ready-To-Eat/Instant-Noodles` and `4: Snacks/Chocolate-Bar`. Verify the run directory was overwritten in place (not `train2`):

```bash
ls runs/detect/
```

Expected: only `train/` — no `train2/`.

- [ ] **Step 4: Validate the retrained detector**

Run: `python main_day3_fine_tune.py`
Expected: mAP50 / mAP50-95 printed for all 5 classes. Sanity-check that Instant-Noodles and Chocolate-Bar aren't dramatically worse than the original 3 classes — if one is far weaker, consider a follow-up run of `main_day4_optimize.py` with `weak_class_id=3` or `weak_class_id=4` before retraining again (optional, not required for this plan).

- [ ] **Step 5: Manually verify the deployed dashboard**

Run: `python main_day5_deploy.py`
In the Gradio UI, upload or capture a photo containing an instant-noodle and/or chocolate-bar item and confirm:
- the detector draws a bounding box around it,
- the receipt lists it by name (`Ready-To-Eat/Instant-Noodles` or `Snacks/Chocolate-Bar`),
- the price shown is `$3.00` / `$6.20` respectively.

- [ ] **Step 6: Commit the code-level outcome**

The retrained weights (`runs/detect/train/weights/*.pt`), the regenerated `synthetic_dataset/`, and `artifacts/` are large/generated outputs — do not commit them unless the project's `.gitignore` already excludes them and the user wants them tracked. Confirm with the user before committing anything under `runs/`, `synthetic_dataset/`, or `artifacts/`.
