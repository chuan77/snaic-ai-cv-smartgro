# FastAPI Detection Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a real FastAPI backend that serves the trained YOLO detector and catalog data, then rewire the frontend's mocked `api.ts` to call it, so dragging in or capturing a real product photo produces a real detection with correct name/price.

**Architecture:** A new `src/deploy/api_server.py` module hosts a FastAPI app with `GET /catalog` and `POST /predict`, reusing the existing `YoloDetector` (`src/models/yolo_detector.py`) for inference. A thin `main_api_server.py` entrypoint runs it via `uvicorn`. `frontend/src/lib/api.ts` swaps its mock implementation for real `fetch` calls against this backend.

**Tech Stack:** FastAPI, uvicorn, python-multipart (Python backend); existing `fetch`/Vite env vars (frontend, no new deps).

## Global Constraints

- `GET /catalog` returns `[{sku, name, priceUsd}]`; `sku` must equal the trained model's `model.names` string values exactly (verified via `src/data/gallery.py`'s class-path convention, e.g. `"Ready-To-Eat/Instant-Noodles"`).
- `POST /predict` returns `[{id, label, confidence, bbox}]` where `bbox` is fractional `[x, y, w, h]`, top-left origin (per `frontend/src/components/DetectionOverlay.tsx:26-31`) — never the pixel `xyxy` that `YoloDetector.detect()` returns natively.
- `addItem`/`removeItem` are dropped from `ApiClient` entirely — confirmed unused anywhere in `frontend/src/`.
- CORS allows `http://localhost:5173` (Vite's default dev origin) — no dev proxy.
- Weights path defaults to `./runs/detect/train/weights/best.pt`, overridable via `SMARTCART_WEIGHTS_PATH` env var. Catalog path is fixed at `./artifacts/catalog_prices.csv` (not configurable).
- New code follows this repo's existing conventions: `src/` holds the real logic, a thin root `main_*.py` script is the entrypoint, tests live under `tests/<pkg>/test_*.py` using plain pytest + `unittest.mock`/`app.dependency_overrides` (no new test framework, no Pydantic models — plain dicts, matching this codebase's existing lightweight style).

---

### Task 1: Add backend dependencies

**Files:**
- Modify: `pyproject.toml`

**Interfaces:** none (dependency setup only).

- [ ] **Step 1: Add fastapi, uvicorn, python-multipart, and httpx to pyproject.toml**

In `pyproject.toml`, replace:

```toml
dependencies = [
    "gradio>=4.0.0",
    "numpy>=1.24.0",
    "opencv-python>=4.7.0",
    "pandas>=2.0.0",
    "pillow>=9.5.0",
    "pyyaml>=6.0",
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "typing>=3.10.0.0",
    "ultralytics>=8.1.0",
]

[dependency-groups]
dev = [
    "pytest>=9.1.1",
]
```

with:

```toml
dependencies = [
    "fastapi>=0.115.0",
    "gradio>=4.0.0",
    "numpy>=1.24.0",
    "opencv-python>=4.7.0",
    "pandas>=2.0.0",
    "pillow>=9.5.0",
    "python-multipart>=0.0.9",
    "pyyaml>=6.0",
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "typing>=3.10.0.0",
    "ultralytics>=8.1.0",
    "uvicorn>=0.30.0",
]

[dependency-groups]
dev = [
    "httpx>=0.27.0",
    "pytest>=9.1.1",
]
```

- [ ] **Step 2: Install and verify**

Run: `uv sync`
Expected: completes with no errors, and `fastapi`/`uvicorn`/`httpx` appear in `uv.lock`.

Run: `uv run python -c "import fastapi, uvicorn, httpx; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat: add fastapi/uvicorn backend dependencies"
```

---

### Task 2: Pure conversion helpers (TDD)

**Files:**
- Create: `src/deploy/api_server.py`
- Create: `tests/deploy/__init__.py`
- Create: `tests/deploy/test_api_server.py`

**Interfaces:**
- Produces: `leaf_display_name(product_name: str) -> str`
- Produces: `xyxy_to_fractional_xywh(box: list[float], img_width: int, img_height: int) -> tuple[float, float, float, float]`

- [ ] **Step 1: Create the test package marker**

```bash
mkdir -p tests/deploy
touch tests/deploy/__init__.py
```

- [ ] **Step 2: Write the failing tests**

Create `tests/deploy/test_api_server.py`:

```python
import pytest

from src.deploy.api_server import leaf_display_name, xyxy_to_fractional_xywh


@pytest.mark.parametrize("product_name,expected", [
    ("Fruit/Apple/Royal-Gala", "Royal Gala"),
    ("Vegetables/Carrots", "Carrots"),
    ("Ready-To-Eat/Instant-Noodles", "Instant Noodles"),
    ("Snacks/Chocolate-Bar", "Chocolate Bar"),
])
def test_leaf_display_name_derives_human_readable_label(product_name, expected):
    assert leaf_display_name(product_name) == expected


def test_xyxy_to_fractional_xywh_converts_pixel_box_to_fractional_top_left_xywh():
    result = xyxy_to_fractional_xywh([50.0, 100.0, 150.0, 200.0], img_width=200, img_height=400)

    assert result == pytest.approx((0.25, 0.25, 0.5, 0.25))
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/deploy/test_api_server.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.deploy.api_server'`

- [ ] **Step 4: Implement the helpers**

Create `src/deploy/api_server.py`:

```python
"""FastAPI backend serving real YOLO detections and catalog data to the frontend."""


def leaf_display_name(product_name: str) -> str:
    """Derives a human-readable label from a dataset-relative class path, e.g.
    'Fruit/Apple/Royal-Gala' -> 'Royal Gala'."""
    leaf = product_name.split("/")[-1]
    return leaf.replace("-", " ").title()


def xyxy_to_fractional_xywh(
    box: list[float], img_width: int, img_height: int
) -> tuple[float, float, float, float]:
    """Converts a pixel xyxy box to a fractional top-left-origin (x, y, w, h) box."""
    x_min, y_min, x_max, y_max = box
    return (
        x_min / img_width,
        y_min / img_height,
        (x_max - x_min) / img_width,
        (y_max - y_min) / img_height,
    )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/deploy/test_api_server.py -v`
Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add src/deploy/api_server.py tests/deploy/__init__.py tests/deploy/test_api_server.py
git commit -m "feat: add bbox/display-name conversion helpers for the detection backend"
```

---

### Task 3: Catalog endpoint (TDD)

**Files:**
- Modify: `src/deploy/api_server.py`
- Modify: `tests/deploy/test_api_server.py`

**Interfaces:**
- Consumes: `leaf_display_name` from Task 2 (same file).
- Produces: `load_catalog(csv_path: Path) -> list[dict]`, `get_catalog() -> list[dict]` (FastAPI dependency), the module-level `app` (`FastAPI` instance) with `GET /catalog` registered.

- [ ] **Step 1: Write the failing tests**

Append to `tests/deploy/test_api_server.py`:

```python
from fastapi.testclient import TestClient

from src.deploy.api_server import app, get_catalog, load_catalog


def test_load_catalog_reads_csv_and_derives_display_name(tmp_path):
    csv_path = tmp_path / "catalog_prices.csv"
    csv_path.write_text(
        "class_id,product_name,price_usd\n"
        "0,Fruit/Apple/Royal-Gala,1.75\n"
        "1,Snacks/Chocolate-Bar,6.2\n"
    )

    catalog = load_catalog(csv_path)

    assert catalog == [
        {"sku": "Fruit/Apple/Royal-Gala", "name": "Royal Gala", "priceUsd": 1.75},
        {"sku": "Snacks/Chocolate-Bar", "name": "Chocolate Bar", "priceUsd": 6.2},
    ]


def test_catalog_endpoint_returns_the_overridden_catalog():
    app.dependency_overrides[get_catalog] = lambda: [
        {"sku": "Fruit/Apple/Royal-Gala", "name": "Royal Gala", "priceUsd": 1.75}
    ]
    client = TestClient(app)

    response = client.get("/catalog")

    assert response.status_code == 200
    assert response.json() == [{"sku": "Fruit/Apple/Royal-Gala", "name": "Royal Gala", "priceUsd": 1.75}]
    app.dependency_overrides.clear()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/deploy/test_api_server.py -v`
Expected: the 2 new tests FAIL with `ImportError: cannot import name 'app'`

- [ ] **Step 3: Implement the catalog endpoint**

Add to the top of `src/deploy/api_server.py` (after the module docstring):

```python
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

DEFAULT_WEIGHTS_PATH = Path("./runs/detect/train/weights/best.pt")
DEFAULT_CATALOG_PATH = Path("./artifacts/catalog_prices.csv")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Then append to the end of `src/deploy/api_server.py`:

```python
def load_catalog(csv_path: Path) -> list[dict]:
    df = pd.read_csv(csv_path)
    return [
        {"sku": row.product_name, "name": leaf_display_name(row.product_name), "priceUsd": float(row.price_usd)}
        for row in df.itertuples()
    ]


@lru_cache
def get_catalog() -> list[dict]:
    return load_catalog(DEFAULT_CATALOG_PATH)


@app.get("/catalog")
def catalog_endpoint(catalog: list[dict] = Depends(get_catalog)) -> list[dict]:
    return catalog
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/deploy/test_api_server.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/deploy/api_server.py tests/deploy/test_api_server.py
git commit -m "feat: add GET /catalog endpoint to the detection backend"
```

---

### Task 4: Predict endpoint (TDD)

**Files:**
- Modify: `src/deploy/api_server.py`
- Modify: `tests/deploy/test_api_server.py`

**Interfaces:**
- Consumes: `xyxy_to_fractional_xywh` from Task 2, the `app` instance from Task 3 (same file), `YoloDetector` from `src/models/yolo_detector.py` (`YoloDetector(weights_path).detect(frame_rgb) -> list[dict]` with `class_name`/`confidence`/`bbox` keys, `bbox` = pixel xyxy).
- Produces: `get_detector() -> YoloDetector` (FastAPI dependency), `build_detections(detections: list[dict], img_width: int, img_height: int) -> list[dict]`, `POST /predict` registered on `app`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/deploy/test_api_server.py`:

```python
import io
from unittest.mock import MagicMock

from PIL import Image

from src.deploy.api_server import build_detections, get_detector


def test_build_detections_converts_raw_detections_to_response_shape():
    raw = [{"class_name": "Snacks/Chocolate-Bar", "confidence": 0.9, "bbox": [50.0, 100.0, 150.0, 200.0]}]

    result = build_detections(raw, img_width=200, img_height=400)

    assert len(result) == 1
    assert result[0]["label"] == "Snacks/Chocolate-Bar"
    assert result[0]["confidence"] == 0.9
    assert result[0]["bbox"] == pytest.approx([0.25, 0.25, 0.5, 0.25])
    assert isinstance(result[0]["id"], str) and result[0]["id"]


def test_build_detections_returns_empty_list_for_no_detections():
    assert build_detections([], img_width=200, img_height=400) == []


def test_predict_endpoint_returns_converted_detections_from_mocked_detector():
    mock_detector = MagicMock()
    mock_detector.detect.return_value = [
        {"class_name": "Snacks/Chocolate-Bar", "confidence": 0.9, "bbox": [50.0, 100.0, 150.0, 200.0]}
    ]
    app.dependency_overrides[get_detector] = lambda: mock_detector
    client = TestClient(app)

    image_bytes = io.BytesIO()
    Image.new("RGB", (200, 400)).save(image_bytes, format="JPEG")
    image_bytes.seek(0)

    response = client.post("/predict", files={"file": ("test.jpg", image_bytes, "image/jpeg")})

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["label"] == "Snacks/Chocolate-Bar"
    assert body[0]["bbox"] == pytest.approx([0.25, 0.25, 0.5, 0.25])
    app.dependency_overrides.clear()


def test_predict_endpoint_returns_empty_list_when_no_detections():
    mock_detector = MagicMock()
    mock_detector.detect.return_value = []
    app.dependency_overrides[get_detector] = lambda: mock_detector
    client = TestClient(app)

    image_bytes = io.BytesIO()
    Image.new("RGB", (100, 100)).save(image_bytes, format="JPEG")
    image_bytes.seek(0)

    response = client.post("/predict", files={"file": ("test.jpg", image_bytes, "image/jpeg")})

    assert response.status_code == 200
    assert response.json() == []
    app.dependency_overrides.clear()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/deploy/test_api_server.py -v`
Expected: the 4 new tests FAIL with `ImportError: cannot import name 'build_detections'`

- [ ] **Step 3: Implement the predict endpoint**

Add to the imports at the top of `src/deploy/api_server.py`:

```python
import uuid

import numpy as np
from fastapi import UploadFile
from PIL import Image

from src.models.yolo_detector import YoloDetector
```

Append to the end of `src/deploy/api_server.py`:

```python
@lru_cache
def get_detector() -> YoloDetector:
    weights_path = Path(os.environ.get("SMARTCART_WEIGHTS_PATH", str(DEFAULT_WEIGHTS_PATH)))
    return YoloDetector(weights_path)


def build_detections(detections: list[dict], img_width: int, img_height: int) -> list[dict]:
    return [
        {
            "id": str(uuid.uuid4()),
            "label": d["class_name"],
            "confidence": d["confidence"],
            "bbox": list(xyxy_to_fractional_xywh(d["bbox"], img_width, img_height)),
        }
        for d in detections
    ]


@app.post("/predict")
def predict_endpoint(file: UploadFile, detector: YoloDetector = Depends(get_detector)) -> list[dict]:
    image = Image.open(file.file).convert("RGB")
    frame_rgb = np.array(image)
    detections = detector.detect(frame_rgb)
    return build_detections(detections, image.width, image.height)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/deploy/test_api_server.py -v`
Expected: 11 passed

- [ ] **Step 5: Run the full test suite to confirm nothing else broke**

Run: `uv run pytest -v`
Expected: all tests pass (existing `tests/data/`, `tests/models/` suites plus the new `tests/deploy/` suite).

- [ ] **Step 6: Commit**

```bash
git add src/deploy/api_server.py tests/deploy/test_api_server.py
git commit -m "feat: add POST /predict endpoint to the detection backend"
```

---

### Task 5: `main_api_server.py` entrypoint

**Files:**
- Create: `main_api_server.py`

**Interfaces:**
- Consumes: `app` (the `FastAPI` instance) from `src/deploy/api_server.py`, imported by dotted path string for uvicorn's reload support.

- [ ] **Step 1: Write the entrypoint**

Create `main_api_server.py`:

```python
"""Runs the FastAPI detection backend via uvicorn."""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("src.deploy.api_server:app", host="0.0.0.0", port=8000, reload=True)
```

- [ ] **Step 2: Verify it starts and serves real data**

Run: `uv run python main_api_server.py` (leave running)
Expected: log output ending in `Uvicorn running on http://0.0.0.0:8000`

In another terminal, run: `curl http://localhost:8000/catalog`
Expected: a JSON array including `{"sku":"Ready-To-Eat/Instant-Noodles","name":"Instant Noodles","priceUsd":3.0}` and `{"sku":"Snacks/Chocolate-Bar","name":"Chocolate Bar","priceUsd":6.2}` among the entries.

Stop the server with Ctrl-C.

- [ ] **Step 3: Commit**

```bash
git add main_api_server.py
git commit -m "feat: add main_api_server.py entrypoint"
```

---

### Task 6: Rewire the frontend to the real backend

**Files:**
- Modify: `frontend/src/lib/api.ts`

**Interfaces:**
- Consumes: `GET /catalog` and `POST /predict` from Tasks 3-4, matching the `Detection`/`CatalogItem` shapes in `frontend/src/types/index.ts` exactly (`bbox` as fractional `[x, y, w, h]`, `sku` equal to `label`/`model.names` values).
- Produces: `ApiClient` with only `predict`/`getCatalog` (no `addItem`/`removeItem`), consumed by `frontend/src/hooks/useSmartCart.ts` (unchanged — its calls to `api.predict`/`api.getCatalog` already match this shape).

- [ ] **Step 1: Replace the mock implementation**

Replace the full contents of `frontend/src/lib/api.ts` with:

```typescript
import type { CatalogItem, Detection } from '@/types'

export interface ApiClient {
  predict(input: File): Promise<Detection[]>
  getCatalog(): Promise<CatalogItem[]>
}

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? 'http://localhost:8000'

export const api: ApiClient = {
  async predict(input) {
    const formData = new FormData()
    formData.append('file', input)
    const response = await fetch(`${API_BASE_URL}/predict`, { method: 'POST', body: formData })
    if (!response.ok) throw new Error(`predict failed: ${response.status}`)
    return response.json() as Promise<Detection[]>
  },

  async getCatalog() {
    const response = await fetch(`${API_BASE_URL}/catalog`)
    if (!response.ok) throw new Error(`getCatalog failed: ${response.status}`)
    return response.json() as Promise<CatalogItem[]>
  },
}
```

- [ ] **Step 2: Verify the frontend still type-checks and builds**

Run: `cd frontend && npm run build`
Expected: completes with no TypeScript errors (in particular, no "unused import" errors from the removed `CATALOG_SEED`/`matchDroppedFileToSku` imports, and no error from `useSmartCart.ts` or any other caller referencing the now-removed `addItem`/`removeItem` — there are none, confirmed by `grep -rn "api\.\(addItem\|removeItem\)" frontend/src/` returning no matches).

- [ ] **Step 3: Commit**

```bash
cd frontend && git add src/lib/api.ts && cd ..
git commit -m "feat: wire frontend to the real FastAPI detection backend"
```

---

### Task 7: End-to-end verification

**Files:** none (manual verification only).

**Interfaces:** none.

- [ ] **Step 1: Start the backend**

Run: `uv run python main_api_server.py`
Expected: `Uvicorn running on http://0.0.0.0:8000`

- [ ] **Step 2: Start the frontend**

In another terminal: `cd frontend && npm run dev`
Expected: Vite prints a local URL, typically `http://localhost:5173`

- [ ] **Step 3: Verify real catalog data loads**

Open the printed URL in a browser. In the cart sidebar / catalog display, confirm prices match `artifacts/catalog_prices.csv` (e.g. an item under Ready-To-Eat priced at $3.00), not the old mocked `catalogSeed.ts` values.

- [ ] **Step 4: Verify real detection**

Drag `frontend/public/samples/instant-noodles.jpg` onto the camera feed drop zone (or use "Freeze & Detect" with a real product in webcam view). Confirm:
- A bounding box appears at a plausible location on the image (not a random mock box).
- The label matches an actual detected class (e.g. `Ready-To-Eat/Instant-Noodles`).
- The cart sidebar shows the correct name and price for that detection.

- [ ] **Step 5: Verify the empty-detection case doesn't error**

Drag an image with no recognizable product (e.g. a blank or unrelated photo). Confirm the UI shows "no items scanned" / empty state rather than an error toast — this exercises `predict_endpoint`'s empty-list path end-to-end, not just in the mocked unit test.
