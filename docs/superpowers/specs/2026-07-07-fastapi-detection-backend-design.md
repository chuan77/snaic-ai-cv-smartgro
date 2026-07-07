# FastAPI Detection Backend — Design

## Context

The SmartCart pipeline (Days 1-5) trains a YOLO11 detector and serves it through a Gradio dashboard (`main_day5_deploy.py`). Separately, `frontend/` holds a fully-built React/TypeScript/Vite UI (camera capture, drag-drop, cart sidebar, debug panel) that was designed against a `ApiClient` interface (`frontend/src/lib/api.ts`) but never wired to a real backend — `predict()` is currently a mock that fuzzy-matches the dropped file's *name* against a hardcoded, stale catalog snapshot (`catalogSeed.ts`, predating today's Instant-Noodles/Chocolate-Bar work) instead of running real inference.

This spec covers building the real backend: a FastAPI service that loads the trained weights (`runs/detect/train/weights/best.pt`, currently a 5-class detector) and serves real detections and real catalog data, then rewiring `api.ts` to call it instead of mocking. The goal is that dragging in or capturing a real product photo in the existing frontend produces a real bounding box with the correct name and price.

## Architecture

A new `src/deploy/api_server.py` module — alongside the existing `register.py`, since both are "serve the trained model" deployment code — hosts a FastAPI app with two endpoints. It reuses the already-implemented `YoloDetector` (`src/models/yolo_detector.py:11-29`), which wraps model loading + device selection + predict + box/class/confidence parsing, and is already unit-tested (`tests/models/test_yolo_detector.py`). A thin `main_api_server.py` root entrypoint runs the app via `uvicorn`, matching the existing `main_dayN_*.py` convention of "thin script imports from `src/`".

CORS is enabled (via `CORSMiddleware`) for the Vite dev origin (`http://localhost:5173`), so the frontend calls the backend directly over HTTP with no dev-proxy layer — two independent processes, simplest to reason about, easiest to later deploy separately.

## Backend endpoints

### `GET /catalog`

Reads `artifacts/catalog_prices.csv` (`class_id, product_name, price_usd`) once at process startup into an in-memory list, and returns `[{sku, name, priceUsd}]` for every row.

- `sku` = `product_name` verbatim (e.g. `"Ready-To-Eat/Instant-Noodles"`). This **must** match the string values in the trained model's `model.names` dict exactly, since the frontend joins a detection to a catalog row by exact string equality (`frontend/src/hooks/useSmartCart.ts:64`, `catalog.find((item) => item.sku === detection.label)`).
- `name` = a derived human-readable label: take the last `/`-separated path segment, replace `-` with spaces, title-case it (e.g. `"Ready-To-Eat/Instant-Noodles"` → `"Instant Noodles"`, `"Fruit/Apple/Royal-Gala"` → `"Royal Gala"`). This is a small pure function (`leaf_display_name(product_name: str) -> str`), independently unit-tested.
- `priceUsd` = `price_usd` as a float.

### `POST /predict`

Accepts one multipart file upload (`UploadFile`, an image). Loads it into an RGB array, runs it through the process-wide `YoloDetector` singleton's `.detect()`, and for each returned detection converts its pixel `xyxy` box (plus the image's width/height) into the fractional top-left-origin `[x, y, w, h]` the frontend's `DetectionOverlay` renders (`frontend/src/components/DetectionOverlay.tsx:16,26-31`). This conversion is a second small pure function (`xyxy_to_fractional_xywh(box, img_width, img_height) -> tuple[float, float, float, float]`), independently unit-tested.

Returns `[{id, label, confidence, bbox}]` — `id` generated server-side (`uuid4`), `label` = the detector's class name string (already equal to a catalog `sku`), `confidence` = the box's confidence score. Returns `[]` (not an error) when nothing is detected — this is the expected, common case, not a failure.

### Dropped from scope: `addItem` / `removeItem`

The current `ApiClient` interface declares `addItem(sku)`/`removeItem(sku)`, but nothing in the frontend calls them — cart mutation is pure client-side zustand state (`toggleLine`/`clearCart` in `useSmartCart.ts`). There is no server-side cart concept to build. Both methods are removed from `ApiClient` and `api.ts` entirely, rather than implemented as no-op stubs.

## Model & catalog loading

- Weights path: defaults to `./runs/detect/train/weights/best.pt`, overridable via a `SMARTCART_WEIGHTS_PATH` environment variable. Loaded once, at process startup (module-level singleton), not per-request — mirrors how `AutonomousPOSRegister.__init__` already loads the model once (`src/deploy/register.py:17`).
- If the weights file is missing at startup, the process should fail loudly (raise), not silently no-op — consistent with "fail fast" being preferable to serving a broken endpoint.
- Catalog CSV path: `./artifacts/catalog_prices.csv`, read once at startup the same way.

## New Python dependencies

`fastapi`, `uvicorn`, `python-multipart` (required by FastAPI/Starlette to parse `multipart/form-data` uploads) — added to `pyproject.toml`.

## Frontend changes

`frontend/src/lib/api.ts` loses its entire mock body — the `delay`/`randomFractionalBbox`/`NETWORK_DELAY_MS` helpers and the `matchDroppedFileToSku`/`CATALOG_SEED` import — replaced with real `fetch` calls:

- `predict(file)`: `POST {API_BASE_URL}/predict` with `FormData` containing the file; parse the JSON response as `Detection[]`.
- `getCatalog()`: `GET {API_BASE_URL}/catalog`; parse the JSON response as `CatalogItem[]`.

`API_BASE_URL` comes from a Vite env var `VITE_API_BASE_URL` (via `import.meta.env.VITE_API_BASE_URL`), defaulting to `http://localhost:8000` if unset — not hardcoded into application code, using Vite's native `.env` support.

`ApiClient`'s `addItem`/`removeItem` methods are removed from the interface (`frontend/src/lib/api.ts:5-10`) as covered above.

`frontend/src/lib/matchCatalog.ts` is **not** touched — `explainMatch` is still used independently by the debug panel (`useSmartCart.ts`'s `debugLog`) to show *why* a dropped filename did or didn't match, which is unrelated to how real detections are matched (by the backend-returned `label`, not by filename).

`frontend/src/data/catalogSeed.ts` becomes dead code once `getCatalog()` calls the real endpoint — it is left in place but unused (not deleted in this pass; deleting it is a candidate for a later cleanup, out of scope here since nothing else in this spec depends on it being gone).

## Testing

Matching this repo's existing pytest conventions (`tests/models/test_yolo_detector.py`'s `unittest.mock.patch` style):

- `tests/deploy/test_api_server.py`:
  - Unit tests for `leaf_display_name` (pure function, table of `product_name` → expected display name).
  - Unit tests for `xyxy_to_fractional_xywh` (pure function, table of pixel box + image dims → expected fractional box).
  - `TestClient`-based tests for `GET /catalog` against a fixture CSV (via `tmp_path`), asserting the returned JSON shape and values.
  - `TestClient`-based tests for `POST /predict` with the module's `YoloDetector` instance mocked (so no real weights are loaded in tests), asserting: a mocked multi-box result converts correctly to the fractional-bbox JSON shape; a mocked zero-box result returns `[]`.

No new frontend test framework is introduced — none exists in `frontend/` today (no vitest/jest in `package.json`), and adding one is out of scope for this change.

## Manual end-to-end verification

1. Start the backend: `uv run python main_api_server.py` (or `uv run uvicorn src.deploy.api_server:app --reload --port 8000`).
2. Start the frontend: `npm run dev` from `frontend/`.
3. In the browser, drag in a real sample photo (e.g. `frontend/public/samples/instant-noodles.jpg`) or use the webcam "Freeze & Detect" button with a real product in frame.
4. Confirm: a real bounding box appears (not a random mock box), the label matches an actual detected class, and the cart sidebar shows the correct catalog price — not the mock's fuzzy filename-based match.
5. Confirm the catalog panel reflects real prices from `artifacts/catalog_prices.csv` (including `Ready-To-Eat/Instant-Noodles` at $3.00 and `Snacks/Chocolate-Bar` at $6.20), not the stale hardcoded `catalogSeed.ts` values.
