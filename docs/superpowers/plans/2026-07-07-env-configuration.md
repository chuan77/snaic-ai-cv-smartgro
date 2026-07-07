# .env-based Configuration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the FastAPI backend's host/port/reload/CORS/paths and the React frontend's dev-server-port/API-base-URL configurable via `.env` files instead of hardcoded values.

**Architecture:** Backend reads plain env vars (loaded from a root `.env` via `python-dotenv`) through small pure functions in `src/deploy/api_server.py`; `main_api_server.py` stays a thin entrypoint that calls one of those functions. Frontend uses Vite's native `.env` support — `loadEnv()` in `vite.config.ts` for the dev server port, `import.meta.env` (already wired) for the API base URL.

**Tech Stack:** Python (FastAPI, `python-dotenv`), TypeScript (Vite).

## Global Constraints

- Backend config format: `.env` only (not YAML), loaded via `python-dotenv`. Source: design spec, "Backend format" decision.
- No config/settings class — plain `os.environ.get(KEY, default)` reads, matching the existing `SMARTCART_WEIGHTS_PATH` pattern in `src/deploy/api_server.py`.
- Gradio dashboard (`main_day5_deploy.py`, `src/deploy/register.py`) is out of scope — do not modify.
- `SMARTCART_CORS_ORIGINS` is comma-separated, split into a list.
- `.env` files are gitignored; `.env.example` files are committed.

---

## File Structure

- **Modify** `pyproject.toml` — add `python-dotenv` dependency.
- **Modify** `src/deploy/api_server.py` — add `load_dotenv()` call, `get_cors_origins()`, `get_catalog_path()`, `parse_bool_env()`, `get_server_config()`; wire `get_cors_origins()`/`get_catalog_path()` into existing `app.add_middleware(...)` and `get_catalog()`.
- **Modify** `main_api_server.py` — use `get_server_config()` instead of hardcoded `host`/`port`/`reload`.
- **Modify** `tests/deploy/test_api_server.py` — add tests for the four new functions.
- **Create** `.env.example` (repo root) — documents all six backend variables.
- **Create** `frontend/.env.example` — documents both frontend variables.
- **Modify** `frontend/vite.config.ts` — read `VITE_DEV_SERVER_PORT` via `loadEnv()`, apply to `server.port`.
- **Modify** `.gitignore` — ignore `.env` and `frontend/.env`.

---

### Task 1: Backend — env-driven CORS origins and catalog path

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/deploy/api_server.py:1-25` (imports, module-level constants, CORS middleware setup), `src/deploy/api_server.py:56-58` (`get_catalog`)
- Test: `tests/deploy/test_api_server.py`
- Create: `.env.example`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `get_cors_origins() -> list[str]`, `get_catalog_path() -> Path` — both in `src/deploy/api_server.py`, used later by Task 2 (no dependency) and already consumed by `get_catalog()` in this task.

- [ ] **Step 1: Add the `python-dotenv` dependency**

Run: `uv add python-dotenv`

Expected: `pyproject.toml`'s `dependencies` list gains a `"python-dotenv>=1.0.0"`-style line, and `uv.lock` is updated. Confirm with:

```bash
grep python-dotenv pyproject.toml
```
Expected output: `    "python-dotenv>=1.0.0",`

- [ ] **Step 2: Write the failing tests for `get_cors_origins` and `get_catalog_path`**

Add to `tests/deploy/test_api_server.py`, after the existing imports (keep the existing `from src.deploy.api_server import (...)` block, just add the two new names to it so the whole import block reads):

```python
from src.deploy.api_server import (
    app,
    build_detections,
    get_catalog,
    get_catalog_path,
    get_cors_origins,
    get_detector,
    leaf_display_name,
    load_catalog,
    xyxy_to_fractional_xywh,
)
```

Then add these tests anywhere after the imports (e.g. right after `test_leaf_display_name_derives_human_readable_label`):

```python
def test_get_cors_origins_defaults_to_localhost_5173(monkeypatch):
    monkeypatch.delenv("SMARTCART_CORS_ORIGINS", raising=False)

    assert get_cors_origins() == ["http://localhost:5173"]


def test_get_cors_origins_splits_comma_separated_values_and_strips_whitespace(monkeypatch):
    monkeypatch.setenv("SMARTCART_CORS_ORIGINS", "http://localhost:5173, http://127.0.0.1:5173 ,http://example.com")

    assert get_cors_origins() == [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://example.com",
    ]


def test_get_catalog_path_defaults_to_artifacts_catalog_prices_csv(monkeypatch):
    monkeypatch.delenv("SMARTCART_CATALOG_PATH", raising=False)

    assert get_catalog_path() == Path("./artifacts/catalog_prices.csv")


def test_get_catalog_path_reads_from_environment(monkeypatch, tmp_path):
    custom_path = tmp_path / "custom_catalog.csv"
    monkeypatch.setenv("SMARTCART_CATALOG_PATH", str(custom_path))

    assert get_catalog_path() == custom_path
```

Add `from pathlib import Path` to the top of `tests/deploy/test_api_server.py` (it isn't imported there yet):

```python
import io
from pathlib import Path
from unittest.mock import MagicMock
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/deploy/test_api_server.py -k "cors_origins or catalog_path" -v`

Expected: `ImportError: cannot import name 'get_cors_origins' from 'src.deploy.api_server'` (or similar collection error) — the functions don't exist yet.

- [ ] **Step 4: Implement `get_cors_origins` and `get_catalog_path`, wire them in**

Replace lines 1-25 of `src/deploy/api_server.py`:

```python
"""FastAPI backend serving real YOLO detections and catalog data to the frontend."""
import os
from functools import lru_cache
from pathlib import Path

import uuid

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from src.models.yolo_detector import YoloDetector

load_dotenv()

DEFAULT_WEIGHTS_PATH = Path("./runs/detect/train/weights/best.pt")
DEFAULT_CATALOG_PATH = Path("./artifacts/catalog_prices.csv")
DEFAULT_CORS_ORIGINS = "http://localhost:5173"


def get_cors_origins() -> list[str]:
    raw = os.environ.get("SMARTCART_CORS_ORIGINS", DEFAULT_CORS_ORIGINS)
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def get_catalog_path() -> Path:
    return Path(os.environ.get("SMARTCART_CATALOG_PATH", str(DEFAULT_CATALOG_PATH)))


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Replace lines 56-58 of `src/deploy/api_server.py` (the `get_catalog` function — find it by its `@lru_cache` decorator directly above `def get_catalog() -> list[dict]:`):

```python
@lru_cache
def get_catalog() -> list[dict]:
    return load_catalog(get_catalog_path())
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/deploy/test_api_server.py -v`

Expected: all tests pass, including the pre-existing ones (`test_catalog_endpoint_returns_the_overridden_catalog` etc. — confirm none regressed).

- [ ] **Step 6: Create `.env.example` at the repo root**

```
# FastAPI backend configuration. Copy this file to `.env` and adjust values as needed.

# Host the uvicorn server binds to.
SMARTCART_HOST=0.0.0.0
# Port the uvicorn server listens on.
SMARTCART_PORT=8000
# Enable uvicorn's auto-reload (dev only; set to false in production).
SMARTCART_RELOAD=true
# Comma-separated list of origins allowed to call this API (CORS). Must match the frontend's dev server URL (see frontend/.env.example's VITE_DEV_SERVER_PORT).
SMARTCART_CORS_ORIGINS=http://localhost:5173
# Path to the trained YOLO weights file used for detection.
SMARTCART_WEIGHTS_PATH=./runs/detect/train/weights/best.pt
# Path to the catalog CSV (class_id,product_name,price_usd) served by GET /catalog.
SMARTCART_CATALOG_PATH=./artifacts/catalog_prices.csv
```

- [ ] **Step 7: Ignore real `.env` files in git**

Add to `.gitignore`, after the `# OS artifacts` section (before `# Frontend (frontend/)`):

```
# Environment configuration (copy .env.example -> .env)
.env
frontend/.env
```

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock src/deploy/api_server.py tests/deploy/test_api_server.py .env.example .gitignore
git commit -m "$(cat <<'EOF'
feat: make backend CORS origins and catalog path configurable via .env

Adds python-dotenv so SMARTCART_CORS_ORIGINS and SMARTCART_CATALOG_PATH
can be set from a .env file instead of being hardcoded.
EOF
)"
```

---

### Task 2: Backend — env-driven host/port/reload

**Files:**
- Modify: `src/deploy/api_server.py` (add `parse_bool_env`, `get_server_config`)
- Modify: `main_api_server.py`
- Test: `tests/deploy/test_api_server.py`
- Modify: `.env.example`

**Interfaces:**
- Consumes: none from Task 1 directly (independent functions in the same file).
- Produces: `parse_bool_env(value: str) -> bool` and `get_server_config() -> dict[str, str | int | bool]` (keys: `"host"`, `"port"`, `"reload"`) in `src/deploy/api_server.py`, consumed by `main_api_server.py`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/deploy/test_api_server.py`'s import block (extend the same `from src.deploy.api_server import (...)` block from Task 1 to also include `get_server_config` and `parse_bool_env`, keeping alphabetical order):

```python
from src.deploy.api_server import (
    app,
    build_detections,
    get_catalog,
    get_catalog_path,
    get_cors_origins,
    get_detector,
    get_server_config,
    leaf_display_name,
    load_catalog,
    parse_bool_env,
    xyxy_to_fractional_xywh,
)
```

Add these tests (e.g. after the `get_catalog_path` tests from Task 1):

```python
@pytest.mark.parametrize("value,expected", [
    ("true", True),
    ("True", True),
    ("1", True),
    ("yes", True),
    ("false", False),
    ("False", False),
    ("0", False),
    ("no", False),
    ("", False),
])
def test_parse_bool_env_recognizes_truthy_and_falsy_strings(value, expected):
    assert parse_bool_env(value) is expected


def test_get_server_config_uses_defaults_when_env_unset(monkeypatch):
    monkeypatch.delenv("SMARTCART_HOST", raising=False)
    monkeypatch.delenv("SMARTCART_PORT", raising=False)
    monkeypatch.delenv("SMARTCART_RELOAD", raising=False)

    assert get_server_config() == {"host": "0.0.0.0", "port": 8000, "reload": True}


def test_get_server_config_reads_from_environment(monkeypatch):
    monkeypatch.setenv("SMARTCART_HOST", "127.0.0.1")
    monkeypatch.setenv("SMARTCART_PORT", "9090")
    monkeypatch.setenv("SMARTCART_RELOAD", "false")

    assert get_server_config() == {"host": "127.0.0.1", "port": 9090, "reload": False}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/deploy/test_api_server.py -k "server_config or parse_bool_env" -v`

Expected: `ImportError: cannot import name 'get_server_config' from 'src.deploy.api_server'`.

- [ ] **Step 3: Implement `parse_bool_env` and `get_server_config`**

Add to `src/deploy/api_server.py`, directly below the `get_catalog_path` function added in Task 1 (i.e. after `def get_catalog_path() -> Path: ...` and before `app = FastAPI()`):

```python
def parse_bool_env(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes")


def get_server_config() -> dict[str, str | int | bool]:
    return {
        "host": os.environ.get("SMARTCART_HOST", "0.0.0.0"),
        "port": int(os.environ.get("SMARTCART_PORT", "8000")),
        "reload": parse_bool_env(os.environ.get("SMARTCART_RELOAD", "true")),
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/deploy/test_api_server.py -v`

Expected: all tests pass (including Task 1's and the pre-existing ones).

- [ ] **Step 5: Wire `main_api_server.py` to use `get_server_config`**

Replace the full contents of `main_api_server.py`:

```python
"""Runs the FastAPI detection backend via uvicorn."""
import uvicorn

from src.deploy.api_server import get_server_config

if __name__ == "__main__":
    config = get_server_config()
    uvicorn.run(
        "src.deploy.api_server:app",
        host=config["host"],
        port=config["port"],
        reload=config["reload"],
    )
```

- [ ] **Step 6: Extend `.env.example` — no change needed**

The three variables (`SMARTCART_HOST`, `SMARTCART_PORT`, `SMARTCART_RELOAD`) were already added to `.env.example` in Task 1 Step 6. Confirm they're present:

```bash
grep -E "SMARTCART_(HOST|PORT|RELOAD)=" .env.example
```

Expected: all three lines present.

- [ ] **Step 7: Manually verify the server actually honors these variables**

```bash
SMARTCART_PORT=9191 SMARTCART_RELOAD=false uv run python main_api_server.py &
sleep 2
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:9191/catalog
kill %1
```

Expected: `200` printed (server bound to the custom port 9191, confirming the env var took effect), then the background server is killed.

- [ ] **Step 8: Commit**

```bash
git add src/deploy/api_server.py main_api_server.py tests/deploy/test_api_server.py
git commit -m "$(cat <<'EOF'
feat: make backend host/port/reload configurable via .env

main_api_server.py now reads SMARTCART_HOST/SMARTCART_PORT/SMARTCART_RELOAD
through get_server_config() instead of hardcoding uvicorn.run's arguments.
EOF
)"
```

---

### Task 3: Frontend — configurable dev server port + `.env.example`

**Files:**
- Modify: `frontend/vite.config.ts`
- Create: `frontend/.env.example`

**Interfaces:**
- Consumes: none (independent of backend tasks).
- Produces: nothing consumed by later tasks — this is the last file-change task.

- [ ] **Step 1: Update `vite.config.ts` to read `VITE_DEV_SERVER_PORT`**

Replace the full contents of `frontend/vite.config.ts`:

```ts
import path from 'node:path'
import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const port = Number(env.VITE_DEV_SERVER_PORT) || 5173

  return {
    plugins: [react()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
      },
    },
    server: {
      port,
    },
  }
})
```

- [ ] **Step 2: Create `frontend/.env.example`**

```
# Frontend configuration. Copy this file to `.env` and adjust values as needed.

# Base URL of the FastAPI backend this app talks to.
VITE_API_BASE_URL=http://localhost:8000
# Port the Vite dev server listens on. Must match SMARTCART_CORS_ORIGINS on the backend (see ../.env.example).
VITE_DEV_SERVER_PORT=5173
```

- [ ] **Step 3: Verify the build still succeeds**

Run: `cd frontend && npm run build`

Expected: `tsc -b && vite build` completes with `✓ built in ...ms` and no errors, same as before this change (confirms `vite.config.ts`'s new `defineConfig(({ mode }) => ...)` callback form is valid).

- [ ] **Step 4: Manually verify the dev server honors the custom port**

```bash
cd frontend
VITE_DEV_SERVER_PORT=5199 npm run dev &
sleep 2
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:5199
kill %1
```

Expected: `200` printed (dev server actually bound to port 5199), then the background process is killed.

- [ ] **Step 5: Commit**

```bash
git add frontend/vite.config.ts frontend/.env.example
git commit -m "$(cat <<'EOF'
feat: make frontend dev server port configurable via .env

vite.config.ts now reads VITE_DEV_SERVER_PORT via Vite's loadEnv, and
frontend/.env.example documents both frontend env vars.
EOF
)"
```

---

### Task 4: End-to-end verification with real `.env` files

**Files:** none (verification only — no new files).

**Interfaces:** none.

- [ ] **Step 1: Create real `.env` files from the examples with non-default values**

```bash
cp .env.example .env
sed -i '' 's/SMARTCART_PORT=8000/SMARTCART_PORT=8500/' .env
sed -i '' 's#SMARTCART_CORS_ORIGINS=http://localhost:5173#SMARTCART_CORS_ORIGINS=http://localhost:5280#' .env

cp frontend/.env.example frontend/.env
sed -i '' 's/VITE_DEV_SERVER_PORT=5173/VITE_DEV_SERVER_PORT=5280/' frontend/.env
sed -i '' 's#VITE_API_BASE_URL=http://localhost:8000#VITE_API_BASE_URL=http://localhost:8500#' frontend/.env
```

- [ ] **Step 2: Start both processes with the custom `.env` files**

```bash
uv run python main_api_server.py &
BACKEND_PID=$!
sleep 2
cd frontend && npm run dev &
FRONTEND_PID=$!
cd ..
sleep 2
```

- [ ] **Step 3: Confirm the backend bound to the custom port and returns the catalog**

```bash
curl -s http://localhost:8500/catalog | head -c 200
```

Expected: a JSON array (e.g. starting with `[{"sku":...`), not a connection error.

- [ ] **Step 4: Confirm the frontend bound to the custom port**

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:5280
```

Expected: `200`.

- [ ] **Step 5: Confirm the frontend's requests to the backend succeed cross-origin (CORS)**

Open `http://localhost:5280` in a browser (or, headlessly, use the Playwright pattern from earlier in this session — launch Chromium, `page.goto('http://localhost:5280')`, wait, then check `page.on('console')` for CORS errors). Confirm the catalog loads (cart sidebar shows no fetch error) and there are no `CORS policy` errors in the console — this proves `VITE_API_BASE_URL=http://localhost:8500` (frontend) and `SMARTCART_CORS_ORIGINS=http://localhost:5280` (backend) are correctly cross-referencing each other.

- [ ] **Step 6: Clean up**

```bash
kill $BACKEND_PID $FRONTEND_PID
rm .env frontend/.env
```

Expected: both background processes stopped; the real `.env` files removed (they were only created for this manual verification — `.env.example` files remain, committed).

- [ ] **Step 7: Confirm defaults still work with no `.env` present**

```bash
uv run pytest tests/deploy/test_api_server.py -v
cd frontend && npm run build && npm run lint
```

Expected: all backend tests pass, frontend build and lint stay clean — confirming the feature is additive and nothing breaks when no `.env` file exists (matching current/default behavior).
