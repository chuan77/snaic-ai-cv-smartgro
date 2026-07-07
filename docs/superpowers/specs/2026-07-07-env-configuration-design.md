# .env-based configuration for FastAPI backend + React frontend

## Context

The FastAPI backend (`main_api_server.py`, `src/deploy/api_server.py`) and the Vite/React frontend (`frontend/`) currently hardcode all runtime configuration: backend host/port/reload flag, CORS allowed origin, YOLO weights path, catalog CSV path, and the frontend dev server port. Only one value (`SMARTCART_WEIGHTS_PATH`) reads an environment variable today, and nothing loads a `.env` file — the variable would have to be exported manually in the shell. This makes it awkward to run the backend on a different port, point it at different model weights/catalog data, or run the frontend dev server on a non-default port without editing source.

This spec covers adding `.env`-driven configuration to both halves of the app, scoped to the FastAPI backend and the React frontend only (the Gradio dashboard at `src/deploy/register.py` / `main_day5_deploy.py` is explicitly out of scope — it's a separate, currently-unmaintained demo path).

## Backend: `.env` at repo root, loaded via `python-dotenv`

Add `python-dotenv` (`>=1.0.0`) to `pyproject.toml` dependencies. This is a new dependency, justified because the codebase already has the *pattern* of reading config from `os.environ` (`api_server.py`'s `SMARTCART_WEIGHTS_PATH`) but no mechanism to load a file into the environment — `python-dotenv` is the standard, minimal way to do that without inventing a custom parser.

No settings/config class is introduced. Given there are only six flat, independent values and no existing config-object abstraction anywhere in this codebase, a `pydantic-settings`-style class would be new ceremony for no real benefit — plain `os.environ.get(KEY, default)` calls, consistent with the existing `SMARTCART_WEIGHTS_PATH` line, are used throughout.

`load_dotenv()` is called once, at the top of `main_api_server.py`, before `uvicorn.run(...)` — this is the process entrypoint, so the environment is populated before anything else (including the `src.deploy.api_server` import) reads it.

### Variables

| Variable | Default | Used in | Replaces |
|---|---|---|---|
| `SMARTCART_HOST` | `0.0.0.0` | `main_api_server.py` | hardcoded `host="0.0.0.0"` |
| `SMARTCART_PORT` | `8000` | `main_api_server.py` | hardcoded `port=8000` |
| `SMARTCART_RELOAD` | `true` | `main_api_server.py` | hardcoded `reload=True` |
| `SMARTCART_CORS_ORIGINS` | `http://localhost:5173` | `src/deploy/api_server.py` | hardcoded `allow_origins=["http://localhost:5173"]` |
| `SMARTCART_WEIGHTS_PATH` | `./runs/detect/train/weights/best.pt` | `src/deploy/api_server.py` | already exists as an env var; now loadable from `.env` too |
| `SMARTCART_CATALOG_PATH` | `./artifacts/catalog_prices.csv` | `src/deploy/api_server.py` | hardcoded `DEFAULT_CATALOG_PATH` |

- `SMARTCART_PORT` is parsed with `int(...)`.
- `SMARTCART_RELOAD` is parsed as a boolean via a small helper (`value.lower() in ("1", "true", "yes")`), since env vars are always strings.
- `SMARTCART_CORS_ORIGINS` is comma-separated and split into a list, so multiple origins can be allowed (e.g. `http://localhost:5173,http://127.0.0.1:5173`) without further code changes.

### Files touched
- `main_api_server.py`: add `load_dotenv()` call; read `SMARTCART_HOST`/`SMARTCART_PORT`/`SMARTCART_RELOAD` and pass to `uvicorn.run(...)`.
- `src/deploy/api_server.py`: replace `DEFAULT_CATALOG_PATH` constant and the hardcoded `allow_origins` list with env-driven reads (module-level, same style as the existing `get_detector()` env read).
- `pyproject.toml`: add `python-dotenv` dependency.

## Frontend: `frontend/.env`, Vite's native mechanism (no new dependency)

Vite already loads `.env`/`.env.local` files from the project root and exposes any `VITE_`-prefixed variable via `import.meta.env` to application code — this is why `VITE_API_BASE_URL` already works in `frontend/src/lib/api.ts` today, it just has no example file documenting it.

### Variables

| Variable | Default | Used in |
|---|---|---|
| `VITE_API_BASE_URL` | `http://localhost:8000` | `frontend/src/lib/api.ts` (already implemented) |
| `VITE_DEV_SERVER_PORT` | `5173` | `frontend/vite.config.ts` (new) |

`vite.config.ts` currently doesn't read any env at all. It will use Vite's `loadEnv(mode, process.cwd(), '')` helper (the empty prefix means it loads all vars, not just `VITE_`-prefixed ones, since `vite.config.ts` runs in Node context, not client code) to read `VITE_DEV_SERVER_PORT` and set it as `server.port` in the config, falling back to Vite's own default (5173) if unset.

### Files touched
- `frontend/vite.config.ts`: read `VITE_DEV_SERVER_PORT` via `loadEnv`, set `server.port`.
- No changes needed to `frontend/src/lib/api.ts` — it already reads `VITE_API_BASE_URL` correctly.

## Documentation & git hygiene

- `.env.example` (repo root): lists all six backend variables with their defaults and a one-line comment each.
- `frontend/.env.example`: lists both frontend variables with defaults and comments.
- `.gitignore`: add `.env` and `frontend/.env` (the `.example` files are committed; the real files are not).

## Out of scope

- The Gradio dashboard (`main_day5_deploy.py`, `src/deploy/register.py`) keeps its current hardcoded paths/port — not part of this change.
- No config validation library, no nested/structured config — six backend values and two frontend values don't warrant it.
- No attempt to auto-sync the frontend's dev server port with the backend's CORS origin across the two separate `.env` files — if a user changes `VITE_DEV_SERVER_PORT`, they must also update `SMARTCART_CORS_ORIGINS` to match, documented via comments in both `.env.example` files.

## Verification

1. Backend with no `.env` present: `python main_api_server.py` starts on `0.0.0.0:8000` with reload on, `/catalog` and `/predict` behave as before (defaults match current hardcoded behavior).
2. Backend with a custom `.env` (different port, custom CORS origin, custom catalog path): confirm the server actually binds to the new port and the new catalog path is used by `/catalog`.
3. Frontend with no `.env`: `npm run dev` starts on port 5173 as today.
4. Frontend with `frontend/.env` setting `VITE_DEV_SERVER_PORT=5180` and a custom `VITE_API_BASE_URL`: confirm the dev server binds to 5180 and `api.ts`'s fetch calls target the custom base URL (visible in devtools network tab).
5. `npm run build` and `npm run lint` stay clean.
