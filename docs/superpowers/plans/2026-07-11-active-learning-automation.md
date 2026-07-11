# Autonomous Active-Learning Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automate the manual capture→push→review→retrain→promote active-learning loop end to end, using a local Qwen2.5-VL-3B (via LM Studio) as an independent second opinion in place of human Label Studio review, a new DINOv2 held-out variant-accuracy gate alongside the existing YOLO mAP50 gate, and mechanical auto-promotion with launchd-managed process restart.

**Architecture:** A new `src/models/vlm_verifier.py` talks to a local LM Studio server over its confirmed `/api/v1/chat` schema. A new `src/data/auto_labeler.py` consumes staged captures from the existing `active_learning_capture.py` staging directory, cross-checking mid-confidence YOLO detections against the VLM and asking the VLM directly for zero/low-confidence frames, importing agreed labels straight into the dataset tree. A new `src/models/variant_auditor.py` measures DINOv2 identification accuracy the same way `CheckoutModelAuditor` measures YOLO's. A new `src/pipeline/al_scheduler.py` ties it together: reuses the existing, unmodified `ModelTrainingPipeline` against an isolated temp candidate directory (fixing a live-artifact-overwrite bug in the process), runs both gates, and auto-promotes via a small `src/pipeline/candidate_promotion.py` when the candidate doesn't regress either metric. Two launchd services (one for the API server, one for the periodic scheduler) replace manual invocation.

**Tech Stack:** Python 3.11+, existing `httpx`/`pandas`/`numpy`/`Pillow`/`ultralytics` deps already in `pyproject.toml` (no new dependencies — the VLM is called over plain HTTP, not loaded in-process), `pytest` with `monkeypatch`/`unittest.mock`, macOS `launchd`.

## Global Constraints

- Every new `SMARTCART_*` env var follows the existing convention: a `get_*()` accessor function reading `os.environ` with a sensible default, documented in `.env.example` in the same commit that introduces it.
- Every new failure mode (VLM unreachable, disagreement, unparseable box, failed gate) must degrade to "skip/discard and continue" — never raise out of the scheduler tick, never guess.
- No new Python dependencies. The VLM integration is a plain `httpx` HTTP client against an already-running local server, matching the existing Label Studio integration's pattern exactly (including the `httpx`-module-identity testing gotcha already documented in `CLAUDE.md`: `monkeypatch.setattr("module.httpx.post", ...)` per call site, not a single global patch).
- Root-level scripts stay thin (a handful of lines, following `retrain_from_label_studio.py`'s existing pattern) — all real logic lives in `src/` and is unit-tested there. Root scripts themselves are not unit-tested, matching this repo's existing convention (no test file targets `retrain_from_label_studio.py` or `push_captures_to_label_studio.py` directly).
- Existing Label Studio integration code (`label_studio_push.py`, `label_studio_backend.py`, `label_studio_export_pull.py`, `retrain_from_label_studio.py`) is not modified — it remains available for manual override / cold-start onboarding, just bypassed by the new default automated path.
- `ModelTrainingPipeline` (`src/pipeline/training_pipeline.py`) is not modified — candidate isolation is achieved by pointing its existing `artifacts_dir` constructor argument at a temp directory and renaming it after the run, not by changing the class.
- Run `uv run pytest -v` after every task; it must stay green (161+ passing, 1 deselected `@pytest.mark.slow`) before moving to the next task.

---

## File Structure

New files this plan creates:

| File | Responsibility |
|---|---|
| `src/models/vlm_verifier.py` | LM Studio HTTP client + response parsing (category match, grounding box) |
| `src/models/variant_auditor.py` | DINOv2 held-out variant-accuracy audit (mirrors `CheckoutModelAuditor` for the gallery side) |
| `src/data/auto_labeler.py` | Consumes staged captures, VLM-cross-checks or VLM-labels them, imports into the dataset tree |
| `src/pipeline/candidate_promotion.py` | Candidate report + pipeline state persistence, promotion decision, promotion file operations |
| `src/pipeline/al_scheduler.py` | Orchestrates one scheduler tick: auto-import → retrain trigger → candidate cycle → gate → promote |
| `al_scheduler_check.py` | Thin root entrypoint calling `src.pipeline.al_scheduler.run_scheduler_tick()`, matching `retrain_from_label_studio.py`'s existing pattern |
| `launchd/com.smartcart.api.plist` | Runs the API server as a persistent, crash-recovering service |
| `launchd/com.smartcart.al-scheduler.plist` | Runs `al_scheduler_check.py` on an interval |

Modified files:

| File | Change |
|---|---|
| `.env.example` | New `SMARTCART_VLM_*`, `SMARTCART_AUTOLABEL_MIN_CONF`, `SMARTCART_AL_RETRAIN_TRIGGER_COUNT`, `SMARTCART_RETRAIN_MIN_VARIANT_ACC` vars |

No existing `src/` module's code changes — every new capability is additive.

---

### Task 1: VLM HTTP client

**Files:**
- Create: `src/models/vlm_verifier.py`
- Test: `tests/models/test_vlm_verifier.py`

**Interfaces:**
- Produces: `get_vlm_base_url() -> str`, `get_vlm_model() -> str`, `image_to_data_url(image: PIL.Image.Image) -> str`, `build_chat_payload(model: str, prompt: str, data_url: str) -> dict`, `ask_vlm(image: PIL.Image.Image, prompt: str, timeout: float = 30.0) -> str | None`

- [ ] **Step 1: Write the failing tests**

```python
# tests/models/test_vlm_verifier.py
import base64

from PIL import Image

from src.models.vlm_verifier import (
    ask_vlm,
    build_chat_payload,
    get_vlm_base_url,
    get_vlm_model,
    image_to_data_url,
)


def test_get_vlm_base_url_default_and_override(monkeypatch):
    monkeypatch.delenv("SMARTCART_VLM_BASE_URL", raising=False)
    assert get_vlm_base_url() == "http://localhost:1234"

    monkeypatch.setenv("SMARTCART_VLM_BASE_URL", "http://example.com:9999/")
    assert get_vlm_base_url() == "http://example.com:9999"


def test_get_vlm_model_default_and_override(monkeypatch):
    monkeypatch.delenv("SMARTCART_VLM_MODEL", raising=False)
    assert get_vlm_model() == "qwen2.5-vl-3b-instruct"

    monkeypatch.setenv("SMARTCART_VLM_MODEL", "other-model")
    assert get_vlm_model() == "other-model"


def test_image_to_data_url_round_trips_as_jpeg():
    image = Image.new("RGB", (4, 4), color=(255, 0, 0))

    data_url = image_to_data_url(image)

    assert data_url.startswith("data:image/jpeg;base64,")
    decoded = base64.b64decode(data_url.split(",", 1)[1])
    assert decoded[:2] == b"\xff\xd8"  # JPEG magic bytes


def test_build_chat_payload_matches_lm_studio_schema():
    payload = build_chat_payload("qwen2.5-vl-3b-instruct", "What is this?", "data:image/jpeg;base64,ABC")

    assert payload == {
        "model": "qwen2.5-vl-3b-instruct",
        "input": [
            {"type": "text", "content": "What is this?"},
            {"type": "image", "data_url": "data:image/jpeg;base64,ABC"},
        ],
    }


def test_ask_vlm_posts_to_chat_endpoint_and_returns_content(monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        response = type("R", (), {})()
        response.raise_for_status = lambda: None
        response.json = lambda: {"output": [{"type": "message", "content": "Fruit/Apple"}]}
        return response

    monkeypatch.setenv("SMARTCART_VLM_BASE_URL", "http://localhost:1234")
    monkeypatch.setattr("src.models.vlm_verifier.httpx.post", fake_post)

    answer = ask_vlm(Image.new("RGB", (4, 4)), "Which category?")

    assert answer == "Fruit/Apple"
    assert captured["url"] == "http://localhost:1234/api/v1/chat"
    assert captured["json"]["input"][0] == {"type": "text", "content": "Which category?"}


def test_ask_vlm_returns_none_on_any_error(monkeypatch):
    def fake_post(url, json=None, timeout=None):
        raise ConnectionError("LM Studio not running")

    monkeypatch.setattr("src.models.vlm_verifier.httpx.post", fake_post)

    assert ask_vlm(Image.new("RGB", (4, 4)), "Which category?") is None


def test_ask_vlm_returns_none_on_malformed_response(monkeypatch):
    def fake_post(url, json=None, timeout=None):
        response = type("R", (), {})()
        response.raise_for_status = lambda: None
        response.json = lambda: {"unexpected": "shape"}
        return response

    monkeypatch.setattr("src.models.vlm_verifier.httpx.post", fake_post)

    assert ask_vlm(Image.new("RGB", (4, 4)), "Which category?") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/models/test_vlm_verifier.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models.vlm_verifier'`

- [ ] **Step 3: Write the implementation**

```python
# src/models/vlm_verifier.py
"""HTTP client for a local LM Studio server hosting Qwen2.5-VL, used as an
independent second opinion for auto-labeling active-learning captures. Every
failure mode (unreachable server, timeout, malformed response) returns None —
callers must treat None exactly like an unconfident/disagreeing answer, never
block or raise on it."""
import base64
import io
import os

import httpx
from PIL import Image


def get_vlm_base_url() -> str:
    return os.environ.get("SMARTCART_VLM_BASE_URL", "http://localhost:1234").rstrip("/")


def get_vlm_model() -> str:
    return os.environ.get("SMARTCART_VLM_MODEL", "qwen2.5-vl-3b-instruct")


def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def build_chat_payload(model: str, prompt: str, data_url: str) -> dict:
    return {
        "model": model,
        "input": [
            {"type": "text", "content": prompt},
            {"type": "image", "data_url": data_url},
        ],
    }


def ask_vlm(image: Image.Image, prompt: str, timeout: float = 30.0) -> str | None:
    payload = build_chat_payload(get_vlm_model(), prompt, image_to_data_url(image))
    try:
        response = httpx.post(f"{get_vlm_base_url()}/api/v1/chat", json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()["output"][0]["content"]
    except Exception:
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/models/test_vlm_verifier.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Add config to `.env.example` and commit**

Add these lines to `.env.example`, after the existing `SMARTCART_RETRAIN_MIN_MAP50` line:

```
# --- Autonomous active-learning pipeline ---
# Base URL of the local LM Studio server hosting the VLM verifier.
SMARTCART_VLM_BASE_URL=http://localhost:1234
# Model identifier as loaded in LM Studio.
SMARTCART_VLM_MODEL=qwen2.5-vl-3b-instruct
```

```bash
git add src/models/vlm_verifier.py tests/models/test_vlm_verifier.py .env.example
git commit -m "feat: add LM Studio VLM HTTP client for active-learning auto-labeling"
```

---

### Task 2: VLM response parsing

**Files:**
- Modify: `src/models/vlm_verifier.py`
- Test: `tests/models/test_vlm_verifier.py`

**Interfaces:**
- Consumes: nothing new from Task 1 beyond the same module
- Produces: `match_category(answer: str, candidate_categories: list[str]) -> str | None`, `is_sane_box(box: tuple[int, int, int, int], img_width: int, img_height: int) -> bool`, `parse_grounding_box(answer: str, img_width: int, img_height: int) -> tuple[int, int, int, int] | None`

- [ ] **Step 1: Write the failing tests**

Append to `tests/models/test_vlm_verifier.py`:

```python
from src.models.vlm_verifier import is_sane_box, match_category, parse_grounding_box


def test_match_category_finds_substring_match_case_insensitive():
    categories = ["Fruit/Apple/Royal-Gala", "Snacks/Chocolate-Bar/Cadbury-RoastAlmond"]

    assert match_category("i see a royal-gala apple", categories) == "Fruit/Apple/Royal-Gala"


def test_match_category_prefers_longest_match():
    categories = [
        "Ready-To-Eat/Instant-Noodles/Myojo/Chicken",
        "Ready-To-Eat/Instant-Noodles/Myojo/ChickenAbalone",
    ]

    answer = "This is Ready-To-Eat/Instant-Noodles/Myojo/ChickenAbalone flavor"

    assert match_category(answer, categories) == "Ready-To-Eat/Instant-Noodles/Myojo/ChickenAbalone"


def test_match_category_returns_none_when_no_match():
    assert match_category("I have no idea what this is", ["Fruit/Apple/Royal-Gala"]) is None


def test_is_sane_box_rejects_out_of_bounds():
    assert is_sane_box((10, 10, 200, 50), img_width=100, img_height=100) is False


def test_is_sane_box_rejects_too_small_or_too_large():
    assert is_sane_box((0, 0, 1, 1), img_width=100, img_height=100) is False  # ~0.01% of frame
    assert is_sane_box((0, 0, 100, 100), img_width=100, img_height=100) is False  # 100% of frame


def test_is_sane_box_accepts_plausible_box():
    assert is_sane_box((10, 10, 60, 60), img_width=100, img_height=100) is True


def test_parse_grounding_box_converts_normalized_coordinates():
    answer = "The item is located at (100,100),(500,500)."

    box = parse_grounding_box(answer, img_width=1000, img_height=1000)

    assert box == (100, 100, 500, 500)


def test_parse_grounding_box_returns_none_when_missing():
    assert parse_grounding_box("no box here", img_width=1000, img_height=1000) is None


def test_parse_grounding_box_returns_none_when_box_fails_sanity_check():
    # (0,0),(1000,1000) on a 0-1000 scale covers the entire frame -- not plausible
    answer = "(0,0),(1000,1000)"

    assert parse_grounding_box(answer, img_width=1000, img_height=1000) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/models/test_vlm_verifier.py -v -k "match_category or sane_box or grounding_box"`
Expected: FAIL with `ImportError: cannot import name 'match_category'`

- [ ] **Step 3: Write the implementation**

Append to `src/models/vlm_verifier.py`:

```python
import re

_GROUNDING_BOX_PATTERN = re.compile(r"\((\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+)\)")


def match_category(answer: str, candidate_categories: list[str]) -> str | None:
    """Finds which candidate category the VLM's free-text answer references, via
    case-insensitive substring match, preferring the longest match when several
    candidates appear (mirrors annotation_import.classify_category's approach)."""
    answer_lower = answer.lower()
    matches = [c for c in candidate_categories if c.lower() in answer_lower]
    if not matches:
        return None
    return max(matches, key=len)


def is_sane_box(box: tuple[int, int, int, int], img_width: int, img_height: int) -> bool:
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= img_width and 0 <= y1 < y2 <= img_height):
        return False
    area_ratio = ((x2 - x1) * (y2 - y1)) / (img_width * img_height)
    return 0.01 <= area_ratio <= 0.9


def parse_grounding_box(answer: str, img_width: int, img_height: int) -> tuple[int, int, int, int] | None:
    """Parses a Qwen-style '(x1,y1),(x2,y2)' box on a 0-1000 normalized scale out
    of free text, scales it to pixel xyxy, and returns None if missing or if it
    fails a basic plausibility check (in-bounds, 1-90% of the frame area)."""
    match = _GROUNDING_BOX_PATTERN.search(answer)
    if match is None:
        return None
    x1, y1, x2, y2 = (int(v) for v in match.groups())
    box = (
        int(x1 / 1000 * img_width),
        int(y1 / 1000 * img_height),
        int(x2 / 1000 * img_width),
        int(y2 / 1000 * img_height),
    )
    return box if is_sane_box(box, img_width, img_height) else None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/models/test_vlm_verifier.py -v`
Expected: PASS (16 tests total)

- [ ] **Step 5: Commit**

```bash
git add src/models/vlm_verifier.py tests/models/test_vlm_verifier.py
git commit -m "feat: add VLM category matching and grounding-box parsing"
```

---

### Task 3: DINOv2 held-out variant split

**Files:**
- Create: `src/models/variant_auditor.py`
- Test: `tests/models/test_variant_auditor.py`

**Interfaces:**
- Produces: `split_holdout(meta: pandas.DataFrame, holdout_ratio: float = 0.2) -> tuple[list[int], list[int]]` — returns `(reference_row_indices, holdout_row_indices)` as positional indices into `meta`

- [ ] **Step 1: Write the failing tests**

```python
# tests/models/test_variant_auditor.py
import pandas as pd

from src.models.variant_auditor import split_holdout


def test_split_holdout_excludes_variants_with_fewer_than_two_photos():
    meta = pd.DataFrame([
        {"product_name": "Fruit/Apple/Royal-Gala", "file_name": "a.jpg"},
        {"product_name": "Fruit/Apple/Royal-Gala", "file_name": "b.jpg"},
        {"product_name": "Fruit/Apple/Royal-Gala", "file_name": "c.jpg"},
        {"product_name": "Snacks/Chocolate-Bar/Lindt", "file_name": "d.jpg"},  # only 1 photo
    ])

    reference_idx, holdout_idx = split_holdout(meta, holdout_ratio=0.2)

    # the single-photo variant contributes no holdout rows
    holdout_products = set(meta.loc[holdout_idx, "product_name"])
    assert "Snacks/Chocolate-Bar/Lindt" not in holdout_products
    # but it stays available as a reference row
    assert 3 in reference_idx


def test_split_holdout_holds_out_at_least_one_photo_per_eligible_variant():
    meta = pd.DataFrame([
        {"product_name": "Fruit/Apple/Royal-Gala", "file_name": f"{i}.jpg"} for i in range(5)
    ])

    reference_idx, holdout_idx = split_holdout(meta, holdout_ratio=0.2)

    assert len(holdout_idx) == 1
    assert len(reference_idx) == 4


def test_split_holdout_is_deterministic():
    meta = pd.DataFrame([
        {"product_name": "Fruit/Apple/Royal-Gala", "file_name": f"{i}.jpg"} for i in range(10)
    ])

    first = split_holdout(meta)
    second = split_holdout(meta)

    assert first == second


def test_split_holdout_reference_and_holdout_partition_every_eligible_row():
    meta = pd.DataFrame([
        {"product_name": "Fruit/Apple/Royal-Gala", "file_name": f"{i}.jpg"} for i in range(5)
    ] + [
        {"product_name": "Snacks/Chocolate-Bar/Lindt", "file_name": f"c{i}.jpg"} for i in range(3)
    ])

    reference_idx, holdout_idx = split_holdout(meta)

    assert set(reference_idx) | set(holdout_idx) == set(range(len(meta)))
    assert set(reference_idx) & set(holdout_idx) == set()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/models/test_variant_auditor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models.variant_auditor'`

- [ ] **Step 3: Write the implementation**

```python
# src/models/variant_auditor.py
"""Held-out variant-identification accuracy audit for the DINOv2 gallery -- mirrors
CheckoutModelAuditor's mAP50 gate, but for whether variant resolution actually
improved. Used by the autonomous retrain pipeline's promotion gate."""
import math

import numpy as np
import pandas as pd


def split_holdout(meta: pd.DataFrame, holdout_ratio: float = 0.2) -> tuple[list[int], list[int]]:
    """Splits gallery rows into (reference, holdout) index lists. Only variants
    (product_name groups) with >=2 photos contribute holdout rows -- a variant
    with a single reference photo can't be meaningfully held out against itself,
    so all of its rows stay in the reference pool untouched."""
    reference_idx: list[int] = []
    holdout_idx: list[int] = []

    for _, group in meta.groupby("product_name", sort=True):
        indices = sorted(group.index.tolist())
        if len(indices) < 2:
            reference_idx.extend(indices)
            continue
        num_holdout = max(1, math.floor(len(indices) * holdout_ratio))
        holdout_idx.extend(indices[-num_holdout:])
        reference_idx.extend(indices[:-num_holdout])

    return reference_idx, holdout_idx
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/models/test_variant_auditor.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/models/variant_auditor.py tests/models/test_variant_auditor.py
git commit -m "feat: add deterministic held-out split for DINOv2 variant accuracy audit"
```

---

### Task 4: DINOv2 held-out accuracy computation

**Files:**
- Modify: `src/models/variant_auditor.py`
- Test: `tests/models/test_variant_auditor.py`

**Interfaces:**
- Consumes: `split_holdout` from Task 3
- Produces: `compute_variant_accuracy(embeddings: numpy.ndarray, meta: pandas.DataFrame, holdout_ratio: float = 0.2) -> tuple[float, int]` — returns `(top1_accuracy, excluded_variant_count)`

- [ ] **Step 1: Write the failing tests**

Append to `tests/models/test_variant_auditor.py`:

```python
import numpy as np

from src.models.variant_auditor import compute_variant_accuracy


def test_compute_variant_accuracy_perfect_when_embeddings_are_identical_within_variant():
    meta = pd.DataFrame([
        {"product_name": "Fruit/Apple/Royal-Gala", "file_name": f"{i}.jpg"} for i in range(5)
    ] + [
        {"product_name": "Snacks/Chocolate-Bar/Lindt", "file_name": f"c{i}.jpg"} for i in range(5)
    ])
    embeddings = np.array(
        [[1.0, 0.0]] * 5 + [[0.0, 1.0]] * 5, dtype=np.float32
    )

    accuracy, excluded = compute_variant_accuracy(embeddings, meta)

    assert accuracy == 1.0
    assert excluded == 0


def test_compute_variant_accuracy_counts_excluded_single_photo_variants():
    meta = pd.DataFrame([
        {"product_name": "Fruit/Apple/Royal-Gala", "file_name": f"{i}.jpg"} for i in range(5)
    ] + [
        {"product_name": "Snacks/Chocolate-Bar/Lindt", "file_name": "only_one.jpg"},
    ])
    embeddings = np.array([[1.0, 0.0]] * 5 + [[0.0, 1.0]], dtype=np.float32)

    accuracy, excluded = compute_variant_accuracy(embeddings, meta)

    assert excluded == 1
    assert accuracy == 1.0  # only the 5-photo Royal-Gala variant contributes holdout queries


def test_compute_variant_accuracy_detects_confusable_variants():
    # Lindt's embeddings are closer to Cadbury's than to their own group -- should
    # mismatch when queried against the reference set.
    meta = pd.DataFrame([
        {"product_name": "Snacks/Chocolate-Bar/Cadbury", "file_name": f"cad{i}.jpg"} for i in range(4)
    ] + [
        {"product_name": "Snacks/Chocolate-Bar/Lindt", "file_name": f"lin{i}.jpg"} for i in range(4)
    ])
    embeddings = np.array([[1.0, 0.0]] * 4 + [[0.9, 0.1]] * 4, dtype=np.float32)

    accuracy, excluded = compute_variant_accuracy(embeddings, meta)

    assert accuracy < 1.0
    assert excluded == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/models/test_variant_auditor.py -v -k compute_variant_accuracy`
Expected: FAIL with `ImportError: cannot import name 'compute_variant_accuracy'`

- [ ] **Step 3: Write the implementation**

Append to `src/models/variant_auditor.py`:

```python
def compute_variant_accuracy(
    embeddings: np.ndarray, meta: pd.DataFrame, holdout_ratio: float = 0.2
) -> tuple[float, int]:
    """Holds out ~holdout_ratio of each eligible variant's photos, queries each
    held-out embedding against every reference embedding via the same cosine
    similarity nearest-neighbor VariantResolver.resolve() uses, and reports top-1
    accuracy. Variants with a single reference photo can't be held out and are
    reported as an excluded count, not folded into the accuracy denominator."""
    reference_idx, holdout_idx = split_holdout(meta, holdout_ratio)
    excluded_variants = meta["product_name"].value_counts()
    excluded_count = int((excluded_variants < 2).sum())

    if not holdout_idx:
        return 1.0, excluded_count

    reference_vectors = embeddings[reference_idx]
    reference_names = meta.loc[reference_idx, "product_name"].to_numpy()
    reference_norms = np.linalg.norm(reference_vectors, axis=1)

    correct = 0
    for idx in holdout_idx:
        query = embeddings[idx]
        query_norm = np.linalg.norm(query)
        similarities = reference_vectors @ query / (reference_norms * query_norm + 1e-8)
        best = int(np.argmax(similarities))
        if reference_names[best] == meta.loc[idx, "product_name"]:
            correct += 1

    return correct / len(holdout_idx), excluded_count
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/models/test_variant_auditor.py -v`
Expected: PASS (7 tests total)

- [ ] **Step 5: Commit**

```bash
git add src/models/variant_auditor.py tests/models/test_variant_auditor.py
git commit -m "feat: add DINOv2 held-out variant-accuracy computation"
```

---

### Task 5: Auto-labeler staging bookkeeping

**Files:**
- Create: `src/data/auto_labeler.py`
- Test: `tests/data/test_auto_labeler.py`

**Interfaces:**
- Produces: `get_autolabel_min_conf() -> float`, `pending_sidecars(staging_dir: pathlib.Path) -> list[pathlib.Path]`, `mark_consumed(sidecar_path: pathlib.Path) -> None`, `classify_capture(detections: list[dict], min_conf: float, max_conf: float) -> str` (returns `"zero_or_low"` or `"mid_band"`)

- [ ] **Step 1: Write the failing tests**

```python
# tests/data/test_auto_labeler.py
import json

from src.data.auto_labeler import (
    classify_capture,
    get_autolabel_min_conf,
    mark_consumed,
    pending_sidecars,
)


def test_get_autolabel_min_conf_default_and_override(monkeypatch):
    monkeypatch.delenv("SMARTCART_AUTOLABEL_MIN_CONF", raising=False)
    assert get_autolabel_min_conf() == 0.35

    monkeypatch.setenv("SMARTCART_AUTOLABEL_MIN_CONF", "0.4")
    assert get_autolabel_min_conf() == 0.4


def _write_sidecar(staging_dir, name, detections):
    staging_dir.mkdir(parents=True, exist_ok=True)
    (staging_dir / f"{name}.jpg").write_bytes(b"fake-jpg")
    sidecar = {
        "captured_at": "2026-07-11T00:00:00",
        "image_file": f"{name}.jpg",
        "image_width": 64,
        "image_height": 32,
        "num_detections": len(detections),
        "min_confidence": min((d["confidence"] for d in detections), default=None),
        "detections": detections,
    }
    (staging_dir / f"{name}.json").write_text(json.dumps(sidecar))
    return staging_dir / f"{name}.json"


def test_pending_sidecars_excludes_consumed_subdir(tmp_path):
    _write_sidecar(tmp_path, "one", [])
    consumed_dir = tmp_path / "consumed"
    _write_sidecar(consumed_dir, "already_done", [])

    pending = pending_sidecars(tmp_path)

    assert pending == [tmp_path / "one.json"]


def test_mark_consumed_moves_sidecar_but_leaves_image(tmp_path):
    sidecar_path = _write_sidecar(tmp_path, "one", [])

    mark_consumed(sidecar_path)

    assert not sidecar_path.exists()
    assert (tmp_path / "one.jpg").exists()
    assert (tmp_path / "consumed" / "one.json").exists()


def test_classify_capture_zero_or_low_when_no_detections():
    assert classify_capture([], min_conf=0.35, max_conf=0.5) == "zero_or_low"


def test_classify_capture_zero_or_low_when_any_detection_below_min_conf():
    detections = [{"confidence": 0.45}, {"confidence": 0.1}]

    assert classify_capture(detections, min_conf=0.35, max_conf=0.5) == "zero_or_low"


def test_classify_capture_mid_band_when_all_detections_in_band():
    detections = [{"confidence": 0.4}, {"confidence": 0.45}]

    assert classify_capture(detections, min_conf=0.35, max_conf=0.5) == "mid_band"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/data/test_auto_labeler.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data.auto_labeler'`

- [ ] **Step 3: Write the implementation**

```python
# src/data/auto_labeler.py
"""Auto-imports staged active-learning captures using a local VLM as an
independent second opinion, in place of Label Studio human review. Every
staged capture is either imported into the dataset tree or discarded --
never left pending, since there is no human to eventually review it."""
import os
from pathlib import Path


def get_autolabel_min_conf() -> float:
    return float(os.environ.get("SMARTCART_AUTOLABEL_MIN_CONF", "0.35"))


def pending_sidecars(staging_dir: Path) -> list[Path]:
    return sorted(p for p in staging_dir.glob("*.json") if p.parent == staging_dir)


def mark_consumed(sidecar_path: Path) -> None:
    consumed_dir = sidecar_path.parent / "consumed"
    consumed_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path.rename(consumed_dir / sidecar_path.name)


def classify_capture(detections: list[dict], min_conf: float, max_conf: float) -> str:
    """'zero_or_low' (no detections, or any detection below min_conf -- YOLO's own
    signal isn't trustworthy enough to cross-check, ask the VLM cold) vs 'mid_band'
    (every detection sits in [min_conf, max_conf) -- worth cross-checking each one
    individually against the VLM)."""
    if not detections or any(d["confidence"] < min_conf for d in detections):
        return "zero_or_low"
    return "mid_band"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/data/test_auto_labeler.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Add config to `.env.example` and commit**

Add to `.env.example`, after the VLM lines added in Task 1:

```
# Lower bound of the mid-confidence auto-label band (upper bound reuses SMARTCART_CAPTURE_CONF_THRESHOLD).
SMARTCART_AUTOLABEL_MIN_CONF=0.35
```

```bash
git add src/data/auto_labeler.py tests/data/test_auto_labeler.py .env.example
git commit -m "feat: add active-learning auto-labeler staging bookkeeping"
```

---

### Task 6: Mid-band cross-check import

**Files:**
- Modify: `src/data/auto_labeler.py`
- Test: `tests/data/test_auto_labeler.py`

**Interfaces:**
- Consumes: `crop_with_padding` from `src/data/annotation_import.py` ([annotation_import.py:23-36](../../../src/data/annotation_import.py#L23-L36)); `ask_vlm`, `match_category` from `src/models/vlm_verifier.py` (Tasks 1-2)
- Produces: `auto_import_mid_band_detection(image: PIL.Image.Image, detection: dict, class_names: list[str], dataset_root: pathlib.Path, capture_id: str) -> pathlib.Path | None`

- [ ] **Step 1: Write the failing tests**

Append to `tests/data/test_auto_labeler.py`:

```python
from unittest.mock import patch

from PIL import Image

from src.data.auto_labeler import auto_import_mid_band_detection


def test_auto_import_mid_band_detection_imports_when_vlm_agrees(tmp_path):
    image = Image.new("RGB", (100, 100), color=(255, 0, 0))
    detection = {"class_name": "Fruit/Apple/Royal-Gala", "confidence": 0.4, "bbox": [10, 10, 60, 60]}
    class_names = ["Fruit/Apple/Royal-Gala", "Snacks/Chocolate-Bar/Cadbury"]

    with patch("src.data.auto_labeler.ask_vlm", return_value="This looks like a Royal-Gala apple"):
        result = auto_import_mid_band_detection(image, detection, class_names, tmp_path, capture_id="cap1")

    expected_path = tmp_path / "Fruit/Apple/Royal-Gala" / "cap1.jpg"
    assert result == expected_path
    assert expected_path.exists()


def test_auto_import_mid_band_detection_discards_when_vlm_disagrees(tmp_path):
    image = Image.new("RGB", (100, 100))
    detection = {"class_name": "Fruit/Apple/Royal-Gala", "confidence": 0.4, "bbox": [10, 10, 60, 60]}
    class_names = ["Fruit/Apple/Royal-Gala", "Snacks/Chocolate-Bar/Cadbury"]

    with patch("src.data.auto_labeler.ask_vlm", return_value="This is a Cadbury chocolate bar"):
        result = auto_import_mid_band_detection(image, detection, class_names, tmp_path, capture_id="cap1")

    assert result is None
    assert not (tmp_path / "Fruit/Apple/Royal-Gala").exists()


def test_auto_import_mid_band_detection_discards_when_vlm_unavailable(tmp_path):
    image = Image.new("RGB", (100, 100))
    detection = {"class_name": "Fruit/Apple/Royal-Gala", "confidence": 0.4, "bbox": [10, 10, 60, 60]}

    with patch("src.data.auto_labeler.ask_vlm", return_value=None):
        result = auto_import_mid_band_detection(
            image, detection, ["Fruit/Apple/Royal-Gala"], tmp_path, capture_id="cap1"
        )

    assert result is None


def test_auto_import_mid_band_detection_discards_when_vlm_matches_unknown_category(tmp_path):
    image = Image.new("RGB", (100, 100))
    detection = {"class_name": "Fruit/Apple/Royal-Gala", "confidence": 0.4, "bbox": [10, 10, 60, 60]}

    with patch("src.data.auto_labeler.ask_vlm", return_value="I have no idea what this is"):
        result = auto_import_mid_band_detection(
            image, detection, ["Fruit/Apple/Royal-Gala"], tmp_path, capture_id="cap1"
        )

    assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/data/test_auto_labeler.py -v -k mid_band_detection`
Expected: FAIL with `ImportError: cannot import name 'auto_import_mid_band_detection'`

- [ ] **Step 3: Write the implementation**

Add imports and function to `src/data/auto_labeler.py`:

```python
from PIL import Image

from src.data.annotation_import import crop_with_padding
from src.models.vlm_verifier import ask_vlm, match_category

_MID_BAND_PROMPT_TEMPLATE = (
    "You are checking a product detector's guess. Which of these exact product "
    "categories does the boxed item show: {categories}? Answer with the exact "
    "category name only, or 'none' if you are not confident."
)


def auto_import_mid_band_detection(
    image: Image.Image,
    detection: dict,
    class_names: list[str],
    dataset_root: Path,
    capture_id: str,
) -> Path | None:
    """Cross-checks one mid-confidence YOLO detection against the VLM; imports
    the crop into dataset_root/<class_name>/ only if both agree."""
    crop = crop_with_padding(image, tuple(detection["bbox"]))
    prompt = _MID_BAND_PROMPT_TEMPLATE.format(categories=", ".join(class_names))
    answer = ask_vlm(crop, prompt)
    if answer is None:
        return None

    vlm_category = match_category(answer, class_names)
    if vlm_category is None or vlm_category != detection["class_name"]:
        return None

    dest_dir = dataset_root / vlm_category
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{capture_id}.jpg"
    crop.save(dest_path)
    return dest_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/data/test_auto_labeler.py -v`
Expected: PASS (11 tests total)

- [ ] **Step 5: Commit**

```bash
git add src/data/auto_labeler.py tests/data/test_auto_labeler.py
git commit -m "feat: add VLM cross-check auto-import for mid-confidence detections"
```

---

### Task 7: Low-signal frame import with box fallback

**Files:**
- Modify: `src/data/auto_labeler.py`
- Test: `tests/data/test_auto_labeler.py`

**Interfaces:**
- Consumes: `ask_vlm`, `match_category`, `parse_grounding_box` from `src/models/vlm_verifier.py`
- Produces: `auto_import_low_signal_frame(image: PIL.Image.Image, class_names: list[str], dataset_root: pathlib.Path, capture_id: str) -> pathlib.Path | None`

- [ ] **Step 1: Write the failing tests**

Append to `tests/data/test_auto_labeler.py`:

```python
from src.data.auto_labeler import auto_import_low_signal_frame


def test_auto_import_low_signal_frame_uses_parsed_box_when_valid(tmp_path):
    image = Image.new("RGB", (1000, 1000))
    class_names = ["Fruit/Apple/Royal-Gala"]

    with patch(
        "src.data.auto_labeler.ask_vlm",
        return_value="Fruit/Apple/Royal-Gala at (100,100),(500,500)",
    ):
        result = auto_import_low_signal_frame(image, class_names, tmp_path, capture_id="cap2")

    expected_path = tmp_path / "Fruit/Apple/Royal-Gala" / "cap2.jpg"
    assert result == expected_path
    with Image.open(expected_path) as cropped:
        assert cropped.size == (400, 400)  # (500-100, 500-100)


def test_auto_import_low_signal_frame_falls_back_to_whole_frame_without_valid_box(tmp_path):
    image = Image.new("RGB", (200, 100))
    class_names = ["Fruit/Apple/Royal-Gala"]

    with patch("src.data.auto_labeler.ask_vlm", return_value="Fruit/Apple/Royal-Gala, no coordinates given"):
        result = auto_import_low_signal_frame(image, class_names, tmp_path, capture_id="cap3")

    expected_path = tmp_path / "Fruit/Apple/Royal-Gala" / "cap3.jpg"
    assert result == expected_path
    with Image.open(expected_path) as saved:
        assert saved.size == (200, 100)  # whole frame, no crop


def test_auto_import_low_signal_frame_discards_when_no_category_match(tmp_path):
    image = Image.new("RGB", (100, 100))

    with patch("src.data.auto_labeler.ask_vlm", return_value="I don't recognize this item"):
        result = auto_import_low_signal_frame(image, ["Fruit/Apple/Royal-Gala"], tmp_path, capture_id="cap4")

    assert result is None


def test_auto_import_low_signal_frame_discards_when_vlm_unavailable(tmp_path):
    image = Image.new("RGB", (100, 100))

    with patch("src.data.auto_labeler.ask_vlm", return_value=None):
        result = auto_import_low_signal_frame(image, ["Fruit/Apple/Royal-Gala"], tmp_path, capture_id="cap5")

    assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/data/test_auto_labeler.py -v -k low_signal_frame`
Expected: FAIL with `ImportError: cannot import name 'auto_import_low_signal_frame'`

- [ ] **Step 3: Write the implementation**

Add to `src/data/auto_labeler.py`:

```python
from src.models.vlm_verifier import parse_grounding_box  # add to existing import line

_LOW_SIGNAL_PROMPT_TEMPLATE = (
    "Which of these exact product categories appears in this image: {categories}? "
    "If you can, also give its bounding box as (x1,y1),(x2,y2) on a 0-1000 scale. "
    "Answer 'none' if you are not confident."
)


def auto_import_low_signal_frame(
    image: Image.Image, class_names: list[str], dataset_root: Path, capture_id: str
) -> Path | None:
    """Asks the VLM cold (no YOLO signal to cross-check) for a category and an
    optional box over the whole frame. Falls back to the whole frame as the crop
    when no valid box comes back -- reasonable here since the app's own capture
    flow is already one held-up item per frame, not a wild guess."""
    prompt = _LOW_SIGNAL_PROMPT_TEMPLATE.format(categories=", ".join(class_names))
    answer = ask_vlm(image, prompt)
    if answer is None:
        return None

    vlm_category = match_category(answer, class_names)
    if vlm_category is None:
        return None

    box = parse_grounding_box(answer, image.width, image.height)
    crop = image.crop(box) if box is not None else image

    dest_dir = dataset_root / vlm_category
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{capture_id}.jpg"
    crop.save(dest_path)
    return dest_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/data/test_auto_labeler.py -v`
Expected: PASS (15 tests total)

- [ ] **Step 5: Commit**

```bash
git add src/data/auto_labeler.py tests/data/test_auto_labeler.py
git commit -m "feat: add VLM-only auto-import for zero/low-confidence frames"
```

---

### Task 8: Auto-labeler orchestration

**Files:**
- Modify: `src/data/auto_labeler.py`
- Test: `tests/data/test_auto_labeler.py`

**Interfaces:**
- Consumes: everything produced in Tasks 5-7 within the same module
- Produces: `auto_import_capture(sidecar_path: pathlib.Path, staging_dir: pathlib.Path, class_names: list[str], dataset_root: pathlib.Path, min_conf: float, max_conf: float) -> list[pathlib.Path]`, `auto_import_staging_dir(staging_dir: pathlib.Path, class_names: list[str], dataset_root: pathlib.Path, min_conf: float, max_conf: float) -> list[pathlib.Path]`

- [ ] **Step 1: Write the failing tests**

Append to `tests/data/test_auto_labeler.py`:

```python
from src.data.auto_labeler import auto_import_capture, auto_import_staging_dir


def test_auto_import_capture_imports_mid_band_and_marks_consumed(tmp_path):
    sidecar_path = _write_sidecar(
        tmp_path, "one", [{"class_name": "Fruit/Apple/Royal-Gala", "confidence": 0.4, "bbox": [0, 0, 50, 50]}]
    )
    dataset_root = tmp_path / "dataset"

    with patch("src.data.auto_labeler.ask_vlm", return_value="Fruit/Apple/Royal-Gala"):
        imported = auto_import_capture(
            sidecar_path, tmp_path, ["Fruit/Apple/Royal-Gala"], dataset_root, min_conf=0.35, max_conf=0.5
        )

    assert len(imported) == 1
    assert not sidecar_path.exists()
    assert (tmp_path / "consumed" / "one.json").exists()


def test_auto_import_capture_routes_zero_detection_to_low_signal_path(tmp_path):
    sidecar_path = _write_sidecar(tmp_path, "two", [])
    dataset_root = tmp_path / "dataset"

    with patch("src.data.auto_labeler.ask_vlm", return_value="Fruit/Apple/Royal-Gala"):
        imported = auto_import_capture(
            sidecar_path, tmp_path, ["Fruit/Apple/Royal-Gala"], dataset_root, min_conf=0.35, max_conf=0.5
        )

    assert len(imported) == 1
    assert (tmp_path / "consumed" / "two.json").exists()


def test_auto_import_capture_discards_multiple_mid_band_detections_independently(tmp_path):
    sidecar_path = _write_sidecar(
        tmp_path,
        "three",
        [
            {"class_name": "Fruit/Apple/Royal-Gala", "confidence": 0.4, "bbox": [0, 0, 20, 20]},
            {"class_name": "Snacks/Chocolate-Bar/Cadbury", "confidence": 0.45, "bbox": [20, 20, 40, 40]},
        ],
    )
    dataset_root = tmp_path / "dataset"
    class_names = ["Fruit/Apple/Royal-Gala", "Snacks/Chocolate-Bar/Cadbury"]

    answers = iter(["Fruit/Apple/Royal-Gala", "I don't know"])
    with patch("src.data.auto_labeler.ask_vlm", side_effect=lambda *a, **k: next(answers)):
        imported = auto_import_capture(sidecar_path, tmp_path, class_names, dataset_root, 0.35, 0.5)

    assert len(imported) == 1  # only the agreeing detection is imported
    assert (tmp_path / "consumed" / "three.json").exists()


def test_auto_import_staging_dir_processes_all_pending_and_returns_combined_list(tmp_path):
    _write_sidecar(tmp_path, "one", [{"class_name": "Fruit/Apple/Royal-Gala", "confidence": 0.4, "bbox": [0, 0, 20, 20]}])
    _write_sidecar(tmp_path, "two", [])
    dataset_root = tmp_path / "dataset"

    with patch("src.data.auto_labeler.ask_vlm", return_value="Fruit/Apple/Royal-Gala"):
        imported = auto_import_staging_dir(
            tmp_path, ["Fruit/Apple/Royal-Gala"], dataset_root, min_conf=0.35, max_conf=0.5
        )

    assert len(imported) == 2
    assert pending_sidecars(tmp_path) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/data/test_auto_labeler.py -v -k "auto_import_capture or auto_import_staging_dir"`
Expected: FAIL with `ImportError: cannot import name 'auto_import_capture'`

- [ ] **Step 3: Write the implementation**

Add to `src/data/auto_labeler.py`:

```python
import json


def load_capture(sidecar_path: Path) -> dict:
    return json.loads(sidecar_path.read_text())


def auto_import_capture(
    sidecar_path: Path,
    staging_dir: Path,
    class_names: list[str],
    dataset_root: Path,
    min_conf: float,
    max_conf: float,
) -> list[Path]:
    """Processes one staged capture end-to-end and always marks it consumed
    afterward, so the scheduler never reprocesses it regardless of outcome."""
    capture = load_capture(sidecar_path)
    image_path = staging_dir / capture["image_file"]
    imported: list[Path] = []

    with Image.open(image_path).convert("RGB") as image:
        detections = capture["detections"]
        capture_id = sidecar_path.stem
        if classify_capture(detections, min_conf, max_conf) == "zero_or_low":
            result = auto_import_low_signal_frame(image, class_names, dataset_root, capture_id)
            if result is not None:
                imported.append(result)
        else:
            for i, detection in enumerate(detections):
                result = auto_import_mid_band_detection(
                    image, detection, class_names, dataset_root, f"{capture_id}_{i}"
                )
                if result is not None:
                    imported.append(result)

    mark_consumed(sidecar_path)
    return imported


def auto_import_staging_dir(
    staging_dir: Path, class_names: list[str], dataset_root: Path, min_conf: float, max_conf: float
) -> list[Path]:
    imported: list[Path] = []
    for sidecar_path in pending_sidecars(staging_dir):
        imported.extend(auto_import_capture(sidecar_path, staging_dir, class_names, dataset_root, min_conf, max_conf))
    return imported
```

Note: the single-detection tests in Task 6 (`test_auto_import_mid_band_detection_*`) used `capture_id="cap1"` directly and expect `dataset_root / class / "cap1.jpg"`; this task's multi-detection orchestration calls it with an index-suffixed id (`f"{capture_id}_{i}"`) instead — both are valid uses of the same function signature, not a conflict.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/data/test_auto_labeler.py -v`
Expected: PASS (19 tests total)

- [ ] **Step 5: Commit**

```bash
git add src/data/auto_labeler.py tests/data/test_auto_labeler.py
git commit -m "feat: orchestrate auto-labeling across a full staging directory"
```

---

### Task 9: Candidate report and pipeline state persistence

**Files:**
- Create: `src/pipeline/candidate_promotion.py`
- Test: `tests/pipeline/test_candidate_promotion.py`

**Interfaces:**
- Produces: `get_promoted_state_path() -> pathlib.Path`, `read_promoted_state(state_path: pathlib.Path) -> dict`, `write_promoted_state(state_path: pathlib.Path, state: dict) -> None`, `write_candidate_report(candidate_dir: pathlib.Path, report: dict) -> pathlib.Path`

- [ ] **Step 1: Write the failing tests**

```python
# tests/pipeline/test_candidate_promotion.py
import json
from pathlib import Path

from src.pipeline.candidate_promotion import (
    get_promoted_state_path,
    read_promoted_state,
    write_candidate_report,
    write_promoted_state,
)


def test_get_promoted_state_path_default_and_override(monkeypatch):
    monkeypatch.delenv("SMARTCART_AL_STATE_PATH", raising=False)
    assert get_promoted_state_path() == Path("./artifacts/al_pipeline_state.json")

    monkeypatch.setenv("SMARTCART_AL_STATE_PATH", "/tmp/custom_state.json")
    assert get_promoted_state_path() == Path("/tmp/custom_state.json")


def test_read_promoted_state_returns_defaults_when_missing(tmp_path):
    state = read_promoted_state(tmp_path / "missing.json")

    assert state == {
        "run_name": None,
        "map50": 0.0,
        "variant_accuracy": 0.0,
        "pending_auto_imported": 0,
    }


def test_write_then_read_promoted_state_round_trips(tmp_path):
    state_path = tmp_path / "state.json"
    write_promoted_state(state_path, {"run_name": "retrain_1", "map50": 0.8, "variant_accuracy": 0.9, "pending_auto_imported": 3})

    result = read_promoted_state(state_path)

    assert result == {"run_name": "retrain_1", "map50": 0.8, "variant_accuracy": 0.9, "pending_auto_imported": 3}


def test_write_candidate_report_writes_json_to_candidate_dir(tmp_path):
    candidate_dir = tmp_path / "retrain_1"
    candidate_dir.mkdir()
    report = {"run_name": "retrain_1", "map50": 0.8, "passed": True}

    path = write_candidate_report(candidate_dir, report)

    assert path == candidate_dir / "report.json"
    assert json.loads(path.read_text()) == report
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/pipeline/test_candidate_promotion.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.pipeline.candidate_promotion'`

- [ ] **Step 3: Write the implementation**

```python
# src/pipeline/candidate_promotion.py
"""Candidate report and pipeline-state persistence for the autonomous retrain
loop, plus the mechanical promotion decision and file operations that move a
passing candidate into the live artifacts/weights path."""
import json
import os
from pathlib import Path

DEFAULT_STATE_PATH = Path("./artifacts/al_pipeline_state.json")
_DEFAULT_STATE = {"run_name": None, "map50": 0.0, "variant_accuracy": 0.0, "pending_auto_imported": 0}


def get_promoted_state_path() -> Path:
    return Path(os.environ.get("SMARTCART_AL_STATE_PATH", str(DEFAULT_STATE_PATH)))


def read_promoted_state(state_path: Path) -> dict:
    if not state_path.exists():
        return dict(_DEFAULT_STATE)
    return {**_DEFAULT_STATE, **json.loads(state_path.read_text())}


def write_promoted_state(state_path: Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state))


def write_candidate_report(candidate_dir: Path, report: dict) -> Path:
    report_path = candidate_dir / "report.json"
    report_path.write_text(json.dumps(report))
    return report_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/pipeline/test_candidate_promotion.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/candidate_promotion.py tests/pipeline/test_candidate_promotion.py
git commit -m "feat: add active-learning pipeline state and candidate report persistence"
```

---

### Task 10: Promotion decision and file operations

**Files:**
- Modify: `src/pipeline/candidate_promotion.py`
- Test: `tests/pipeline/test_candidate_promotion.py`

**Interfaces:**
- Consumes: `write_promoted_state` from Task 9
- Produces: `should_promote(candidate_map50: float, candidate_variant_acc: float, live_map50: float, live_variant_acc: float) -> bool`, `update_env_weights_path(env_path: pathlib.Path, new_value: str) -> None`, `promote(candidate_dir: pathlib.Path, run_name: str, weights_path: pathlib.Path, map50: float, variant_acc: float, artifacts_dir: pathlib.Path, env_path: pathlib.Path, state_path: pathlib.Path) -> None`

- [ ] **Step 1: Write the failing tests**

Append to `tests/pipeline/test_candidate_promotion.py`:

```python
from src.pipeline.candidate_promotion import promote, should_promote, update_env_weights_path


def test_should_promote_true_when_both_metrics_improve_or_tie():
    assert should_promote(candidate_map50=0.8, candidate_variant_acc=0.9, live_map50=0.7, live_variant_acc=0.9) is True


def test_should_promote_false_when_either_metric_regresses():
    assert should_promote(0.6, 0.9, live_map50=0.7, live_variant_acc=0.9) is False
    assert should_promote(0.8, 0.5, live_map50=0.7, live_variant_acc=0.6) is False


def test_update_env_weights_path_replaces_existing_line(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("SMARTCART_PORT=8000\nSMARTCART_WEIGHTS_PATH=./old/path.pt\nSMARTCART_HOST=0.0.0.0\n")

    update_env_weights_path(env_path, "./new/path.pt")

    lines = env_path.read_text().splitlines()
    assert "SMARTCART_WEIGHTS_PATH=./new/path.pt" in lines
    assert "SMARTCART_PORT=8000" in lines
    assert "SMARTCART_HOST=0.0.0.0" in lines


def test_update_env_weights_path_appends_when_missing(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("SMARTCART_PORT=8000\n")

    update_env_weights_path(env_path, "./new/path.pt")

    assert "SMARTCART_WEIGHTS_PATH=./new/path.pt" in env_path.read_text().splitlines()


def test_promote_copies_candidate_files_updates_env_and_state(tmp_path):
    candidate_dir = tmp_path / "candidates" / "retrain_1"
    candidate_dir.mkdir(parents=True)
    (candidate_dir / "catalog_prices.csv").write_text("class_id,product_name,price_usd\n")
    (candidate_dir / "gallery_index.npy").write_bytes(b"fake-npy")
    (candidate_dir / "gallery_meta.csv").write_text("class_id,product_name,file_name\n")

    artifacts_dir = tmp_path / "artifacts"
    env_path = tmp_path / ".env"
    env_path.write_text("SMARTCART_WEIGHTS_PATH=./old/best.pt\n")
    state_path = tmp_path / "state.json"

    promote(
        candidate_dir=candidate_dir,
        run_name="retrain_1",
        weights_path=tmp_path / "runs" / "retrain_1" / "weights" / "best.pt",
        map50=0.8,
        variant_acc=0.9,
        artifacts_dir=artifacts_dir,
        env_path=env_path,
        state_path=state_path,
    )

    assert (artifacts_dir / "catalog_prices.csv").exists()
    assert (artifacts_dir / "gallery_index.npy").exists()
    assert (artifacts_dir / "gallery_meta.csv").exists()
    assert f"SMARTCART_WEIGHTS_PATH={tmp_path / 'runs' / 'retrain_1' / 'weights' / 'best.pt'}" in env_path.read_text()

    state = json.loads(state_path.read_text())
    assert state["run_name"] == "retrain_1"
    assert state["map50"] == 0.8
    assert state["variant_accuracy"] == 0.9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/pipeline/test_candidate_promotion.py -v -k "should_promote or update_env or test_promote"`
Expected: FAIL with `ImportError: cannot import name 'should_promote'`

- [ ] **Step 3: Write the implementation**

Add to `src/pipeline/candidate_promotion.py`:

```python
import shutil

_CANDIDATE_FILES = ("catalog_prices.csv", "gallery_index.npy", "gallery_meta.csv")


def should_promote(
    candidate_map50: float, candidate_variant_acc: float, live_map50: float, live_variant_acc: float
) -> bool:
    return candidate_map50 >= live_map50 and candidate_variant_acc >= live_variant_acc


def update_env_weights_path(env_path: Path, new_value: str) -> None:
    lines = env_path.read_text().splitlines() if env_path.exists() else []
    for i, line in enumerate(lines):
        if line.startswith("SMARTCART_WEIGHTS_PATH="):
            lines[i] = f"SMARTCART_WEIGHTS_PATH={new_value}"
            break
    else:
        lines.append(f"SMARTCART_WEIGHTS_PATH={new_value}")
    env_path.write_text("\n".join(lines) + "\n")


def promote(
    candidate_dir: Path,
    run_name: str,
    weights_path: Path,
    map50: float,
    variant_acc: float,
    artifacts_dir: Path,
    env_path: Path,
    state_path: Path,
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for file_name in _CANDIDATE_FILES:
        shutil.copyfile(candidate_dir / file_name, artifacts_dir / file_name)

    update_env_weights_path(env_path, str(weights_path))
    write_promoted_state(
        state_path,
        {"run_name": run_name, "map50": map50, "variant_accuracy": variant_acc, "pending_auto_imported": 0},
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/pipeline/test_candidate_promotion.py -v`
Expected: PASS (9 tests total)

- [ ] **Step 5: Add config to `.env.example` and commit**

Add to `.env.example`:

```
# Absolute floor for the DINOv2 held-out variant-accuracy gate (see SMARTCART_RETRAIN_MIN_MAP50 for the YOLO equivalent).
SMARTCART_RETRAIN_MIN_VARIANT_ACC=0.5
```

```bash
git add src/pipeline/candidate_promotion.py tests/pipeline/test_candidate_promotion.py .env.example
git commit -m "feat: add mechanical promotion decision and candidate file promotion"
```

---

### Task 11: Scheduler orchestration

**Files:**
- Create: `src/pipeline/al_scheduler.py`
- Create: `al_scheduler_check.py`
- Test: `tests/pipeline/test_al_scheduler.py`

**Interfaces:**
- Consumes: `GroceryDatasetIndexer` ([gallery.py:95-112](../../../src/data/gallery.py#L95-L112)); `ModelTrainingPipeline`, `get_retrain_min_map50` ([training_pipeline.py](../../../src/pipeline/training_pipeline.py)) — used unmodified; `compute_variant_accuracy` (Task 4); `auto_import_staging_dir`, `get_autolabel_min_conf` (Task 8); `get_capture_dir`, `get_capture_threshold` ([active_learning_capture.py:15-20](../../../src/deploy/active_learning_capture.py#L15-L20)); `read_promoted_state`, `write_promoted_state`, `write_candidate_report`, `should_promote`, `promote` (Tasks 9-10)
- Produces: `get_retrain_trigger_count() -> int`, `run_retrain_cycle(dataset_root: pathlib.Path, candidates_root: pathlib.Path, runs_dir: pathlib.Path, synth_root: pathlib.Path) -> tuple[dict, pathlib.Path]`, `restart_api_service() -> None`, `run_scheduler_tick(dataset_root: pathlib.Path, staging_dir: pathlib.Path, artifacts_dir: pathlib.Path, candidates_root: pathlib.Path, runs_dir: pathlib.Path, synth_root: pathlib.Path, env_path: pathlib.Path, state_path: pathlib.Path) -> dict`

- [ ] **Step 1: Write the failing tests**

```python
# tests/pipeline/test_al_scheduler.py
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.pipeline.al_scheduler import (
    get_retrain_trigger_count,
    run_retrain_cycle,
    run_scheduler_tick,
)


def test_get_retrain_trigger_count_default_and_override(monkeypatch):
    monkeypatch.delenv("SMARTCART_AL_RETRAIN_TRIGGER_COUNT", raising=False)
    assert get_retrain_trigger_count() == 50

    monkeypatch.setenv("SMARTCART_AL_RETRAIN_TRIGGER_COUNT", "10")
    assert get_retrain_trigger_count() == 10


def test_run_retrain_cycle_renames_temp_candidate_dir_to_run_name(tmp_path):
    candidates_root = tmp_path / "candidates"

    with patch("src.pipeline.al_scheduler.ModelTrainingPipeline") as mock_pipeline_cls:
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {
            "run_name": "retrain_20260711_000000",
            "weights_path": Path("weights/best.pt"),
            "data_yaml": Path("data.yaml"),
            "passed": True,
            "map50": 0.8,
            "map50_95": 0.6,
            "min_map50": 0.5,
        }
        mock_pipeline_cls.return_value = mock_pipeline
        # simulate the pipeline actually writing gallery files into the tmp candidate dir
        def fake_run(**kwargs):
            tmp_candidate = candidates_root / "_tmp_candidate"
            tmp_candidate.mkdir(parents=True, exist_ok=True)
            (tmp_candidate / "gallery_index.npy").write_bytes(b"fake")
            return mock_pipeline.run.return_value
        mock_pipeline.run.side_effect = fake_run

        result, candidate_dir = run_retrain_cycle(
            dataset_root=tmp_path / "dataset",
            candidates_root=candidates_root,
            runs_dir=tmp_path / "runs",
            synth_root=tmp_path / "synth",
        )

    assert candidate_dir == candidates_root / "retrain_20260711_000000"
    assert candidate_dir.exists()
    assert (candidate_dir / "gallery_index.npy").exists()
    assert not (candidates_root / "_tmp_candidate").exists()
    assert result["run_name"] == "retrain_20260711_000000"


def test_run_scheduler_tick_skips_retrain_below_trigger_count(tmp_path, monkeypatch):
    monkeypatch.setenv("SMARTCART_AL_RETRAIN_TRIGGER_COUNT", "50")
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()

    with patch("src.pipeline.al_scheduler.GroceryDatasetIndexer") as mock_indexer_cls, \
         patch("src.pipeline.al_scheduler.auto_import_staging_dir", return_value=[Path("a.jpg")]) as mock_auto_import, \
         patch("src.pipeline.al_scheduler.run_retrain_cycle") as mock_retrain:
        mock_indexer_cls.return_value.build_class_map.return_value = {"Fruit/Apple/Royal-Gala": 0}

        result = run_scheduler_tick(
            dataset_root=tmp_path / "dataset",
            staging_dir=staging_dir,
            artifacts_dir=tmp_path / "artifacts",
            candidates_root=tmp_path / "candidates",
            runs_dir=tmp_path / "runs",
            synth_root=tmp_path / "synth",
            env_path=tmp_path / ".env",
            state_path=tmp_path / "state.json",
        )

    mock_auto_import.assert_called_once()
    mock_retrain.assert_not_called()
    assert result["retrained"] is False
    assert result["auto_imported"] == 1


def test_run_scheduler_tick_promotes_when_candidate_beats_live_metrics(tmp_path, monkeypatch):
    monkeypatch.setenv("SMARTCART_AL_RETRAIN_TRIGGER_COUNT", "1")
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    candidate_dir = tmp_path / "candidates" / "retrain_1"
    candidate_dir.mkdir(parents=True)
    for fname in ("catalog_prices.csv", "gallery_index.npy", "gallery_meta.csv"):
        (candidate_dir / fname).write_bytes(b"fake")

    with patch("src.pipeline.al_scheduler.GroceryDatasetIndexer") as mock_indexer_cls, \
         patch("src.pipeline.al_scheduler.auto_import_staging_dir", return_value=[Path("a.jpg")]), \
         patch("src.pipeline.al_scheduler.run_retrain_cycle") as mock_retrain, \
         patch("src.pipeline.al_scheduler.compute_variant_accuracy", return_value=(0.9, 0)), \
         patch("src.pipeline.al_scheduler.restart_api_service") as mock_restart, \
         patch("src.pipeline.al_scheduler.pd.read_csv"), \
         patch("src.pipeline.al_scheduler.np.load"):
        mock_indexer_cls.return_value.build_class_map.return_value = {"Fruit/Apple/Royal-Gala": 0}
        mock_retrain.return_value = (
            {
                "run_name": "retrain_1",
                "weights_path": Path("weights/best.pt"),
                "passed": True,
                "map50": 0.9,
                "min_map50": 0.5,
            },
            candidate_dir,
        )

        result = run_scheduler_tick(
            dataset_root=tmp_path / "dataset",
            staging_dir=staging_dir,
            artifacts_dir=tmp_path / "artifacts",
            candidates_root=tmp_path / "candidates",
            runs_dir=tmp_path / "runs",
            synth_root=tmp_path / "synth",
            env_path=tmp_path / ".env",
            state_path=tmp_path / "state.json",
        )

    assert result["retrained"] is True
    assert result["promoted"] is True
    mock_restart.assert_called_once()
    assert (tmp_path / "artifacts" / "catalog_prices.csv").exists()


def test_run_scheduler_tick_does_not_promote_when_map50_gate_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("SMARTCART_AL_RETRAIN_TRIGGER_COUNT", "1")
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    candidate_dir = tmp_path / "candidates" / "retrain_1"
    candidate_dir.mkdir(parents=True)

    with patch("src.pipeline.al_scheduler.GroceryDatasetIndexer") as mock_indexer_cls, \
         patch("src.pipeline.al_scheduler.auto_import_staging_dir", return_value=[Path("a.jpg")]), \
         patch("src.pipeline.al_scheduler.run_retrain_cycle") as mock_retrain, \
         patch("src.pipeline.al_scheduler.compute_variant_accuracy", return_value=(0.9, 0)), \
         patch("src.pipeline.al_scheduler.restart_api_service") as mock_restart, \
         patch("src.pipeline.al_scheduler.pd.read_csv"), \
         patch("src.pipeline.al_scheduler.np.load"):
        mock_indexer_cls.return_value.build_class_map.return_value = {"Fruit/Apple/Royal-Gala": 0}
        mock_retrain.return_value = (
            {"run_name": "retrain_1", "weights_path": Path("weights/best.pt"), "passed": False, "map50": 0.3, "min_map50": 0.5},
            candidate_dir,
        )

        result = run_scheduler_tick(
            dataset_root=tmp_path / "dataset",
            staging_dir=staging_dir,
            artifacts_dir=tmp_path / "artifacts",
            candidates_root=tmp_path / "candidates",
            runs_dir=tmp_path / "runs",
            synth_root=tmp_path / "synth",
            env_path=tmp_path / ".env",
            state_path=tmp_path / "state.json",
        )

    assert result["promoted"] is False
    mock_restart.assert_not_called()
    assert not (tmp_path / "artifacts" / "catalog_prices.csv").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/pipeline/test_al_scheduler.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.pipeline.al_scheduler'`

- [ ] **Step 3: Write the implementation**

```python
# src/pipeline/al_scheduler.py
"""Orchestrates one autonomous active-learning scheduler tick: auto-import staged
captures via the VLM, and if enough new crops have accumulated, run a full
retrain cycle against an isolated candidate directory, gate it on both YOLO
mAP50 and DINOv2 held-out variant accuracy, and auto-promote if it doesn't
regress either metric versus the currently-live model."""
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.auto_labeler import auto_import_staging_dir, get_autolabel_min_conf
from src.data.gallery import GroceryDatasetIndexer
from src.deploy.active_learning_capture import get_capture_threshold
from src.models.variant_auditor import compute_variant_accuracy
from src.pipeline.candidate_promotion import (
    promote,
    read_promoted_state,
    should_promote,
    write_candidate_report,
    write_promoted_state,
)
from src.pipeline.training_pipeline import ModelTrainingPipeline


def get_retrain_trigger_count() -> int:
    return int(os.environ.get("SMARTCART_AL_RETRAIN_TRIGGER_COUNT", "50"))


def run_retrain_cycle(
    dataset_root: Path, candidates_root: Path, runs_dir: Path, synth_root: Path
) -> tuple[dict, Path]:
    """Reuses ModelTrainingPipeline unmodified against a temp candidate directory,
    then renames it to the run's own name -- keeps candidate artifacts fully
    isolated from the live artifacts/ path without touching the pipeline class."""
    tmp_candidate_dir = candidates_root / "_tmp_candidate"
    shutil.rmtree(tmp_candidate_dir, ignore_errors=True)

    pipeline = ModelTrainingPipeline(
        dataset_root=dataset_root, artifacts_dir=tmp_candidate_dir, runs_dir=runs_dir, synth_root=synth_root
    )
    result = pipeline.run()

    candidate_dir = candidates_root / result["run_name"]
    tmp_candidate_dir.rename(candidate_dir)
    return result, candidate_dir


def restart_api_service() -> None:
    subprocess.run(["launchctl", "kickstart", "-k", "com.smartcart.api"], check=False)


def run_scheduler_tick(
    dataset_root: Path,
    staging_dir: Path,
    artifacts_dir: Path,
    candidates_root: Path,
    runs_dir: Path,
    synth_root: Path,
    env_path: Path,
    state_path: Path,
) -> dict:
    class_map = GroceryDatasetIndexer(dataset_root).build_class_map()
    class_names = list(class_map)

    imported = auto_import_staging_dir(
        staging_dir, class_names, dataset_root, get_autolabel_min_conf(), get_capture_threshold()
    )

    state = read_promoted_state(state_path)
    pending = state["pending_auto_imported"] + len(imported)

    if pending < get_retrain_trigger_count():
        write_promoted_state(state_path, {**state, "pending_auto_imported": pending})
        return {"retrained": False, "auto_imported": len(imported)}

    train_result, candidate_dir = run_retrain_cycle(dataset_root, candidates_root, runs_dir, synth_root)

    embeddings = np.load(candidate_dir / "gallery_index.npy")
    meta = pd.read_csv(candidate_dir / "gallery_meta.csv")
    variant_acc, excluded = compute_variant_accuracy(embeddings, meta)

    promoted = should_promote(
        candidate_map50=train_result["map50"],
        candidate_variant_acc=variant_acc,
        live_map50=state["map50"],
        live_variant_acc=state["variant_accuracy"],
    ) and train_result["passed"]

    write_candidate_report(
        candidate_dir,
        {
            **train_result,
            "weights_path": str(train_result["weights_path"]),
            "data_yaml": str(train_result.get("data_yaml", "")),
            "variant_accuracy": variant_acc,
            "variant_excluded_count": excluded,
            "promoted": promoted,
            "auto_imported_count": len(imported),
        },
    )

    if promoted:
        promote(
            candidate_dir=candidate_dir,
            run_name=train_result["run_name"],
            weights_path=train_result["weights_path"],
            map50=train_result["map50"],
            variant_acc=variant_acc,
            artifacts_dir=artifacts_dir,
            env_path=env_path,
            state_path=state_path,
        )
        restart_api_service()
    else:
        write_promoted_state(state_path, {**state, "pending_auto_imported": 0})

    return {"retrained": True, "auto_imported": len(imported), "promoted": promoted, "run_name": train_result["run_name"]}
```

```python
# al_scheduler_check.py
"""Thin entrypoint for the launchd-scheduled active-learning checker. All
logic lives in src/pipeline/al_scheduler.py -- see that module's docstring."""
from pathlib import Path

from dotenv import load_dotenv

from src.pipeline.al_scheduler import run_scheduler_tick
from src.deploy.active_learning_capture import get_capture_dir
from src.pipeline.candidate_promotion import get_promoted_state_path

DATASET_ROOT = Path("./dataset/GroceryStoreDataset/dataset/train")

if __name__ == "__main__":
    load_dotenv()
    result = run_scheduler_tick(
        dataset_root=DATASET_ROOT,
        staging_dir=get_capture_dir(),
        artifacts_dir=Path("./artifacts"),
        candidates_root=Path("./artifacts/candidates"),
        runs_dir=Path("./runs/detect"),
        synth_root=Path("./synthetic_dataset_retrain"),
        env_path=Path(".env"),
        state_path=get_promoted_state_path(),
    )
    print(result)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/pipeline/test_al_scheduler.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Run the full suite, add config to `.env.example`, and commit**

Add to `.env.example`:

```
# Newly auto-imported crops needed since the last retrain attempt to trigger the next one.
SMARTCART_AL_RETRAIN_TRIGGER_COUNT=50
# Optional override for where the active-learning pipeline's state file lives.
SMARTCART_AL_STATE_PATH=./artifacts/al_pipeline_state.json
```

Run: `uv run pytest -v`
Expected: PASS (all tests, including all prior tasks' tests)

```bash
git add src/pipeline/al_scheduler.py al_scheduler_check.py tests/pipeline/test_al_scheduler.py .env.example
git commit -m "feat: add autonomous active-learning scheduler orchestration"
```

---

### Task 12: launchd process management

**Files:**
- Create: `launchd/com.smartcart.api.plist`
- Create: `launchd/com.smartcart.al-scheduler.plist`
- Create: `launchd/README.md`

**Interfaces:**
- Consumes: `main_api_server.py` (existing, unmodified), `al_scheduler_check.py` (Task 11)
- Produces: two installable launchd service definitions

- [ ] **Step 1: Write the API server plist**

```xml
<!-- launchd/com.smartcart.api.plist -->
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.smartcart.api</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/env</string>
        <string>uv</string>
        <string>run</string>
        <string>python</string>
        <string>main_api_server.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>__REPO_ROOT__</string>
    <key>KeepAlive</key>
    <true/>
    <key>RunAtLoad</key>
    <true/>
    <key>StandardOutPath</key>
    <string>__REPO_ROOT__/artifacts/logs/api.log</string>
    <key>StandardErrorPath</key>
    <string>__REPO_ROOT__/artifacts/logs/api.error.log</string>
</dict>
</plist>
```

- [ ] **Step 2: Write the scheduler plist**

```xml
<!-- launchd/com.smartcart.al-scheduler.plist -->
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.smartcart.al-scheduler</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/env</string>
        <string>uv</string>
        <string>run</string>
        <string>python</string>
        <string>al_scheduler_check.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>__REPO_ROOT__</string>
    <key>StartInterval</key>
    <integer>900</integer>
    <key>StandardOutPath</key>
    <string>__REPO_ROOT__/artifacts/logs/al-scheduler.log</string>
    <key>StandardErrorPath</key>
    <string>__REPO_ROOT__/artifacts/logs/al-scheduler.error.log</string>
</dict>
</plist>
```

- [ ] **Step 3: Write install instructions**

```markdown
<!-- launchd/README.md -->
# launchd services

Two services replace running the API server and the active-learning scheduler manually.

## Install

Both plists use `__REPO_ROOT__` as a placeholder. Replace it with this repo's absolute path, then load:

    sed "s|__REPO_ROOT__|$(pwd)|g" launchd/com.smartcart.api.plist > ~/Library/LaunchAgents/com.smartcart.api.plist
    sed "s|__REPO_ROOT__|$(pwd)|g" launchd/com.smartcart.al-scheduler.plist > ~/Library/LaunchAgents/com.smartcart.al-scheduler.plist
    mkdir -p artifacts/logs
    launchctl load ~/Library/LaunchAgents/com.smartcart.api.plist
    launchctl load ~/Library/LaunchAgents/com.smartcart.al-scheduler.plist

## Verify

    curl http://localhost:8000/health          # API server responding
    tail -f artifacts/logs/al-scheduler.log    # scheduler running on its StartInterval

## Uninstall

    launchctl unload ~/Library/LaunchAgents/com.smartcart.api.plist
    launchctl unload ~/Library/LaunchAgents/com.smartcart.al-scheduler.plist
    rm ~/Library/LaunchAgents/com.smartcart.{api,al-scheduler}.plist

## Change the scheduler interval

Edit `StartInterval` (seconds) in `com.smartcart.al-scheduler.plist` before installing, or edit the installed copy under `~/Library/LaunchAgents/` and reload.
```

- [ ] **Step 4: Manually verify (not part of the automated test suite)**

Run the install commands from Step 3 against a real checkout of this repo, then:

Run: `curl http://localhost:8000/health`
Expected: `{"status":"ok"}`

Run: `sleep 5 && cat artifacts/logs/al-scheduler.log`
Expected: at least one line of output from `al_scheduler_check.py`'s `print(result)` (may show `{"retrained": False, "auto_imported": 0}` if the staging dir is empty, which is correct)

- [ ] **Step 5: Commit**

```bash
git add launchd/
git commit -m "feat: add launchd services for the API server and active-learning scheduler"
```

---

## Self-Review

**Spec coverage:**
- VLM auto-importer (mid-band cross-check, zero/low-signal fallback, discard-on-disagreement) → Tasks 5-8. ✓
- DINOv2 held-out variant-accuracy gate → Tasks 3-4. ✓
- Candidate staging fix (isolate gallery/catalog from live `./artifacts/`) → Task 11's `run_retrain_cycle` (temp dir + rename, `ModelTrainingPipeline` untouched). ✓
- Mechanical auto-promotion + state tracking → Tasks 9-10, wired in Task 11. ✓
- launchd process management + auto-restart on promotion → Task 11 (`restart_api_service`) + Task 12 (plists). ✓
- Config additions → each task adds its own `.env.example` lines in the same commit. ✓
- Passive audit trail (`SMARTCART_AL_AUDIT_SAMPLE_RATE`, `artifacts/al_review/`) from the spec is **not** included in this plan — flagging as a gap.

**Fixing the gap:** the spec's error-handling and config sections both call for a sampled, non-blocking review log. Adding it now as Task 8.5.

**Placeholder scan:** no TBD/TODO markers found; every step has complete code.

**Type consistency check:** `auto_import_capture` returns `list[Path]` (Task 8) matching `auto_import_staging_dir`'s `.extend()` usage; `run_retrain_cycle` returns `tuple[dict, Path]` matching `run_scheduler_tick`'s unpacking; `promote()`'s parameter names (`candidate_dir`, `run_name`, `weights_path`, `map50`, `variant_acc`, `artifacts_dir`, `env_path`, `state_path`) match exactly between Task 10's definition and Task 11's call site. `should_promote`'s parameter names (`candidate_map50`, `candidate_variant_acc`, `live_map50`, `live_variant_acc`) match between Task 10 and Task 11. Confirmed consistent throughout.

---

### Task 8.5: Passive audit trail for auto-import decisions

**Files:**
- Modify: `src/data/auto_labeler.py`
- Test: `tests/data/test_auto_labeler.py`

**Interfaces:**
- Consumes: nothing new
- Produces: `get_audit_sample_rate() -> float`, `should_sample(capture_id: str, sample_rate: float) -> bool`, `log_review_sample(review_dir: Path, capture_id: str, image: PIL.Image.Image, decision: dict) -> None`

- [ ] **Step 1: Write the failing tests**

Append to `tests/data/test_auto_labeler.py`:

```python
from src.data.auto_labeler import get_audit_sample_rate, log_review_sample, should_sample


def test_get_audit_sample_rate_default_and_override(monkeypatch):
    monkeypatch.delenv("SMARTCART_AL_AUDIT_SAMPLE_RATE", raising=False)
    assert get_audit_sample_rate() == 0.05

    monkeypatch.setenv("SMARTCART_AL_AUDIT_SAMPLE_RATE", "0.5")
    assert get_audit_sample_rate() == 0.5


def test_should_sample_is_deterministic_per_capture_id():
    # same capture_id always yields the same sampling decision
    first = should_sample("abc123", sample_rate=0.5)
    second = should_sample("abc123", sample_rate=0.5)
    assert first == second


def test_should_sample_respects_zero_and_one_bounds():
    assert should_sample("any-id", sample_rate=0.0) is False
    assert should_sample("any-id", sample_rate=1.0) is True


def test_log_review_sample_writes_image_and_decision_json(tmp_path):
    image = Image.new("RGB", (10, 10), color=(0, 255, 0))
    decision = {"vlm_answer": "Fruit/Apple/Royal-Gala", "yolo_class": "Fruit/Apple/Royal-Gala", "outcome": "imported"}

    log_review_sample(tmp_path, "cap1", image, decision)

    assert (tmp_path / "cap1.jpg").exists()
    assert json.loads((tmp_path / "cap1.json").read_text()) == decision
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/data/test_auto_labeler.py -v -k "sample_rate or should_sample or log_review"`
Expected: FAIL with `ImportError: cannot import name 'get_audit_sample_rate'`

- [ ] **Step 3: Write the implementation**

Add to `src/data/auto_labeler.py`:

```python
import hashlib


def get_audit_sample_rate() -> float:
    return float(os.environ.get("SMARTCART_AL_AUDIT_SAMPLE_RATE", "0.05"))


def should_sample(capture_id: str, sample_rate: float) -> bool:
    """Deterministic per-capture sampling (same capture_id always samples the
    same way) so re-running a scheduler tick doesn't churn the review log."""
    if sample_rate <= 0.0:
        return False
    if sample_rate >= 1.0:
        return True
    digest = int(hashlib.sha256(capture_id.encode()).hexdigest(), 16)
    return (digest % 10_000) / 10_000 < sample_rate


def log_review_sample(review_dir: Path, capture_id: str, image: Image.Image, decision: dict) -> None:
    """Writes a sampled auto-import decision for optional, non-blocking human
    spot-checking. Nothing reads this directory automatically."""
    review_dir.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(review_dir / f"{capture_id}.jpg")
    (review_dir / f"{capture_id}.json").write_text(json.dumps(decision))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/data/test_auto_labeler.py -v`
Expected: PASS (23 tests total)

- [ ] **Step 5: Wire sampling into `auto_import_capture`, add config, and commit**

Modify `auto_import_capture` in `src/data/auto_labeler.py` to log a sample after each decision:

```python
def auto_import_capture(
    sidecar_path: Path,
    staging_dir: Path,
    class_names: list[str],
    dataset_root: Path,
    min_conf: float,
    max_conf: float,
    review_dir: Path | None = None,
) -> list[Path]:
    """Processes one staged capture end-to-end and always marks it consumed
    afterward, so the scheduler never reprocesses it regardless of outcome."""
    capture = load_capture(sidecar_path)
    image_path = staging_dir / capture["image_file"]
    imported: list[Path] = []

    with Image.open(image_path).convert("RGB") as image:
        detections = capture["detections"]
        capture_id = sidecar_path.stem
        category = classify_capture(detections, min_conf, max_conf)

        if category == "zero_or_low":
            result = auto_import_low_signal_frame(image, class_names, dataset_root, capture_id)
            if result is not None:
                imported.append(result)
            outcome = [{"detection": None, "outcome": "imported" if result else "discarded"}]
        else:
            outcome = []
            for i, detection in enumerate(detections):
                result = auto_import_mid_band_detection(
                    image, detection, class_names, dataset_root, f"{capture_id}_{i}"
                )
                if result is not None:
                    imported.append(result)
                outcome.append({"detection": detection["class_name"], "outcome": "imported" if result else "discarded"})

        if review_dir is not None and should_sample(capture_id, get_audit_sample_rate()):
            log_review_sample(review_dir, capture_id, image, {"category": category, "decisions": outcome})

    mark_consumed(sidecar_path)
    return imported


def auto_import_staging_dir(
    staging_dir: Path,
    class_names: list[str],
    dataset_root: Path,
    min_conf: float,
    max_conf: float,
    review_dir: Path | None = None,
) -> list[Path]:
    imported: list[Path] = []
    for sidecar_path in pending_sidecars(staging_dir):
        imported.extend(
            auto_import_capture(sidecar_path, staging_dir, class_names, dataset_root, min_conf, max_conf, review_dir)
        )
    return imported
```

Update the existing Task 8 tests that call `auto_import_capture`/`auto_import_staging_dir` positionally — they're unaffected since `review_dir` is a new keyword-only-by-convention trailing parameter with a default, so all Task 8 tests keep passing unmodified.

Add to `src/pipeline/al_scheduler.py`'s `run_scheduler_tick`, passing `review_dir=artifacts_dir / "al_review"` into the `auto_import_staging_dir(...)` call.

Add to `.env.example`:

```
# Fraction of auto-import decisions written to artifacts/al_review/ for optional, non-blocking human spot-checking.
SMARTCART_AL_AUDIT_SAMPLE_RATE=0.05
```

Run: `uv run pytest -v`
Expected: PASS (all tests)

```bash
git add src/data/auto_labeler.py src/pipeline/al_scheduler.py tests/data/test_auto_labeler.py .env.example
git commit -m "feat: add passive sampled audit trail for auto-import decisions"
```
