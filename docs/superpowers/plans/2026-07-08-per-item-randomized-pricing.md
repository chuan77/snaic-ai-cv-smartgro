# Per-Item Randomized Pricing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the flat, per-category catalog price with a per-item price that jitters ±15% around the category base and snaps to a charm-pricing (`.49`/`.99`) ending, deterministically seeded per product.

**Architecture:** A single new pure function, `get_item_price(class_str: str) -> float`, added to `src/data/gallery.py` next to the existing `get_base_price`. It's wired into the one live call site that builds `catalog_prices.csv` (`UnifiedProductGallery.compile_gallery`). No other files change.

**Tech Stack:** Python stdlib only (`hashlib`, `random`) — no new dependencies. Tests via `pytest` (already configured in `pyproject.toml`, run with `uv run pytest`).

## Global Constraints

- Jitter range: ±15% of the category base price (from `get_base_price`).
- Determinism: seed the RNG from `int(hashlib.sha256(class_str.encode()).hexdigest(), 16)` — never Python's built-in `hash()` (randomized per-process for strings).
- Rounding: round the jittered price to the nearest $0.50, then subtract $0.01, producing a `.49` or `.99` ending.
- `get_base_price` itself is unchanged.
- Only the live path in `src/data/gallery.py` changes (`UnifiedProductGallery.compile_gallery`); the dead `HardenedProductCatalog`/`DeterministicGalleryBuilder` path is out of scope.
- Spec: `docs/superpowers/specs/2026-07-08-per-item-randomized-pricing-design.md`

---

### Task 1: Add `get_item_price` and wire it into the catalog build

**Files:**
- Modify: `src/data/gallery.py:1-16` (imports), `src/data/gallery.py:23-31` (add new function after `get_base_price`), `src/data/gallery.py:61-63` (call site)
- Test: `tests/data/test_gallery.py`

**Interfaces:**
- Produces: `get_item_price(class_str: str) -> float` in `src/data/gallery.py`, importable as `from src.data.gallery import get_item_price`.
- Consumes: existing `get_base_price(class_str: str) -> float` (unchanged, same file).

- [ ] **Step 1: Write the failing tests**

Append to `tests/data/test_gallery.py`:

```python
from src.data.gallery import get_base_price, get_item_price


def test_get_item_price_varies_within_same_category():
    assert get_item_price("Snacks/Chocolate-Bar") == 5.99
    assert get_item_price("Snacks/Cookies") == 6.49


def test_get_item_price_is_deterministic_across_calls():
    first = get_item_price("Packages/Milk/Arla-Standard-Milk")
    second = get_item_price("Packages/Milk/Arla-Standard-Milk")
    assert first == second == 3.49


@pytest.mark.parametrize("class_str,expected_price", [
    ("Snacks/Chocolate-Bar", 5.99),
    ("Snacks/Cookies", 6.49),
    ("Packages/Milk/Arla-Standard-Milk", 3.49),
    ("Ready-To-Eat/Instant-Noodles", 3.49),
    ("Fruit/Apple/Royal-Gala", 1.99),
    ("Vegetables/Carrots", 1.99),
])
def test_get_item_price_ends_in_charm_pricing(class_str, expected_price):
    price = get_item_price(class_str)
    assert price == expected_price
    cents = round((price % 1) * 100)
    assert cents in (49, 99)


def test_get_item_price_stays_near_category_base():
    for class_str in ("Snacks/Chocolate-Bar", "Snacks/Cookies"):
        base = get_base_price(class_str)
        price = get_item_price(class_str)
        assert base * 0.75 <= price <= base * 1.25
```

The full file should now read:

```python
import pytest

from src.data.gallery import get_base_price, get_item_price


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


def test_get_item_price_varies_within_same_category():
    assert get_item_price("Snacks/Chocolate-Bar") == 5.99
    assert get_item_price("Snacks/Cookies") == 6.49


def test_get_item_price_is_deterministic_across_calls():
    first = get_item_price("Packages/Milk/Arla-Standard-Milk")
    second = get_item_price("Packages/Milk/Arla-Standard-Milk")
    assert first == second == 3.49


@pytest.mark.parametrize("class_str,expected_price", [
    ("Snacks/Chocolate-Bar", 5.99),
    ("Snacks/Cookies", 6.49),
    ("Packages/Milk/Arla-Standard-Milk", 3.49),
    ("Ready-To-Eat/Instant-Noodles", 3.49),
    ("Fruit/Apple/Royal-Gala", 1.99),
    ("Vegetables/Carrots", 1.99),
])
def test_get_item_price_ends_in_charm_pricing(class_str, expected_price):
    price = get_item_price(class_str)
    assert price == expected_price
    cents = round((price % 1) * 100)
    assert cents in (49, 99)


def test_get_item_price_stays_near_category_base():
    for class_str in ("Snacks/Chocolate-Bar", "Snacks/Cookies"):
        base = get_base_price(class_str)
        price = get_item_price(class_str)
        assert base * 0.75 <= price <= base * 1.25
```

The expected values above (`5.99`, `6.49`, `3.49`, `1.99`) were computed by hand-running the exact algorithm from the spec — they are not placeholders, they pin the implementation to a specific, verifiable output.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/data/test_gallery.py -v`
Expected: 4 new tests FAIL with `ImportError: cannot import name 'get_item_price' from 'src.data.gallery'` (the pre-existing `test_get_base_price_maps_top_level_category_to_expected_price` tests also fail as a side effect of the import error at module load — this is expected and resolves once the import succeeds).

- [ ] **Step 3: Implement `get_item_price`**

In `src/data/gallery.py`, add to the imports at the top of the file (after the existing `import os` / `import logging` block, alongside `from pathlib import Path`):

```python
import hashlib
import random
```

Then add the new function immediately after `get_base_price` (after line 31):

```python
def get_item_price(class_str: str) -> float:
    """Derives a stable, per-item charm price by jittering the category base ±15%."""
    base = get_base_price(class_str)
    seed = int(hashlib.sha256(class_str.encode()).hexdigest(), 16)
    jittered = base * random.Random(seed).uniform(0.85, 1.15)
    return round(jittered * 2) / 2 - 0.01
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/data/test_gallery.py -v`
Expected: all tests PASS (the original `test_get_base_price_maps_top_level_category_to_expected_price` parametrized cases plus the 4 new tests).

- [ ] **Step 5: Wire `get_item_price` into the catalog build call site**

In `src/data/gallery.py`, inside `UnifiedProductGallery.compile_gallery` (currently lines 61-63):

```python
# before
# Determine structured base pricing depending on category rules
base_price = get_base_price(class_str)
catalog_records.append({"class_id": class_id, "product_name": class_str, "price_usd": base_price})
```

```python
# after
# Determine per-item price with randomized jitter around the category base
item_price = get_item_price(class_str)
catalog_records.append({"class_id": class_id, "product_name": class_str, "price_usd": item_price})
```

- [ ] **Step 6: Run the full test suite**

Run: `uv run pytest -v`
Expected: all tests pass (no `slow`-marked tests run by default per `addopts = "-m 'not slow'"` in `pyproject.toml`). Confirm no other test references `base_price` or the old call-site variable name.

- [ ] **Step 7: Commit**

```bash
git add src/data/gallery.py tests/data/test_gallery.py
git commit -m "feat: randomize per-item catalog prices around category base"
```
