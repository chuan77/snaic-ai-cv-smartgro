# Per-item randomized pricing for the grocery catalog

## Context

`src/data/gallery.py`'s `get_base_price(class_str)` currently assigns price purely by top-level category (`Packages` → $3.20, `Ready-To-Eat` → $3.00, `Snacks` → $6.20, else $1.75). Every leaf product within a category gets the identical flat price — e.g. every snack, from a candy bar to a bag of chips, is priced at exactly $6.20. This is the live pricing path: `main_day1_catalog.py` → `UnifiedProductGallery.compile_gallery` ([gallery.py:51-82](../../../src/data/gallery.py#L51-L82)) calls it while writing `catalog_prices.csv`, which is later read as-is by both the Gradio checkout dashboard (`src/deploy/register.py`) and the FastAPI `/catalog` endpoint (`src/deploy/api_server.py`).

This spec adds per-item price variation within each category, so individual products look like a real catalog rather than a fixed price table.

## Algorithm

Add `get_item_price(class_str)` to `src/data/gallery.py`, alongside the existing `get_base_price`:

```python
import hashlib
import random

def get_item_price(class_str: str) -> float:
    """Derives a stable, per-item charm price by jittering the category base ±15%."""
    base = get_base_price(class_str)
    seed = int(hashlib.sha256(class_str.encode()).hexdigest(), 16)
    jittered = base * random.Random(seed).uniform(0.85, 1.15)
    price = round(jittered * 10) / 10 - 0.01   # nearest dime, then charm to an X.X9 ending
    return round(price, 2)                     # clear floating-point rounding artifacts (e.g. 2.5900000000000003)
```

- **Jitter**: ±15% around the category base price (e.g. `Snacks` base $6.20 → raw range ~$5.27–$7.13).
- **Determinism**: the RNG is seeded from a SHA-256 hash of `class_str` (the full dataset-relative class path, e.g. `"Fruit/Apple/Royal-Gala"`), not Python's built-in `hash()` (which is randomized per-process for strings via `PYTHONHASHSEED`). The same product always resolves to the same price across repeated Day-1 runs.
- **Charm pricing**: the jittered value is rounded to the nearest $0.10 increment, then $0.01 is subtracted, snapping every price to an `X.X9` ending (e.g. raw $5.27 → nearest dime $5.30 → charmed $5.29; raw $6.83 → nearest dime $6.80 → charmed $6.79). Rounding to the nearest dime (not half-dollar) was chosen after review found that a $0.50 grain collapses the $1.75 default tier (the majority of products, since it's the catch-all for every category except Packages/Ready-To-Eat/Snacks) to just two possible prices ($1.49/$1.99) regardless of jitter width — the grain, not the jitter range, was the bottleneck. A $0.10 grain gives the $1.75 tier 6 distinct prices and the $6.20 tier ~19, under simulation across 500 sample product names.

`get_base_price` itself is unchanged — it remains the source of truth for which price *tier* (category) an item belongs to.

## Call site change

In `UnifiedProductGallery.compile_gallery` ([gallery.py:62-63](../../../src/data/gallery.py#L62-L63)):

```python
# before
base_price = get_base_price(class_str)
catalog_records.append({"class_id": class_id, "product_name": class_str, "price_usd": base_price})

# after
item_price = get_item_price(class_str)
catalog_records.append({"class_id": class_id, "product_name": class_str, "price_usd": item_price})
```

The log line at [gallery.py:65](../../../src/data/gallery.py#L65) is updated to reference the new variable name for consistency.

## Scope

Only the live path in `src/data/gallery.py` changes. Downstream consumers need no changes:

- `src/deploy/register.py` (`AutonomousPOSRegister`) already builds `price_sheet` as a dict keyed by exact `product_name` from the CSV — per-item prices flow through automatically.
- `src/deploy/api_server.py`'s `/catalog` endpoint already serves whatever `price_usd` is in the CSV per row.

The dead, unused pricing path (`HardenedProductCatalog.register_product` / `DeterministicGalleryBuilder` in `src/data/gallery.py` / `src/data/catalog.py`, not called from any entrypoint) is left untouched — out of scope.

## Out of scope

- No changes to the category base prices themselves.
- No config/env var to tune the jitter range or disable randomization — a fixed ±15% is used, consistent with this codebase's existing pattern of inline tunable constants (e.g. `synthesizer.py`'s `random.uniform(0.20, 0.33)` for scale).
- No changes to `HardenedProductCatalog`/`DeterministicGalleryBuilder` (dead code path).

## Verification

No test suite exists in this repo, and exercising the full Day-1 pipeline requires the grocery dataset and a DINOv2 download. Verification is done by exercising `get_item_price` standalone (e.g. via a short Python snippet importing the function):

1. Calling it with several class strings from the same category produces different prices (per-item variation within a category).
2. Calling it twice with the same class string produces the identical price (determinism).
3. All returned prices end in an `X.X9` charm ending.
4. Prices stay in a sane range relative to their category base (no negative or wildly out-of-range values).
