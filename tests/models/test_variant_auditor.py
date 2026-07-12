import numpy as np
import pandas as pd

from src.models.variant_auditor import compute_variant_accuracy, split_holdout


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
    # Cadbury's 4 photos all sit in one direction; Lindt's first 3 reference photos
    # sit in an orthogonal direction, but Lindt's 4th photo (file name sorts last,
    # so split_holdout holds it out) is a near-duplicate of Cadbury's direction --
    # a genuinely confusable case that should misclassify against Cadbury.
    meta = pd.DataFrame([
        {"product_name": "Snacks/Chocolate-Bar/Cadbury", "file_name": f"cad{i}.jpg"} for i in range(4)
    ] + [
        {"product_name": "Snacks/Chocolate-Bar/Lindt", "file_name": f"lin{i}.jpg"} for i in range(4)
    ])
    embeddings = np.array(
        [[1.0, 0.0]] * 4          # all 4 Cadbury photos
        + [[0.0, 1.0]] * 3        # Lindt reference photos (lin0, lin1, lin2)
        + [[0.95, 0.05]],         # Lindt holdout photo (lin3) -- confusable with Cadbury
        dtype=np.float32,
    )

    accuracy, excluded = compute_variant_accuracy(embeddings, meta)

    assert accuracy < 1.0
    assert excluded == 0
