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
