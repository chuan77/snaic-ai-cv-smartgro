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
