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
