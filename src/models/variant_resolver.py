"""Resolves a coarse YOLO detection to its specific catalog variant via DINOv2 nearest-neighbor
lookup against the Day-1 gallery, for categories where fine-grained embeddings exist."""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.models.dino_extractor import DinoFeatureExtractor


class VariantResolver:
    """Matches a cropped detection against gallery embeddings nested under its coarse class."""

    def __init__(
        self,
        gallery_index_path: Path,
        gallery_meta_path: Path,
        extractor: DinoFeatureExtractor | None = None,
    ):
        self.extractor = extractor if extractor is not None else DinoFeatureExtractor()
        self.embeddings = np.load(gallery_index_path)
        self.meta = pd.read_csv(gallery_meta_path)

    def resolve(self, coarse_class_name: str, crop: Image.Image) -> str:
        prefix = f"{coarse_class_name}/"
        candidates = self.meta[self.meta["product_name"].str.startswith(prefix)]
        if candidates.empty:
            return coarse_class_name

        query = np.asarray(self.extractor.extract(crop))
        candidate_indices = candidates.index.to_numpy()
        vectors = self.embeddings[candidate_indices]
        similarities = vectors @ query / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(query) + 1e-8)
        best_index = candidate_indices[int(np.argmax(similarities))]
        return self.meta.loc[best_index, "product_name"]
