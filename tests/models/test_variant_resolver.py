import numpy as np
import pandas as pd
from PIL import Image

from src.models.variant_resolver import VariantResolver


class StubExtractor:
    def __init__(self, vector):
        self.vector = vector

    def extract(self, pil_image):
        return self.vector


def _write_gallery(tmp_path, rows, vectors):
    index_path = tmp_path / "gallery_index.npy"
    meta_path = tmp_path / "gallery_meta.csv"
    np.save(index_path, np.array(vectors, dtype=np.float32))
    pd.DataFrame(rows).to_csv(meta_path, index=False)
    return index_path, meta_path


def test_resolve_returns_coarse_class_name_when_no_gallery_candidates(tmp_path):
    index_path, meta_path = _write_gallery(
        tmp_path,
        rows=[{"class_id": 0, "product_name": "Fruit/Apple/Royal-Gala", "file_name": "a.jpg"}],
        vectors=[[1.0, 0.0]],
    )
    resolver = VariantResolver(index_path, meta_path, extractor=StubExtractor(np.array([1.0, 0.0])))

    result = resolver.resolve("Snacks/Chocolate-Bar", Image.new("RGB", (10, 10)))

    assert result == "Snacks/Chocolate-Bar"


def test_resolve_returns_product_name_of_nearest_embedding(tmp_path):
    index_path, meta_path = _write_gallery(
        tmp_path,
        rows=[
            {"class_id": 0, "product_name": "Snacks/Chocolate-Bar/Cadbury-RoastAlmond", "file_name": "a.jpg"},
            {"class_id": 1, "product_name": "Snacks/Chocolate-Bar/Lindt-SeaSalt", "file_name": "b.jpg"},
        ],
        vectors=[[1.0, 0.0], [0.0, 1.0]],
    )
    resolver = VariantResolver(index_path, meta_path, extractor=StubExtractor(np.array([0.9, 0.1])))

    result = resolver.resolve("Snacks/Chocolate-Bar", Image.new("RGB", (10, 10)))

    assert result == "Snacks/Chocolate-Bar/Cadbury-RoastAlmond"


def test_resolve_ignores_candidates_outside_the_coarse_prefix(tmp_path):
    index_path, meta_path = _write_gallery(
        tmp_path,
        rows=[
            {"class_id": 0, "product_name": "Fruit/Apple/Royal-Gala", "file_name": "a.jpg"},
            {"class_id": 1, "product_name": "Snacks/Chocolate-Bar/Lindt-SeaSalt", "file_name": "b.jpg"},
        ],
        vectors=[[1.0, 0.0], [0.0, 1.0]],
    )
    resolver = VariantResolver(index_path, meta_path, extractor=StubExtractor(np.array([1.0, 0.0])))

    result = resolver.resolve("Snacks/Chocolate-Bar", Image.new("RGB", (10, 10)))

    assert result == "Snacks/Chocolate-Bar/Lindt-SeaSalt"
