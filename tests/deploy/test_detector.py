from pathlib import Path
from unittest.mock import patch

from src.deploy.detector import get_gallery_index_path, get_gallery_meta_path, get_variant_resolver


def test_get_gallery_index_path_defaults_to_artifacts_gallery_index_npy(monkeypatch):
    monkeypatch.delenv("SMARTCART_GALLERY_INDEX_PATH", raising=False)

    assert get_gallery_index_path() == Path("./artifacts/gallery_index.npy")


def test_get_gallery_index_path_reads_from_environment(monkeypatch, tmp_path):
    custom_path = tmp_path / "custom_index.npy"
    monkeypatch.setenv("SMARTCART_GALLERY_INDEX_PATH", str(custom_path))

    assert get_gallery_index_path() == custom_path


def test_get_gallery_meta_path_defaults_to_artifacts_gallery_meta_csv(monkeypatch):
    monkeypatch.delenv("SMARTCART_GALLERY_META_PATH", raising=False)

    assert get_gallery_meta_path() == Path("./artifacts/gallery_meta.csv")


def test_get_gallery_meta_path_reads_from_environment(monkeypatch, tmp_path):
    custom_path = tmp_path / "custom_meta.csv"
    monkeypatch.setenv("SMARTCART_GALLERY_META_PATH", str(custom_path))

    assert get_gallery_meta_path() == custom_path


def test_get_variant_resolver_constructs_with_configured_gallery_paths(monkeypatch, tmp_path):
    index_path = tmp_path / "gallery_index.npy"
    meta_path = tmp_path / "gallery_meta.csv"
    monkeypatch.setenv("SMARTCART_GALLERY_INDEX_PATH", str(index_path))
    monkeypatch.setenv("SMARTCART_GALLERY_META_PATH", str(meta_path))
    get_variant_resolver.cache_clear()

    with patch("src.deploy.detector.VariantResolver") as mock_resolver_cls:
        get_variant_resolver()

    mock_resolver_cls.assert_called_once_with(index_path, meta_path)
    get_variant_resolver.cache_clear()
