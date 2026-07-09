import io
import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.data.annotation_import import classify_category
from src.data.gallery import GroceryDatasetIndexer
from src.data.label_studio_export_pull import (
    build_category_keywords,
    export_project,
    extract_export,
    pull_and_import_from_label_studio,
)


def _make_zip_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


def test_export_project_gets_yolo_endpoint_with_auth(monkeypatch):
    monkeypatch.setenv("LABEL_STUDIO_URL", "http://ls-host:8080")
    monkeypatch.setenv("LABEL_STUDIO_API_KEY", "secret-refresh-token")
    captured = {}
    zip_bytes = _make_zip_bytes({"classes.txt": b"a\nb\n"})

    def fake_token_post(url, json=None, timeout=None):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {"access": "fresh-access-token"}
        return response

    monkeypatch.setattr("src.deploy.label_studio_backend.httpx.post", fake_token_post)

    def fake_get(url, params=None, headers=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        captured["headers"] = headers
        response = MagicMock()
        response.status_code = 200
        response.content = zip_bytes
        response.raise_for_status = MagicMock()
        return response

    monkeypatch.setattr("src.data.label_studio_export_pull.httpx.get", fake_get)

    result = export_project("7")

    assert captured["url"] == "http://ls-host:8080/api/projects/7/export"
    assert captured["params"] == {"exportType": "YOLO"}
    assert captured["headers"] == {"Authorization": "Bearer fresh-access-token"}
    assert result == zip_bytes


def test_export_project_no_auth_header_when_key_unset(monkeypatch):
    monkeypatch.setenv("LABEL_STUDIO_API_KEY", "")
    captured = {}
    zip_bytes = _make_zip_bytes({"classes.txt": b"a\n"})

    def fake_get(url, params=None, headers=None, timeout=None):
        captured["headers"] = headers
        response = MagicMock()
        response.status_code = 200
        response.content = zip_bytes
        response.raise_for_status = MagicMock()
        return response

    monkeypatch.setattr("src.data.label_studio_export_pull.httpx.get", fake_get)

    export_project("7")

    assert captured["headers"] == {}


def test_export_project_raises_on_non_2xx(monkeypatch):
    import httpx

    monkeypatch.setenv("LABEL_STUDIO_API_KEY", "")

    def fake_get(url, params=None, headers=None, timeout=None):
        response = MagicMock()
        response.status_code = 403

        def raise_for_status():
            raise httpx.HTTPStatusError("forbidden", request=MagicMock(), response=response)

        response.raise_for_status = raise_for_status
        return response

    monkeypatch.setattr("src.data.label_studio_export_pull.httpx.get", fake_get)

    with pytest.raises(Exception):
        export_project("7")


def test_export_project_raises_on_non_zip_body(monkeypatch):
    monkeypatch.setenv("LABEL_STUDIO_API_KEY", "")

    def fake_get(url, params=None, headers=None, timeout=None):
        response = MagicMock()
        response.status_code = 200
        response.content = b'{"error": "project not found"}'
        response.raise_for_status = MagicMock()
        return response

    monkeypatch.setattr("src.data.label_studio_export_pull.httpx.get", fake_get)

    with pytest.raises(RuntimeError, match="not a valid zip"):
        export_project("999")


def test_extract_export_writes_tree(tmp_path):
    zip_bytes = _make_zip_bytes({"classes.txt": b"Fruit/Apple/Royal-Gala\n", "images/a.jpg": b"fake"})
    dest = tmp_path / "export"

    result = extract_export(zip_bytes, dest)

    assert result == dest
    assert (dest / "classes.txt").read_text() == "Fruit/Apple/Royal-Gala\n"
    assert (dest / "images" / "a.jpg").exists()


def test_extract_export_clears_stale_dir(tmp_path):
    dest = tmp_path / "export"
    dest.mkdir()
    (dest / "stale.txt").write_text("old data")
    zip_bytes = _make_zip_bytes({"classes.txt": b"fresh\n"})

    extract_export(zip_bytes, dest)

    assert not (dest / "stale.txt").exists()
    assert (dest / "classes.txt").read_text() == "fresh\n"


def test_build_category_keywords_maps_full_paths_to_dataset_dirs():
    class_map = {"Fruit/Apple/Royal-Gala": 0, "Vegetables/Carrots": 1}
    dataset_root = Path("/data/train")

    keywords = build_category_keywords(class_map, dataset_root)

    assert keywords == {
        "Fruit/Apple/Royal-Gala": dataset_root / "Fruit/Apple/Royal-Gala",
        "Vegetables/Carrots": dataset_root / "Vegetables/Carrots",
    }


def test_build_category_keywords_every_real_class_routes_to_itself():
    """Some real class names are substrings of others (e.g. 'Myojo/Chicken' vs.
    'Myojo/ChickenAbalone') now that fine-grained variants exist. classify_category
    resolves this by preferring the longest (most specific) match, so exact-match
    routing stays correct even though the keyword set itself isn't substring-free."""
    dataset_root = Path("./dataset/GroceryStoreDataset/dataset/train")
    if not dataset_root.exists():
        pytest.skip("GroceryStoreDataset not cloned locally")
    class_map = GroceryDatasetIndexer(dataset_root).build_class_map()
    category_keywords = build_category_keywords(class_map, dataset_root)

    for class_name in class_map:
        assert classify_category(class_name, category_keywords) == (class_name, dataset_root / class_name)


def test_pull_and_import_orchestrates(monkeypatch, tmp_path):
    calls = {}

    def fake_export_project(project_id):
        calls["export_project"] = project_id
        return b"zipbytes"

    def fake_extract_export(zip_bytes, dest_dir):
        calls["extract_export"] = (zip_bytes, dest_dir)
        return dest_dir

    monkeypatch.setattr("src.data.label_studio_export_pull.export_project", fake_export_project)
    monkeypatch.setattr("src.data.label_studio_export_pull.extract_export", fake_extract_export)

    written = [Path("/dataset/Fruit/Apple/Royal-Gala/x_0.jpg")]

    def fake_import(export_dir, category_keywords):
        calls["import"] = (export_dir, category_keywords)
        return written

    monkeypatch.setattr("src.data.label_studio_export_pull.import_label_studio_export", fake_import)

    class_map = {"Fruit/Apple/Royal-Gala": 0}
    dataset_root = Path("/dataset")
    staging_dir = tmp_path / "staging"

    result = pull_and_import_from_label_studio("7", class_map, dataset_root, staging_dir)

    assert result == written
    assert calls["export_project"] == "7"
    assert calls["extract_export"] == (b"zipbytes", staging_dir)
    assert calls["import"][0] == staging_dir
    assert calls["import"][1] == {"Fruit/Apple/Royal-Gala": dataset_root / "Fruit/Apple/Royal-Gala"}


def test_pull_and_import_empty_export_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setattr("src.data.label_studio_export_pull.export_project", lambda project_id: b"zipbytes")
    monkeypatch.setattr("src.data.label_studio_export_pull.extract_export", lambda zip_bytes, dest_dir: dest_dir)
    monkeypatch.setattr("src.data.label_studio_export_pull.import_label_studio_export", lambda *a, **k: [])

    result = pull_and_import_from_label_studio("7", {}, Path("/dataset"), tmp_path / "staging")

    assert result == []
