import json
from unittest.mock import MagicMock

from src.deploy.label_studio_push import (
    build_image_url,
    build_import_task,
    mark_pushed,
    push_staging_dir,
    push_tasks,
)


def _write_capture(staging_dir, name, detections, width=64, height=32):
    staging_dir.mkdir(parents=True, exist_ok=True)
    (staging_dir / f"{name}.jpg").write_bytes(b"fake-jpg-bytes")
    sidecar = {
        "captured_at": "2026-07-08T00:00:00",
        "image_file": f"{name}.jpg",
        "image_width": width,
        "image_height": height,
        "num_detections": len(detections),
        "min_confidence": min((d["confidence"] for d in detections), default=None),
        "detections": detections,
    }
    (staging_dir / f"{name}.json").write_text(json.dumps(sidecar))
    return staging_dir / f"{name}.json"


def test_build_image_url_uses_public_base(monkeypatch):
    monkeypatch.setenv("SMARTCART_STAGING_PUBLIC_URL", "http://example.com/staging")

    assert build_image_url("abc123.jpg") == "http://example.com/staging/abc123.jpg"


def test_build_import_task_attaches_predictions(monkeypatch):
    monkeypatch.setenv("SMARTCART_LS_DATA_KEY", "image")
    detections = [{"class_name": "Vegetables/Carrots", "confidence": 0.3, "bbox": [0.0, 0.0, 32.0, 16.0]}]
    capture = {"image_width": 64, "image_height": 32, "detections": detections}

    task = build_import_task(
        capture, image_url="http://x/img.jpg", from_name="label", to_name="image", model_version="v1"
    )

    assert task["data"] == {"image": "http://x/img.jpg"}
    assert len(task["predictions"]) == 1
    prediction = task["predictions"][0]
    assert prediction["model_version"] == "v1"
    assert prediction["result"][0]["value"]["rectanglelabels"] == ["Vegetables/Carrots"]


def test_push_tasks_posts_to_import_endpoint_with_auth(monkeypatch):
    monkeypatch.setenv("LABEL_STUDIO_URL", "http://ls-host:8080")
    monkeypatch.setenv("LABEL_STUDIO_API_KEY", "secret-token")
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        response = MagicMock()
        response.status_code = 201
        return response

    monkeypatch.setattr("src.deploy.label_studio_push.httpx.post", fake_post)

    tasks = [{"data": {"image": "http://x/img.jpg"}, "predictions": []}]
    response = push_tasks(tasks, project_id="7")

    assert captured["url"] == "http://ls-host:8080/api/projects/7/import"
    assert captured["json"] == tasks
    assert captured["headers"] == {"Authorization": "Token secret-token"}
    assert response.status_code == 201


def test_mark_pushed_moves_files_to_pushed_subdir(tmp_path):
    sidecar_path = _write_capture(tmp_path, "abc123", [])
    image_path = tmp_path / "abc123.jpg"

    mark_pushed(sidecar_path, image_path)

    assert not sidecar_path.exists()
    assert not image_path.exists()
    assert (tmp_path / "pushed" / "abc123.json").exists()
    assert (tmp_path / "pushed" / "abc123.jpg").exists()


def test_push_staging_dir_skips_pushed_subdir_and_returns_count(tmp_path, monkeypatch):
    monkeypatch.setenv("SMARTCART_STAGING_PUBLIC_URL", "http://example.com/staging")
    _write_capture(tmp_path, "one", [{"class_name": "Vegetables/Carrots", "confidence": 0.2, "bbox": [0, 0, 1, 1]}])
    _write_capture(tmp_path, "two", [])
    already_pushed_dir = tmp_path / "pushed"
    already_pushed_dir.mkdir()
    _write_capture(already_pushed_dir, "already", [])

    monkeypatch.setattr(
        "src.deploy.label_studio_push.push_tasks",
        lambda tasks, project_id: MagicMock(status_code=201),
    )

    count = push_staging_dir(tmp_path, project_id="7")

    assert count == 2
    assert not (tmp_path / "one.json").exists()
    assert not (tmp_path / "two.json").exists()
    assert (tmp_path / "pushed" / "one.json").exists()
    assert (tmp_path / "pushed" / "two.json").exists()
    assert (tmp_path / "pushed" / "already.json").exists()
