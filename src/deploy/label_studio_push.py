"""Pushes actively-captured, uncertain production frames into a Label Studio project
as pre-annotated review tasks."""
import json
import os
from pathlib import Path

import httpx

from src.deploy.label_studio_backend import (
    build_ls_prediction,
    get_label_studio_api_key,
    get_label_studio_url,
    get_ls_data_key,
    get_ls_from_name,
    get_ls_to_name,
    get_model_version,
)


def get_staging_public_url() -> str:
    return os.environ.get("SMARTCART_STAGING_PUBLIC_URL", "http://localhost:8000/staging").rstrip("/")


def get_ls_project_id() -> str:
    return os.environ.get("SMARTCART_LS_PROJECT_ID", "")


def load_capture(sidecar_path: Path) -> dict:
    return json.loads(sidecar_path.read_text())


def build_image_url(image_file: str) -> str:
    return f"{get_staging_public_url()}/{image_file}"


def build_import_task(capture: dict, image_url: str, from_name: str, to_name: str, model_version: str) -> dict:
    prediction = build_ls_prediction(
        capture["detections"], capture["image_width"], capture["image_height"], from_name, to_name, model_version
    )
    return {"data": {get_ls_data_key(): image_url}, "predictions": [prediction]}


def push_tasks(tasks: list[dict], project_id: str) -> httpx.Response:
    api_key = get_label_studio_api_key()
    headers = {"Authorization": f"Token {api_key}"} if api_key else {}
    return httpx.post(
        f"{get_label_studio_url()}/api/projects/{project_id}/import",
        json=tasks,
        headers=headers,
        timeout=30.0,
    )


def mark_pushed(sidecar_path: Path, image_path: Path) -> None:
    pushed_dir = sidecar_path.parent / "pushed"
    pushed_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path.rename(pushed_dir / sidecar_path.name)
    image_path.rename(pushed_dir / image_path.name)


def push_staging_dir(staging_dir: Path, project_id: str) -> int:
    from_name = get_ls_from_name()
    to_name = get_ls_to_name()
    model_version = get_model_version()

    sidecar_paths = sorted(p for p in staging_dir.glob("*.json") if p.parent == staging_dir)
    count = 0
    for sidecar_path in sidecar_paths:
        capture = load_capture(sidecar_path)
        image_path = staging_dir / capture["image_file"]
        image_url = build_image_url(capture["image_file"])
        task = build_import_task(capture, image_url, from_name, to_name, model_version)

        response = push_tasks([task], project_id)
        if 200 <= response.status_code < 300:
            mark_pushed(sidecar_path, image_path)
            count += 1

    return count
