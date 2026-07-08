"""Pulls corrected annotations back from a Label Studio project for retraining."""
import io
import shutil
import zipfile
from pathlib import Path

import httpx

from src.data.annotation_import import import_label_studio_export
from src.deploy.label_studio_backend import get_label_studio_api_key, get_label_studio_url


def export_project(project_id: str) -> bytes:
    api_key = get_label_studio_api_key()
    headers = {"Authorization": f"Token {api_key}"} if api_key else {}

    response = httpx.get(
        f"{get_label_studio_url()}/api/projects/{project_id}/export",
        params={"exportType": "YOLO"},
        headers=headers,
        timeout=60.0,
    )
    response.raise_for_status()

    if not zipfile.is_zipfile(io.BytesIO(response.content)):
        raise RuntimeError(
            f"Label Studio export for project {project_id} is not a valid zip "
            "(likely a bad project id or auth failure)."
        )

    return response.content


def extract_export(zip_bytes: bytes, dest_dir: Path) -> Path:
    shutil.rmtree(dest_dir, ignore_errors=True)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        zf.extractall(dest_dir)
    return dest_dir


def build_category_keywords(class_map: dict[str, int], dataset_root: Path) -> dict[str, Path]:
    """Maps each known class's exact leaf path to its destination directory, for
    exact-match routing of Label Studio exports (avoids substring-collision risk
    across a large taxonomy — verified collision-free for the current class list)."""
    return {class_name: dataset_root / class_name for class_name in class_map}


def pull_and_import_from_label_studio(
    project_id: str, class_map: dict[str, int], dataset_root: Path, staging_dir: Path
) -> list[Path]:
    zip_bytes = export_project(project_id)
    export_dir = extract_export(zip_bytes, staging_dir)
    category_keywords = build_category_keywords(class_map, dataset_root)
    return import_label_studio_export(export_dir, category_keywords)
