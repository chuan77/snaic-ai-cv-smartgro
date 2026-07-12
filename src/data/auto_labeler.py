"""Auto-imports staged active-learning captures using a local VLM as an
independent second opinion, in place of Label Studio human review. Every
staged capture is either imported into the dataset tree or discarded --
never left pending, since there is no human to eventually review it."""
import os
from pathlib import Path

from PIL import Image

from src.data.annotation_import import crop_with_padding
from src.models.vlm_verifier import ask_vlm, match_category


def get_autolabel_min_conf() -> float:
    return float(os.environ.get("SMARTCART_AUTOLABEL_MIN_CONF", "0.35"))


def pending_sidecars(staging_dir: Path) -> list[Path]:
    return sorted(p for p in staging_dir.glob("*.json") if p.parent == staging_dir)


def mark_consumed(sidecar_path: Path) -> None:
    consumed_dir = sidecar_path.parent / "consumed"
    consumed_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path.rename(consumed_dir / sidecar_path.name)


def classify_capture(detections: list[dict], min_conf: float, max_conf: float) -> str:
    """'zero_or_low' (no detections, or any detection below min_conf -- YOLO's own
    signal isn't trustworthy enough to cross-check, ask the VLM cold) vs 'mid_band'
    (every detection sits in [min_conf, max_conf) -- worth cross-checking each one
    individually against the VLM)."""
    if not detections or any(d["confidence"] < min_conf for d in detections):
        return "zero_or_low"
    return "mid_band"


_MID_BAND_PROMPT_TEMPLATE = (
    "You are checking a product detector's guess. Which of these exact product "
    "categories does the boxed item show: {categories}? Answer with the exact "
    "category name only, or 'none' if you are not confident."
)


def auto_import_mid_band_detection(
    image: Image.Image,
    detection: dict,
    class_names: list[str],
    dataset_root: Path,
    capture_id: str,
) -> Path | None:
    """Cross-checks one mid-confidence YOLO detection against the VLM; imports
    the crop into dataset_root/<class_name>/ only if both agree."""
    crop = crop_with_padding(image, tuple(detection["bbox"]))
    prompt = _MID_BAND_PROMPT_TEMPLATE.format(categories=", ".join(class_names))
    answer = ask_vlm(crop, prompt)
    if answer is None:
        return None

    vlm_category = match_category(answer, class_names)
    if vlm_category is None or vlm_category != detection["class_name"]:
        return None

    dest_dir = dataset_root / vlm_category
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{capture_id}.jpg"
    crop.save(dest_path)
    return dest_path
