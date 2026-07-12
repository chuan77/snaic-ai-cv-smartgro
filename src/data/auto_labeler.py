"""Auto-imports staged active-learning captures using a local VLM as an
independent second opinion, in place of Label Studio human review. Every
staged capture is either imported into the dataset tree or discarded --
never left pending, since there is no human to eventually review it."""
import json
import os
from pathlib import Path

from PIL import Image

from src.data.annotation_import import crop_with_padding
from src.models.vlm_verifier import ask_vlm, match_category, parse_grounding_box


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


_LOW_SIGNAL_PROMPT_TEMPLATE = (
    "Which of these exact product categories appears in this image: {categories}? "
    "If you can, also give its bounding box as (x1,y1),(x2,y2) on a 0-1000 scale. "
    "Answer 'none' if you are not confident."
)


def auto_import_low_signal_frame(
    image: Image.Image, class_names: list[str], dataset_root: Path, capture_id: str
) -> Path | None:
    """Asks the VLM cold (no YOLO signal to cross-check) for a category and an
    optional box over the whole frame. Falls back to the whole frame as the crop
    when no valid box comes back -- reasonable here since the app's own capture
    flow is already one held-up item per frame, not a wild guess."""
    prompt = _LOW_SIGNAL_PROMPT_TEMPLATE.format(categories=", ".join(class_names))
    answer = ask_vlm(image, prompt)
    if answer is None:
        return None

    vlm_category = match_category(answer, class_names)
    if vlm_category is None:
        return None

    box = parse_grounding_box(answer, image.width, image.height)
    crop = image.crop(box) if box is not None else image

    dest_dir = dataset_root / vlm_category
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{capture_id}.jpg"
    crop.save(dest_path)
    return dest_path


def load_capture(sidecar_path: Path) -> dict:
    return json.loads(sidecar_path.read_text())


def auto_import_capture(
    sidecar_path: Path,
    staging_dir: Path,
    class_names: list[str],
    dataset_root: Path,
    min_conf: float,
    max_conf: float,
) -> list[Path]:
    """Processes one staged capture end-to-end and always marks it consumed
    afterward (even on error), so the scheduler never reprocesses it and one
    corrupt capture never blocks the rest of the staging directory."""
    imported: list[Path] = []
    try:
        capture = load_capture(sidecar_path)
        image_path = staging_dir / capture["image_file"]
        with Image.open(image_path).convert("RGB") as image:
            detections = capture["detections"]
            capture_id = sidecar_path.stem
            if classify_capture(detections, min_conf, max_conf) == "zero_or_low":
                result = auto_import_low_signal_frame(image, class_names, dataset_root, capture_id)
                if result is not None:
                    imported.append(result)
            else:
                for i, detection in enumerate(detections):
                    result = auto_import_mid_band_detection(
                        image, detection, class_names, dataset_root, f"{capture_id}_{i}"
                    )
                    if result is not None:
                        imported.append(result)
    except Exception:
        pass
    finally:
        mark_consumed(sidecar_path)
    return imported


def auto_import_staging_dir(
    staging_dir: Path, class_names: list[str], dataset_root: Path, min_conf: float, max_conf: float
) -> list[Path]:
    imported: list[Path] = []
    for sidecar_path in pending_sidecars(staging_dir):
        imported.extend(auto_import_capture(sidecar_path, staging_dir, class_names, dataset_root, min_conf, max_conf))
    return imported
