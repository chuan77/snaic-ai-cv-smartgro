"""Auto-imports staged active-learning captures using a local VLM as an
independent second opinion, in place of Label Studio human review. Every
staged capture is either imported into the dataset tree or discarded --
never left pending, since there is no human to eventually review it."""
import os
from pathlib import Path


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
