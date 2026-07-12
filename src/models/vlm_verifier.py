"""HTTP client for a local LM Studio server hosting Qwen2.5-VL, used as an
independent second opinion for auto-labeling active-learning captures. Every
failure mode (unreachable server, timeout, malformed response) returns None —
callers must treat None exactly like an unconfident/disagreeing answer, never
block or raise on it."""
import base64
import io
import os
import re

import httpx
from PIL import Image


def get_vlm_base_url() -> str:
    return os.environ.get("SMARTCART_VLM_BASE_URL", "http://localhost:1234").rstrip("/")


def get_vlm_model() -> str:
    return os.environ.get("SMARTCART_VLM_MODEL", "qwen2.5-vl-3b-instruct")


def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def build_chat_payload(model: str, prompt: str, data_url: str) -> dict:
    return {
        "model": model,
        "input": [
            {"type": "text", "content": prompt},
            {"type": "image", "data_url": data_url},
        ],
    }


def ask_vlm(image: Image.Image, prompt: str, timeout: float = 30.0) -> str | None:
    try:
        payload = build_chat_payload(get_vlm_model(), prompt, image_to_data_url(image))
        response = httpx.post(f"{get_vlm_base_url()}/api/v1/chat", json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()["output"][0]["content"]
    except Exception:
        return None


_GROUNDING_BOX_PATTERN = re.compile(r"\((\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+)\)")


def match_category(answer: str, candidate_categories: list[str]) -> str | None:
    """Finds which candidate category the VLM's free-text answer references, via
    case-insensitive substring match, preferring the longest match when several
    candidates appear (mirrors annotation_import.classify_category's approach)."""
    answer_lower = answer.lower()
    matches = []
    for c in candidate_categories:
        if any(part.lower() in answer_lower for part in c.split("/")):
            matches.append(c)
    if not matches:
        return None
    return max(matches, key=len)


def is_sane_box(box: tuple[int, int, int, int], img_width: int, img_height: int) -> bool:
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= img_width and 0 <= y1 < y2 <= img_height):
        return False
    area_ratio = ((x2 - x1) * (y2 - y1)) / (img_width * img_height)
    return 0.01 <= area_ratio <= 0.9


def parse_grounding_box(answer: str, img_width: int, img_height: int) -> tuple[int, int, int, int] | None:
    """Parses a Qwen-style '(x1,y1),(x2,y2)' box on a 0-1000 normalized scale out
    of free text, scales it to pixel xyxy, and returns None if missing or if it
    fails a basic plausibility check (in-bounds, 1-90% of the frame area)."""
    match = _GROUNDING_BOX_PATTERN.search(answer)
    if match is None:
        return None
    x1, y1, x2, y2 = (int(v) for v in match.groups())
    box = (
        int(x1 / 1000 * img_width),
        int(y1 / 1000 * img_height),
        int(x2 / 1000 * img_width),
        int(y2 / 1000 * img_height),
    )
    return box if is_sane_box(box, img_width, img_height) else None
