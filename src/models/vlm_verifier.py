"""HTTP client for a local LM Studio server hosting Qwen2.5-VL, used as an
independent second opinion for auto-labeling active-learning captures. Every
failure mode (unreachable server, timeout, malformed response) returns None —
callers must treat None exactly like an unconfident/disagreeing answer, never
block or raise on it."""
import base64
import io
import os

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
    payload = build_chat_payload(get_vlm_model(), prompt, image_to_data_url(image))
    try:
        response = httpx.post(f"{get_vlm_base_url()}/api/v1/chat", json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()["output"][0]["content"]
    except Exception:
        return None
