import base64

from PIL import Image

from src.models.vlm_verifier import (
    ask_vlm,
    build_chat_payload,
    get_vlm_base_url,
    get_vlm_model,
    image_to_data_url,
)


def test_get_vlm_base_url_default_and_override(monkeypatch):
    monkeypatch.delenv("SMARTCART_VLM_BASE_URL", raising=False)
    assert get_vlm_base_url() == "http://localhost:1234"

    monkeypatch.setenv("SMARTCART_VLM_BASE_URL", "http://example.com:9999/")
    assert get_vlm_base_url() == "http://example.com:9999"


def test_get_vlm_model_default_and_override(monkeypatch):
    monkeypatch.delenv("SMARTCART_VLM_MODEL", raising=False)
    assert get_vlm_model() == "qwen2.5-vl-3b-instruct"

    monkeypatch.setenv("SMARTCART_VLM_MODEL", "other-model")
    assert get_vlm_model() == "other-model"


def test_image_to_data_url_round_trips_as_jpeg():
    image = Image.new("RGB", (4, 4), color=(255, 0, 0))

    data_url = image_to_data_url(image)

    assert data_url.startswith("data:image/jpeg;base64,")
    decoded = base64.b64decode(data_url.split(",", 1)[1])
    assert decoded[:2] == b"\xff\xd8"  # JPEG magic bytes


def test_build_chat_payload_matches_lm_studio_schema():
    payload = build_chat_payload("qwen2.5-vl-3b-instruct", "What is this?", "data:image/jpeg;base64,ABC")

    assert payload == {
        "model": "qwen2.5-vl-3b-instruct",
        "input": [
            {"type": "text", "content": "What is this?"},
            {"type": "image", "data_url": "data:image/jpeg;base64,ABC"},
        ],
    }


def test_ask_vlm_posts_to_chat_endpoint_and_returns_content(monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        response = type("R", (), {})()
        response.raise_for_status = lambda: None
        response.json = lambda: {"output": [{"type": "message", "content": "Fruit/Apple"}]}
        return response

    monkeypatch.setenv("SMARTCART_VLM_BASE_URL", "http://localhost:1234")
    monkeypatch.setattr("src.models.vlm_verifier.httpx.post", fake_post)

    answer = ask_vlm(Image.new("RGB", (4, 4)), "Which category?")

    assert answer == "Fruit/Apple"
    assert captured["url"] == "http://localhost:1234/api/v1/chat"
    assert captured["json"]["input"][0] == {"type": "text", "content": "Which category?"}


def test_ask_vlm_returns_none_on_any_error(monkeypatch):
    def fake_post(url, json=None, timeout=None):
        raise ConnectionError("LM Studio not running")

    monkeypatch.setattr("src.models.vlm_verifier.httpx.post", fake_post)

    assert ask_vlm(Image.new("RGB", (4, 4)), "Which category?") is None


def test_ask_vlm_returns_none_on_malformed_response(monkeypatch):
    def fake_post(url, json=None, timeout=None):
        response = type("R", (), {})()
        response.raise_for_status = lambda: None
        response.json = lambda: {"unexpected": "shape"}
        return response

    monkeypatch.setattr("src.models.vlm_verifier.httpx.post", fake_post)

    assert ask_vlm(Image.new("RGB", (4, 4)), "Which category?") is None


def test_ask_vlm_returns_none_when_image_encoding_raises(monkeypatch):
    def fake_image_to_data_url(image):
        raise OSError("corrupt or truncated image")

    def fake_post(url, json=None, timeout=None):
        raise AssertionError("httpx.post should not be called when image encoding fails")

    monkeypatch.setattr("src.models.vlm_verifier.image_to_data_url", fake_image_to_data_url)
    monkeypatch.setattr("src.models.vlm_verifier.httpx.post", fake_post)

    assert ask_vlm(Image.new("RGB", (4, 4)), "Which category?") is None


from src.models.vlm_verifier import is_sane_box, match_category, parse_grounding_box


def test_match_category_finds_substring_match_case_insensitive():
    categories = ["Fruit/Apple/Royal-Gala", "Snacks/Chocolate-Bar/Cadbury-RoastAlmond"]

    assert match_category("i see a royal-gala apple", categories) == "Fruit/Apple/Royal-Gala"


def test_match_category_prefers_longest_match():
    categories = [
        "Ready-To-Eat/Instant-Noodles/Myojo/Chicken",
        "Ready-To-Eat/Instant-Noodles/Myojo/ChickenAbalone",
    ]

    answer = "This is Ready-To-Eat/Instant-Noodles/Myojo/ChickenAbalone flavor"

    assert match_category(answer, categories) == "Ready-To-Eat/Instant-Noodles/Myojo/ChickenAbalone"


def test_match_category_returns_none_when_no_match():
    assert match_category("I have no idea what this is", ["Fruit/Apple/Royal-Gala"]) is None


def test_is_sane_box_rejects_out_of_bounds():
    assert is_sane_box((10, 10, 200, 50), img_width=100, img_height=100) is False


def test_is_sane_box_rejects_too_small_or_too_large():
    assert is_sane_box((0, 0, 1, 1), img_width=100, img_height=100) is False  # ~0.01% of frame
    assert is_sane_box((0, 0, 100, 100), img_width=100, img_height=100) is False  # 100% of frame


def test_is_sane_box_accepts_plausible_box():
    assert is_sane_box((10, 10, 60, 60), img_width=100, img_height=100) is True


def test_parse_grounding_box_converts_normalized_coordinates():
    answer = "The item is located at (100,100),(500,500)."

    box = parse_grounding_box(answer, img_width=1000, img_height=1000)

    assert box == (100, 100, 500, 500)


def test_parse_grounding_box_returns_none_when_missing():
    assert parse_grounding_box("no box here", img_width=1000, img_height=1000) is None


def test_parse_grounding_box_returns_none_when_box_fails_sanity_check():
    # (0,0),(1000,1000) on a 0-1000 scale covers the entire frame -- not plausible
    answer = "(0,0),(1000,1000)"

    assert parse_grounding_box(answer, img_width=1000, img_height=1000) is None
