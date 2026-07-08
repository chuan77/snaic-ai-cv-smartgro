import pytest


@pytest.fixture(autouse=True)
def isolate_active_learning_capture_dir(monkeypatch, tmp_path):
    """Prevents /predict's uncertainty-capture hook from writing into the real
    project artifacts/ directory during tests."""
    monkeypatch.setenv("SMARTCART_CAPTURE_DIR", str(tmp_path / "active_learning_staging"))
