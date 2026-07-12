import json
from pathlib import Path

from src.pipeline.candidate_promotion import (
    get_promoted_state_path,
    read_promoted_state,
    write_candidate_report,
    write_promoted_state,
)


def test_get_promoted_state_path_default_and_override(monkeypatch):
    monkeypatch.delenv("SMARTCART_AL_STATE_PATH", raising=False)
    assert get_promoted_state_path() == Path("./artifacts/al_pipeline_state.json")

    monkeypatch.setenv("SMARTCART_AL_STATE_PATH", "/tmp/custom_state.json")
    assert get_promoted_state_path() == Path("/tmp/custom_state.json")


def test_read_promoted_state_returns_defaults_when_missing(tmp_path):
    state = read_promoted_state(tmp_path / "missing.json")

    assert state == {
        "run_name": None,
        "map50": 0.0,
        "variant_accuracy": 0.0,
        "pending_auto_imported": 0,
    }


def test_write_then_read_promoted_state_round_trips(tmp_path):
    state_path = tmp_path / "state.json"
    write_promoted_state(state_path, {"run_name": "retrain_1", "map50": 0.8, "variant_accuracy": 0.9, "pending_auto_imported": 3})

    result = read_promoted_state(state_path)

    assert result == {"run_name": "retrain_1", "map50": 0.8, "variant_accuracy": 0.9, "pending_auto_imported": 3}


def test_write_candidate_report_writes_json_to_candidate_dir(tmp_path):
    candidate_dir = tmp_path / "retrain_1"
    candidate_dir.mkdir()
    report = {"run_name": "retrain_1", "map50": 0.8, "passed": True}

    path = write_candidate_report(candidate_dir, report)

    assert path == candidate_dir / "report.json"
    assert json.loads(path.read_text()) == report


def test_write_candidate_report_creates_candidate_dir_when_missing(tmp_path):
    candidate_dir = tmp_path / "retrain_2"
    report = {"run_name": "retrain_2", "map50": 0.75, "passed": False}

    path = write_candidate_report(candidate_dir, report)

    assert candidate_dir.is_dir()
    assert path == candidate_dir / "report.json"
    assert json.loads(path.read_text()) == report
