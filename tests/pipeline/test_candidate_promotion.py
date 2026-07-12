import json
from pathlib import Path
from unittest.mock import MagicMock, patch

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


from src.pipeline.candidate_promotion import (
    get_retrain_min_variant_acc,
    promote,
    should_promote,
    update_env_weights_path,
)


def test_get_retrain_min_variant_acc_default_and_override(monkeypatch):
    monkeypatch.delenv("SMARTCART_RETRAIN_MIN_VARIANT_ACC", raising=False)
    assert get_retrain_min_variant_acc() == 0.5

    monkeypatch.setenv("SMARTCART_RETRAIN_MIN_VARIANT_ACC", "0.7")
    assert get_retrain_min_variant_acc() == 0.7


def test_should_promote_true_when_both_metrics_improve_or_tie():
    assert should_promote(candidate_map50=0.8, candidate_variant_acc=0.9, live_map50=0.7, live_variant_acc=0.9) is True


def test_should_promote_false_when_either_metric_regresses():
    assert should_promote(0.6, 0.9, live_map50=0.7, live_variant_acc=0.9) is False
    assert should_promote(0.8, 0.5, live_map50=0.7, live_variant_acc=0.6) is False


def test_update_env_weights_path_replaces_existing_line(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("SMARTCART_PORT=8000\nSMARTCART_WEIGHTS_PATH=./old/path.pt\nSMARTCART_HOST=0.0.0.0\n")

    update_env_weights_path(env_path, "./new/path.pt")

    lines = env_path.read_text().splitlines()
    assert "SMARTCART_WEIGHTS_PATH=./new/path.pt" in lines
    assert "SMARTCART_PORT=8000" in lines
    assert "SMARTCART_HOST=0.0.0.0" in lines


def test_update_env_weights_path_appends_when_missing(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("SMARTCART_PORT=8000\n")

    update_env_weights_path(env_path, "./new/path.pt")

    assert "SMARTCART_WEIGHTS_PATH=./new/path.pt" in env_path.read_text().splitlines()


def test_promote_copies_candidate_files_updates_env_and_state(tmp_path):
    candidate_dir = tmp_path / "candidates" / "retrain_1"
    candidate_dir.mkdir(parents=True)
    (candidate_dir / "catalog_prices.csv").write_text("class_id,product_name,price_usd\n")
    (candidate_dir / "gallery_index.npy").write_bytes(b"fake-npy")
    (candidate_dir / "gallery_meta.csv").write_text("class_id,product_name,file_name\n")

    artifacts_dir = tmp_path / "artifacts"
    env_path = tmp_path / ".env"
    env_path.write_text("SMARTCART_WEIGHTS_PATH=./old/best.pt\n")
    state_path = tmp_path / "state.json"

    promote(
        candidate_dir=candidate_dir,
        run_name="retrain_1",
        weights_path=tmp_path / "runs" / "retrain_1" / "weights" / "best.pt",
        map50=0.8,
        variant_acc=0.9,
        artifacts_dir=artifacts_dir,
        env_path=env_path,
        state_path=state_path,
    )

    assert (artifacts_dir / "catalog_prices.csv").exists()
    assert (artifacts_dir / "gallery_index.npy").exists()
    assert (artifacts_dir / "gallery_meta.csv").exists()
    assert f"SMARTCART_WEIGHTS_PATH={tmp_path / 'runs' / 'retrain_1' / 'weights' / 'best.pt'}" in env_path.read_text()

    state = json.loads(state_path.read_text())
    assert state["run_name"] == "retrain_1"
    assert state["map50"] == 0.8
    assert state["variant_accuracy"] == 0.9


from src.pipeline.candidate_promotion import seed_baseline_state


def test_seed_baseline_state_measures_live_model_and_writes_state(tmp_path):
    weights_path = tmp_path / "best.pt"
    data_yaml = tmp_path / "data.yaml"
    gallery_index_path = tmp_path / "gallery_index.npy"
    gallery_meta_path = tmp_path / "gallery_meta.csv"
    state_path = tmp_path / "al_pipeline_state.json"

    mock_auditor_instance = MagicMock()
    mock_auditor_instance.perform_validation_audit.return_value = (0.79, 0.55)

    with patch("src.models.auditor.CheckoutModelAuditor", return_value=mock_auditor_instance) as mock_auditor_cls, \
         patch("src.models.variant_auditor.compute_variant_accuracy", return_value=(0.92, 0)) as mock_variant_acc, \
         patch("numpy.load", return_value="fake-embeddings") as mock_np_load, \
         patch("pandas.read_csv", return_value="fake-meta") as mock_pd_read_csv:
        result = seed_baseline_state(
            weights_path=weights_path,
            data_yaml=data_yaml,
            gallery_index_path=gallery_index_path,
            gallery_meta_path=gallery_meta_path,
            state_path=state_path,
        )

    mock_auditor_cls.assert_called_once_with(weights_path)
    mock_auditor_instance.perform_validation_audit.assert_called_once_with(data_yaml)
    mock_np_load.assert_called_once_with(gallery_index_path)
    mock_pd_read_csv.assert_called_once_with(gallery_meta_path)
    mock_variant_acc.assert_called_once_with("fake-embeddings", "fake-meta")

    assert result == {
        "run_name": "baseline",
        "map50": 0.79,
        "variant_accuracy": 0.92,
        "pending_auto_imported": 0,
    }

    state = json.loads(state_path.read_text())
    assert state == result
