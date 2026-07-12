from pathlib import Path
from unittest.mock import MagicMock, patch

from src.pipeline.al_scheduler import (
    get_retrain_trigger_count,
    run_retrain_cycle,
    run_scheduler_tick,
)


def test_get_retrain_trigger_count_default_and_override(monkeypatch):
    monkeypatch.delenv("SMARTCART_AL_RETRAIN_TRIGGER_COUNT", raising=False)
    assert get_retrain_trigger_count() == 50

    monkeypatch.setenv("SMARTCART_AL_RETRAIN_TRIGGER_COUNT", "10")
    assert get_retrain_trigger_count() == 10


def test_run_retrain_cycle_renames_temp_candidate_dir_to_run_name(tmp_path):
    candidates_root = tmp_path / "candidates"

    with patch("src.pipeline.al_scheduler.ModelTrainingPipeline") as mock_pipeline_cls:
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {
            "run_name": "retrain_20260711_000000",
            "weights_path": Path("weights/best.pt"),
            "data_yaml": Path("data.yaml"),
            "passed": True,
            "map50": 0.8,
            "map50_95": 0.6,
            "min_map50": 0.5,
        }
        mock_pipeline_cls.return_value = mock_pipeline
        # simulate the pipeline actually writing gallery files into the tmp candidate dir
        def fake_run(**kwargs):
            tmp_candidate = candidates_root / "_tmp_candidate"
            tmp_candidate.mkdir(parents=True, exist_ok=True)
            (tmp_candidate / "gallery_index.npy").write_bytes(b"fake")
            return mock_pipeline.run.return_value
        mock_pipeline.run.side_effect = fake_run

        result, candidate_dir = run_retrain_cycle(
            dataset_root=tmp_path / "dataset",
            candidates_root=candidates_root,
            runs_dir=tmp_path / "runs",
            synth_root=tmp_path / "synth",
        )

    assert candidate_dir == candidates_root / "retrain_20260711_000000"
    assert candidate_dir.exists()
    assert (candidate_dir / "gallery_index.npy").exists()
    assert not (candidates_root / "_tmp_candidate").exists()
    assert result["run_name"] == "retrain_20260711_000000"


def test_run_scheduler_tick_skips_retrain_below_trigger_count(tmp_path, monkeypatch):
    monkeypatch.setenv("SMARTCART_AL_RETRAIN_TRIGGER_COUNT", "50")
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()

    with patch("src.pipeline.al_scheduler.GroceryDatasetIndexer") as mock_indexer_cls, \
         patch("src.pipeline.al_scheduler.auto_import_staging_dir", return_value=[Path("a.jpg")]) as mock_auto_import, \
         patch("src.pipeline.al_scheduler.run_retrain_cycle") as mock_retrain:
        mock_indexer_cls.return_value.build_class_map.return_value = {"Fruit/Apple/Royal-Gala": 0}

        result = run_scheduler_tick(
            dataset_root=tmp_path / "dataset",
            staging_dir=staging_dir,
            artifacts_dir=tmp_path / "artifacts",
            candidates_root=tmp_path / "candidates",
            runs_dir=tmp_path / "runs",
            synth_root=tmp_path / "synth",
            env_path=tmp_path / ".env",
            state_path=tmp_path / "state.json",
        )

    mock_auto_import.assert_called_once()
    mock_retrain.assert_not_called()
    assert result["retrained"] is False
    assert result["auto_imported"] == 1


def test_run_scheduler_tick_promotes_when_candidate_beats_live_metrics(tmp_path, monkeypatch):
    monkeypatch.setenv("SMARTCART_AL_RETRAIN_TRIGGER_COUNT", "1")
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    candidate_dir = tmp_path / "candidates" / "retrain_1"
    candidate_dir.mkdir(parents=True)
    for fname in ("catalog_prices.csv", "gallery_index.npy", "gallery_meta.csv"):
        (candidate_dir / fname).write_bytes(b"fake")

    with patch("src.pipeline.al_scheduler.GroceryDatasetIndexer") as mock_indexer_cls, \
         patch("src.pipeline.al_scheduler.auto_import_staging_dir", return_value=[Path("a.jpg")]), \
         patch("src.pipeline.al_scheduler.run_retrain_cycle") as mock_retrain, \
         patch("src.pipeline.al_scheduler.compute_variant_accuracy", return_value=(0.9, 0)), \
         patch("src.pipeline.al_scheduler.restart_api_service") as mock_restart, \
         patch("src.pipeline.al_scheduler.pd.read_csv"), \
         patch("src.pipeline.al_scheduler.np.load"):
        mock_indexer_cls.return_value.build_class_map.return_value = {"Fruit/Apple/Royal-Gala": 0}
        mock_retrain.return_value = (
            {
                "run_name": "retrain_1",
                "weights_path": Path("weights/best.pt"),
                "passed": True,
                "map50": 0.9,
                "min_map50": 0.5,
            },
            candidate_dir,
        )

        result = run_scheduler_tick(
            dataset_root=tmp_path / "dataset",
            staging_dir=staging_dir,
            artifacts_dir=tmp_path / "artifacts",
            candidates_root=tmp_path / "candidates",
            runs_dir=tmp_path / "runs",
            synth_root=tmp_path / "synth",
            env_path=tmp_path / ".env",
            state_path=tmp_path / "state.json",
        )

    assert result["retrained"] is True
    assert result["promoted"] is True
    mock_restart.assert_called_once()
    assert (tmp_path / "artifacts" / "catalog_prices.csv").exists()


def test_run_scheduler_tick_does_not_promote_when_map50_gate_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("SMARTCART_AL_RETRAIN_TRIGGER_COUNT", "1")
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    candidate_dir = tmp_path / "candidates" / "retrain_1"
    candidate_dir.mkdir(parents=True)

    with patch("src.pipeline.al_scheduler.GroceryDatasetIndexer") as mock_indexer_cls, \
         patch("src.pipeline.al_scheduler.auto_import_staging_dir", return_value=[Path("a.jpg")]), \
         patch("src.pipeline.al_scheduler.run_retrain_cycle") as mock_retrain, \
         patch("src.pipeline.al_scheduler.compute_variant_accuracy", return_value=(0.9, 0)), \
         patch("src.pipeline.al_scheduler.restart_api_service") as mock_restart, \
         patch("src.pipeline.al_scheduler.pd.read_csv"), \
         patch("src.pipeline.al_scheduler.np.load"):
        mock_indexer_cls.return_value.build_class_map.return_value = {"Fruit/Apple/Royal-Gala": 0}
        mock_retrain.return_value = (
            {"run_name": "retrain_1", "weights_path": Path("weights/best.pt"), "passed": False, "map50": 0.3, "min_map50": 0.5},
            candidate_dir,
        )

        result = run_scheduler_tick(
            dataset_root=tmp_path / "dataset",
            staging_dir=staging_dir,
            artifacts_dir=tmp_path / "artifacts",
            candidates_root=tmp_path / "candidates",
            runs_dir=tmp_path / "runs",
            synth_root=tmp_path / "synth",
            env_path=tmp_path / ".env",
            state_path=tmp_path / "state.json",
        )

    assert result["promoted"] is False
    mock_restart.assert_not_called()
    assert not (tmp_path / "artifacts" / "catalog_prices.csv").exists()


def test_run_scheduler_tick_does_not_promote_when_variant_acc_gate_fails(tmp_path, monkeypatch):
    """Correction 2: even when should_promote() and the mAP50 floor both pass,
    a candidate below the DINOv2 variant-accuracy floor must not be promoted."""
    monkeypatch.setenv("SMARTCART_AL_RETRAIN_TRIGGER_COUNT", "1")
    monkeypatch.setenv("SMARTCART_RETRAIN_MIN_VARIANT_ACC", "0.95")
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    candidate_dir = tmp_path / "candidates" / "retrain_1"
    candidate_dir.mkdir(parents=True)

    with patch("src.pipeline.al_scheduler.GroceryDatasetIndexer") as mock_indexer_cls, \
         patch("src.pipeline.al_scheduler.auto_import_staging_dir", return_value=[Path("a.jpg")]), \
         patch("src.pipeline.al_scheduler.run_retrain_cycle") as mock_retrain, \
         patch("src.pipeline.al_scheduler.compute_variant_accuracy", return_value=(0.9, 0)), \
         patch("src.pipeline.al_scheduler.restart_api_service") as mock_restart, \
         patch("src.pipeline.al_scheduler.pd.read_csv"), \
         patch("src.pipeline.al_scheduler.np.load"):
        mock_indexer_cls.return_value.build_class_map.return_value = {"Fruit/Apple/Royal-Gala": 0}
        mock_retrain.return_value = (
            {
                "run_name": "retrain_1",
                "weights_path": Path("weights/best.pt"),
                "passed": True,
                "map50": 0.9,
                "min_map50": 0.5,
            },
            candidate_dir,
        )

        result = run_scheduler_tick(
            dataset_root=tmp_path / "dataset",
            staging_dir=staging_dir,
            artifacts_dir=tmp_path / "artifacts",
            candidates_root=tmp_path / "candidates",
            runs_dir=tmp_path / "runs",
            synth_root=tmp_path / "synth",
            env_path=tmp_path / ".env",
            state_path=tmp_path / "state.json",
        )

    assert result["promoted"] is False
    mock_restart.assert_not_called()
    assert not (tmp_path / "artifacts" / "catalog_prices.csv").exists()


def test_run_scheduler_tick_passes_review_dir_to_auto_import(tmp_path, monkeypatch):
    """Correction 1: auto_import_staging_dir must be called with review_dir wired
    to artifacts_dir / 'al_review' so Task 9's passive audit trail is active."""
    monkeypatch.setenv("SMARTCART_AL_RETRAIN_TRIGGER_COUNT", "50")
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    artifacts_dir = tmp_path / "artifacts"

    with patch("src.pipeline.al_scheduler.GroceryDatasetIndexer") as mock_indexer_cls, \
         patch("src.pipeline.al_scheduler.auto_import_staging_dir", return_value=[]) as mock_auto_import, \
         patch("src.pipeline.al_scheduler.run_retrain_cycle") as mock_retrain:
        mock_indexer_cls.return_value.build_class_map.return_value = {"Fruit/Apple/Royal-Gala": 0}

        run_scheduler_tick(
            dataset_root=tmp_path / "dataset",
            staging_dir=staging_dir,
            artifacts_dir=artifacts_dir,
            candidates_root=tmp_path / "candidates",
            runs_dir=tmp_path / "runs",
            synth_root=tmp_path / "synth",
            env_path=tmp_path / ".env",
            state_path=tmp_path / "state.json",
        )

    mock_retrain.assert_not_called()
    _, kwargs = mock_auto_import.call_args
    assert kwargs.get("review_dir") == artifacts_dir / "al_review"
