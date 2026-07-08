import re
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.pipeline.training_pipeline import ModelTrainingPipeline, get_retrain_min_map50


def _pipeline(tmp_path):
    return ModelTrainingPipeline(
        dataset_root=Path("/dataset"),
        artifacts_dir=tmp_path / "artifacts",
        runs_dir=tmp_path / "runs",
        synth_root=tmp_path / "synthetic",
    )


def test_get_retrain_min_map50_default_and_override(monkeypatch):
    monkeypatch.delenv("SMARTCART_RETRAIN_MIN_MAP50", raising=False)
    assert get_retrain_min_map50() == 0.5

    monkeypatch.setenv("SMARTCART_RETRAIN_MIN_MAP50", "0.7")
    assert get_retrain_min_map50() == 0.7


def test_rebuild_catalog_uses_indexer_and_gallery(tmp_path):
    pipeline = _pipeline(tmp_path)
    fake_class_map = {"Fruit/Apple/Royal-Gala": 0}

    with patch("src.pipeline.training_pipeline.GroceryDatasetIndexer") as mock_indexer_cls, \
         patch("src.pipeline.training_pipeline.UnifiedProductGallery") as mock_gallery_cls:
        mock_indexer_cls.return_value.build_class_map.return_value = fake_class_map
        mock_gallery = MagicMock()
        mock_gallery_cls.return_value = mock_gallery

        result = pipeline.rebuild_catalog()

    mock_indexer_cls.assert_called_once_with(pipeline.dataset_root)
    mock_gallery_cls.assert_called_once_with(fake_class_map, pipeline.artifacts_dir)
    mock_gallery.compile_gallery.assert_called_once_with(pipeline.dataset_root)
    assert result == fake_class_map


def test_resynthesize_and_train_uses_isolated_dirs(tmp_path):
    pipeline = _pipeline(tmp_path)
    class_map = {"Fruit/Apple/Royal-Gala": 0}

    with patch("src.pipeline.training_pipeline.ProgrammaticCheckoutSynthesizer") as mock_synth_cls, \
         patch("src.pipeline.training_pipeline.YOLO") as mock_yolo_cls:
        mock_synth = MagicMock()
        mock_synth.write_yaml_config.return_value = tmp_path / "synthetic" / "my_run" / "data.yaml"
        mock_synth_cls.return_value = mock_synth
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model

        weights_path, data_yaml = pipeline.resynthesize_and_train(
            class_map, run_name="my_run", total_scenes_train=10, total_scenes_val=2, epochs=1
        )

    mock_synth_cls.assert_called_once_with(pipeline.dataset_root, pipeline.synth_root / "my_run", class_map)
    mock_synth.generate_split.assert_any_call("train", total_scenes=10)
    mock_synth.generate_split.assert_any_call("val", total_scenes=2)

    train_kwargs = mock_model.train.call_args.kwargs
    assert train_kwargs["project"] == str(pipeline.runs_dir.resolve())
    assert train_kwargs["name"] == "my_run"
    assert train_kwargs["exist_ok"] is False
    assert train_kwargs["patience"] == 15
    assert train_kwargs["epochs"] == 1
    assert train_kwargs["device"] == "mps"

    assert weights_path == pipeline.runs_dir / "my_run" / "weights" / "best.pt"
    assert data_yaml == mock_synth.write_yaml_config.return_value
    # never targets the shared production run name
    assert train_kwargs["name"] != "train"


def test_audit_passes_above_threshold(tmp_path, monkeypatch):
    monkeypatch.setenv("SMARTCART_RETRAIN_MIN_MAP50", "0.5")
    pipeline = _pipeline(tmp_path)

    with patch("src.pipeline.training_pipeline.CheckoutModelAuditor") as mock_auditor_cls:
        mock_auditor_cls.return_value.perform_validation_audit.return_value = (0.6, 0.4)
        passed, map50, map50_95 = pipeline.audit(Path("best.pt"), Path("data.yaml"))

    assert passed is True
    assert map50 == 0.6
    assert map50_95 == 0.4


def test_audit_fails_below_threshold(tmp_path, monkeypatch):
    monkeypatch.setenv("SMARTCART_RETRAIN_MIN_MAP50", "0.5")
    pipeline = _pipeline(tmp_path)

    with patch("src.pipeline.training_pipeline.CheckoutModelAuditor") as mock_auditor_cls:
        mock_auditor_cls.return_value.perform_validation_audit.return_value = (0.3, 0.2)
        passed, map50, map50_95 = pipeline.audit(Path("best.pt"), Path("data.yaml"))

    assert passed is False
    assert map50 == 0.3


def test_run_orchestrates_and_reports_without_promotion(tmp_path, monkeypatch):
    monkeypatch.delenv("SMARTCART_WEIGHTS_PATH", raising=False)
    pipeline = _pipeline(tmp_path)

    with patch.object(pipeline, "rebuild_catalog", return_value={"Fruit/Apple/Royal-Gala": 0}) as mock_rebuild, \
         patch.object(
             pipeline, "resynthesize_and_train", return_value=(Path("weights/best.pt"), Path("data.yaml"))
         ) as mock_resynth, \
         patch.object(pipeline, "audit", return_value=(True, 0.6, 0.4)) as mock_audit:
        result = pipeline.run(total_scenes_train=10, total_scenes_val=2, epochs=1)

    mock_rebuild.assert_called_once()
    run_name_used = mock_resynth.call_args.args[1] if len(mock_resynth.call_args.args) > 1 else mock_resynth.call_args.kwargs["run_name"]
    assert re.match(r"^retrain_\d{8}_\d{6}$", run_name_used)
    mock_audit.assert_called_once_with(Path("weights/best.pt"), Path("data.yaml"))

    assert result == {
        "run_name": run_name_used,
        "weights_path": Path("weights/best.pt"),
        "data_yaml": Path("data.yaml"),
        "passed": True,
        "map50": 0.6,
        "map50_95": 0.4,
        "min_map50": get_retrain_min_map50(),
    }
    # the pipeline never reads or writes SMARTCART_WEIGHTS_PATH anywhere
    import os
    assert "SMARTCART_WEIGHTS_PATH" not in os.environ
