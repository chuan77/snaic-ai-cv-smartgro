"""Orchestrates one autonomous active-learning scheduler tick: auto-import staged
captures via the VLM, and if enough new crops have accumulated, run a full
retrain cycle against an isolated candidate directory, gate it on both YOLO
mAP50 and DINOv2 held-out variant accuracy, and auto-promote if it doesn't
regress either metric versus the currently-live model."""
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.auto_labeler import auto_import_staging_dir, get_autolabel_min_conf
from src.data.gallery import GroceryDatasetIndexer
from src.deploy.active_learning_capture import get_capture_threshold
from src.models.variant_auditor import compute_variant_accuracy
from src.pipeline.candidate_promotion import (
    get_retrain_min_variant_acc,
    promote,
    read_promoted_state,
    should_promote,
    write_candidate_report,
    write_promoted_state,
)
from src.pipeline.training_pipeline import ModelTrainingPipeline


def get_retrain_trigger_count() -> int:
    return int(os.environ.get("SMARTCART_AL_RETRAIN_TRIGGER_COUNT", "50"))


def run_retrain_cycle(
    dataset_root: Path, candidates_root: Path, runs_dir: Path, synth_root: Path
) -> tuple[dict, Path]:
    """Reuses ModelTrainingPipeline unmodified against a temp candidate directory,
    then renames it to the run's own name -- keeps candidate artifacts fully
    isolated from the live artifacts/ path without touching the pipeline class."""
    tmp_candidate_dir = candidates_root / "_tmp_candidate"
    shutil.rmtree(tmp_candidate_dir, ignore_errors=True)

    pipeline = ModelTrainingPipeline(
        dataset_root=dataset_root, artifacts_dir=tmp_candidate_dir, runs_dir=runs_dir, synth_root=synth_root
    )
    result = pipeline.run()

    candidate_dir = candidates_root / result["run_name"]
    tmp_candidate_dir.rename(candidate_dir)
    return result, candidate_dir


def restart_api_service() -> None:
    subprocess.run(["launchctl", "kickstart", "-k", "com.smartcart.api"], check=False)


def run_scheduler_tick(
    dataset_root: Path,
    staging_dir: Path,
    artifacts_dir: Path,
    candidates_root: Path,
    runs_dir: Path,
    synth_root: Path,
    env_path: Path,
    state_path: Path,
) -> dict:
    class_map = GroceryDatasetIndexer(dataset_root).build_class_map()
    class_names = list(class_map)

    imported = auto_import_staging_dir(
        staging_dir,
        class_names,
        dataset_root,
        get_autolabel_min_conf(),
        get_capture_threshold(),
        review_dir=artifacts_dir / "al_review",
    )

    state = read_promoted_state(state_path)
    pending = state["pending_auto_imported"] + len(imported)

    if pending < get_retrain_trigger_count():
        write_promoted_state(state_path, {**state, "pending_auto_imported": pending})
        return {"retrained": False, "auto_imported": len(imported)}

    train_result, candidate_dir = run_retrain_cycle(dataset_root, candidates_root, runs_dir, synth_root)

    embeddings = np.load(candidate_dir / "gallery_index.npy")
    meta = pd.read_csv(candidate_dir / "gallery_meta.csv")
    variant_acc, excluded = compute_variant_accuracy(embeddings, meta)

    promoted = (
        should_promote(
            candidate_map50=train_result["map50"],
            candidate_variant_acc=variant_acc,
            live_map50=state["map50"],
            live_variant_acc=state["variant_accuracy"],
        )
        and train_result["passed"]
        and variant_acc >= get_retrain_min_variant_acc()
    )

    write_candidate_report(
        candidate_dir,
        {
            **train_result,
            "weights_path": str(train_result["weights_path"]),
            "data_yaml": str(train_result.get("data_yaml", "")),
            "variant_accuracy": variant_acc,
            "variant_excluded_count": excluded,
            "promoted": promoted,
            "auto_imported_count": len(imported),
        },
    )

    if promoted:
        promote(
            candidate_dir=candidate_dir,
            run_name=train_result["run_name"],
            weights_path=train_result["weights_path"],
            map50=train_result["map50"],
            variant_acc=variant_acc,
            artifacts_dir=artifacts_dir,
            env_path=env_path,
            state_path=state_path,
        )
        restart_api_service()
    else:
        write_promoted_state(state_path, {**state, "pending_auto_imported": 0})

    return {"retrained": True, "auto_imported": len(imported), "promoted": promoted, "run_name": train_result["run_name"]}
