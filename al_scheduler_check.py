"""Thin entrypoint for the launchd-scheduled active-learning checker. All
logic lives in src/pipeline/al_scheduler.py -- see that module's docstring."""
from pathlib import Path

from dotenv import load_dotenv

from src.pipeline.al_scheduler import run_scheduler_tick
from src.deploy.active_learning_capture import get_capture_dir
from src.pipeline.candidate_promotion import get_promoted_state_path

DATASET_ROOT = Path("./dataset/GroceryStoreDataset/dataset/train")

if __name__ == "__main__":
    load_dotenv()
    result = run_scheduler_tick(
        dataset_root=DATASET_ROOT,
        staging_dir=get_capture_dir(),
        artifacts_dir=Path("./artifacts"),
        candidates_root=Path("./artifacts/candidates"),
        runs_dir=Path("./runs/detect"),
        synth_root=Path("./synthetic_dataset_retrain"),
        env_path=Path(".env"),
        state_path=get_promoted_state_path(),
    )
    print(result)
