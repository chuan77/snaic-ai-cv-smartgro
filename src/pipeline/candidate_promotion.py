"""Candidate report and pipeline-state persistence for the autonomous retrain
loop, plus the mechanical promotion decision and file operations that move a
passing candidate into the live artifacts/weights path."""
import json
import os
import shutil
from pathlib import Path

DEFAULT_STATE_PATH = Path("./artifacts/al_pipeline_state.json")
_DEFAULT_STATE = {"run_name": None, "map50": 0.0, "variant_accuracy": 0.0, "pending_auto_imported": 0}


def get_promoted_state_path() -> Path:
    return Path(os.environ.get("SMARTCART_AL_STATE_PATH", str(DEFAULT_STATE_PATH)))


def read_promoted_state(state_path: Path) -> dict:
    if not state_path.exists():
        return dict(_DEFAULT_STATE)
    return {**_DEFAULT_STATE, **json.loads(state_path.read_text())}


def write_promoted_state(state_path: Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state))


def write_candidate_report(candidate_dir: Path, report: dict) -> Path:
    candidate_dir.mkdir(parents=True, exist_ok=True)
    report_path = candidate_dir / "report.json"
    report_path.write_text(json.dumps(report))
    return report_path


_CANDIDATE_FILES = ("catalog_prices.csv", "gallery_index.npy", "gallery_meta.csv")


def get_retrain_min_variant_acc() -> float:
    return float(os.environ.get("SMARTCART_RETRAIN_MIN_VARIANT_ACC", "0.5"))


def should_promote(
    candidate_map50: float, candidate_variant_acc: float, live_map50: float, live_variant_acc: float
) -> bool:
    return candidate_map50 >= live_map50 and candidate_variant_acc >= live_variant_acc


def update_env_weights_path(env_path: Path, new_value: str) -> None:
    lines = env_path.read_text().splitlines() if env_path.exists() else []
    for i, line in enumerate(lines):
        if line.startswith("SMARTCART_WEIGHTS_PATH="):
            lines[i] = f"SMARTCART_WEIGHTS_PATH={new_value}"
            break
    else:
        lines.append(f"SMARTCART_WEIGHTS_PATH={new_value}")
    env_path.write_text("\n".join(lines) + "\n")


def promote(
    candidate_dir: Path,
    run_name: str,
    weights_path: Path,
    map50: float,
    variant_acc: float,
    artifacts_dir: Path,
    env_path: Path,
    state_path: Path,
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for file_name in _CANDIDATE_FILES:
        shutil.copyfile(candidate_dir / file_name, artifacts_dir / file_name)

    update_env_weights_path(env_path, str(weights_path))
    write_promoted_state(
        state_path,
        {"run_name": run_name, "map50": map50, "variant_accuracy": variant_acc, "pending_auto_imported": 0},
    )
