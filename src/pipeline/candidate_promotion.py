"""Candidate report and pipeline-state persistence for the autonomous retrain
loop, plus the mechanical promotion decision and file operations that move a
passing candidate into the live artifacts/weights path."""
import json
import os
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
