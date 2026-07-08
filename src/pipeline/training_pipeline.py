"""Orchestrates a full retrain cycle: reindex the catalog, resynthesize scenes, train, and
gate on CheckoutModelAuditor's mAP — without touching whatever weights the checkout API
currently serves (runs/detect/train/, ./synthetic_dataset/, and SMARTCART_WEIGHTS_PATH are
never written to or read by this module)."""
import os
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

from src.data.gallery import GroceryDatasetIndexer, UnifiedProductGallery
from src.data.synthesizer import ProgrammaticCheckoutSynthesizer
from src.models.auditor import CheckoutModelAuditor


def get_retrain_min_map50() -> float:
    return float(os.environ.get("SMARTCART_RETRAIN_MIN_MAP50", "0.5"))


class ModelTrainingPipeline:
    """Orchestrates localized accelerated execution cycles."""

    def __init__(self, dataset_root: Path, artifacts_dir: Path, runs_dir: Path, synth_root: Path):
        self.dataset_root = dataset_root
        self.artifacts_dir = artifacts_dir
        self.runs_dir = runs_dir
        self.synth_root = synth_root

    def rebuild_catalog(self) -> dict[str, int]:
        class_map = GroceryDatasetIndexer(self.dataset_root).build_class_map()
        UnifiedProductGallery(class_map, self.artifacts_dir).compile_gallery(self.dataset_root)
        return class_map

    def resynthesize_and_train(
        self,
        class_map: dict[str, int],
        run_name: str,
        total_scenes_train: int,
        total_scenes_val: int,
        epochs: int,
    ) -> tuple[Path, Path]:
        synth = ProgrammaticCheckoutSynthesizer(self.dataset_root, self.synth_root / run_name, class_map)
        synth.generate_split("train", total_scenes=total_scenes_train)
        synth.generate_split("val", total_scenes=total_scenes_val)
        data_yaml = synth.write_yaml_config()

        model = YOLO("yolo11n.pt")
        model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=640,
            device="mps",
            workers=2,
            patience=15,
            project=str(self.runs_dir.resolve()),
            name=run_name,
            exist_ok=False,
        )
        weights_path = self.runs_dir / run_name / "weights" / "best.pt"
        return weights_path, data_yaml

    def audit(self, weights_path: Path, data_yaml: Path) -> tuple[bool, float, float]:
        map50, map50_95 = CheckoutModelAuditor(weights_path).perform_validation_audit(data_yaml)
        passed = map50 >= get_retrain_min_map50()
        return passed, map50, map50_95

    def run(self, total_scenes_train: int = 3300, total_scenes_val: int = 660, epochs: int = 40) -> dict:
        class_map = self.rebuild_catalog()
        run_name = f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        weights_path, data_yaml = self.resynthesize_and_train(
            class_map, run_name, total_scenes_train, total_scenes_val, epochs
        )
        passed, map50, map50_95 = self.audit(weights_path, data_yaml)
        return {
            "run_name": run_name,
            "weights_path": weights_path,
            "data_yaml": data_yaml,
            "passed": passed,
            "map50": map50,
            "map50_95": map50_95,
            "min_map50": get_retrain_min_map50(),
        }
