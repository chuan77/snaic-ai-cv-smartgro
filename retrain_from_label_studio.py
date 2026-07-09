"""Pulls corrected annotations from Label Studio and runs a full retrain + audit cycle,
without touching the weights the checkout API currently serves. Promotion is manual:
update SMARTCART_WEIGHTS_PATH and restart the API once a candidate passes the audit gate."""
from pathlib import Path

from dotenv import load_dotenv

from src.data.gallery import GroceryDatasetIndexer
from src.data.label_studio_export_pull import pull_and_import_from_label_studio
from src.deploy.label_studio_push import get_ls_project_id
from src.pipeline.training_pipeline import ModelTrainingPipeline

DATASET_ROOT = Path("./dataset/GroceryStoreDataset/dataset/train")

if __name__ == "__main__":
    load_dotenv()
    project_id = get_ls_project_id()
    if not project_id:
        raise SystemExit("SMARTCART_LS_PROJECT_ID must be set to pull corrections from Label Studio.")

    current_class_map = GroceryDatasetIndexer(DATASET_ROOT).build_class_map()
    written = pull_and_import_from_label_studio(
        project_id, current_class_map, DATASET_ROOT, Path("./dataset/label_studio_pulls/latest")
    )
    if written:
        print(f"Imported {len(written)} corrected crops from Label Studio project {project_id}.")
    else:
        print(f"No new corrected crops found in Label Studio project {project_id}; retraining on the current catalog.")

    pipeline = ModelTrainingPipeline(
        dataset_root=DATASET_ROOT,
        artifacts_dir=Path("./artifacts"),
        runs_dir=Path("./runs/detect"),
        synth_root=Path("./synthetic_dataset_retrain"),
    )
    result = pipeline.run()

    status = "PASSED" if result["passed"] else "FAILED"
    print(f"\nRetrain {status}: mAP50={result['map50']:.4f} (min {result['min_map50']:.4f}), mAP50-95={result['map50_95']:.4f}")
    print(f"New weights: {result['weights_path']}")
    if result["passed"]:
        print("To promote: set SMARTCART_WEIGHTS_PATH to the path above in .env and restart the API.")
    else:
        print("Not promoting — current production weights are untouched.")
