import logging
import os
from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import train_test_split

from src.data.recognition_dataset import build_recognition_dataframe, load_synthetic_val_ground_truth
from src.models.closed_loop import run_closed_loop_evaluation
from src.models.dino_extractor import extract_features_from_paths, load_backbone
from src.models.recognition_auditor import RecognitionAuditor
from src.models.recognizer import LinearHead
from src.models.yolo_detector import YoloDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("Day3Recognize")

DATASET_ROOT = Path("./dataset/GroceryStoreDataset/dataset/train")
ARTIFACTS_DIR = Path(os.environ.get("SMARTCART_DAY3_ARTIFACTS_DIR", "./artifacts/day3_recognition"))
YOLO_WEIGHTS = Path("./runs/detect/train/weights/best.pt")
SYNTH_DATA_YAML = Path("./synthetic_dataset/data.yaml")
SYNTH_VAL_IMAGES = Path("./synthetic_dataset/val/images")
SYNTH_VAL_LABELS = Path("./synthetic_dataset/val/labels")


if __name__ == "__main__":
    if not DATASET_ROOT.exists():
        logger.error(f"Dataset root source missing at {DATASET_ROOT}. Please clone the repository first.")
    else:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

        data = build_recognition_dataframe(DATASET_ROOT)
        classes = sorted(data["fine"].unique().tolist())
        cls_to_i = {c: i for i, c in enumerate(classes)}

        features_path = ARTIFACTS_DIR / "features.npy"
        labels_path = ARTIFACTS_DIR / "labels.npy"
        backbone = None

        if features_path.exists() and labels_path.exists():
            logger.info(f"Loading cached DINOv2 features from {ARTIFACTS_DIR}")
            feats = np.load(features_path)
            y = np.load(labels_path)
        else:
            logger.info(f"Extracting DINOv2 features for {len(data)} images...")
            backbone = load_backbone()
            feats = extract_features_from_paths(list(data["crop_path"]), backbone)
            y = np.array([cls_to_i[f] for f in data["fine"]])
            np.save(features_path, feats)
            np.save(labels_path, y)
            data[["crop_path", "fine"]].to_csv(ARTIFACTS_DIR / "crop_paths.csv", index=False)

        # Stratify only when every class has enough examples for a train/val split.
        counts = np.bincount(y, minlength=len(classes))
        strat = y if counts.min() >= 2 else None
        Xtr, Xval, ytr, yval = train_test_split(feats, y, test_size=0.3, random_state=0, stratify=strat)
        logger.info(f"train {Xtr.shape}, val {Xval.shape} | {len(classes)} classes")

        head = LinearHead(in_dim=feats.shape[1], n_classes=len(classes)).fit(Xtr, ytr, epochs=200)
        val_pred = head.predict(Xval)
        val_accuracy = float((val_pred == yval).mean())
        logger.info(f"train accuracy: {head.train_accuracy_:.3f} | validation accuracy: {val_accuracy:.3f}")
        head.save(ARTIFACTS_DIR / "linear_head.pt")

        RecognitionAuditor(class_names=classes, output_dir=ARTIFACTS_DIR).analyze(yval, val_pred)

        # Close the loop: classify Day 2's real trained detector's crops, scored against
        # synthetic ground truth (not YOLO's own class guess).
        if YOLO_WEIGHTS.exists() and SYNTH_VAL_IMAGES.exists() and SYNTH_DATA_YAML.exists():
            data_yaml_names = yaml.safe_load(SYNTH_DATA_YAML.read_text())["names"]
            missing = [name for name in data_yaml_names.values() if name not in cls_to_i]
            if missing:
                logger.error(
                    f"synthetic_dataset/data.yaml references classes no longer in the live "
                    f"dataset: {missing}. Skipping closed-loop evaluation."
                )
            else:
                backbone = backbone if backbone is not None else load_backbone()
                detector = YoloDetector(YOLO_WEIGHTS)
                ground_truth = load_synthetic_val_ground_truth(
                    SYNTH_DATA_YAML, SYNTH_VAL_IMAGES, SYNTH_VAL_LABELS
                )
                closed_loop_result = run_closed_loop_evaluation(
                    detector, head, backbone, ground_truth, cls_to_i
                )
                logger.info(
                    f"Closed-loop accuracy: {closed_loop_result['accuracy']:.3f} "
                    f"({closed_loop_result['n_evaluated']} evaluated, "
                    f"{closed_loop_result['n_unmatched']} unmatched detections)"
                )
        else:
            logger.warning(
                "Skipping closed-loop evaluation: trained YOLO weights or "
                "synthetic_dataset/val not found."
            )
