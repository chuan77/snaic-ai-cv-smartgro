"""Day 3: diagnoses a trained LinearHead's validation errors — per-class accuracy, confusions,
and a confusion-matrix heatmap ("explanation heatmap"). Used by main_day3_recognize.py."""
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("Day3Recognition")

# Above this many classes, per-tick text labels on the heatmap become unreadable.
MAX_LABELED_CLASSES = 20


class RecognitionAuditor:
    """Diagnoses a trained LinearHead's validation predictions against ground truth."""

    def __init__(self, class_names: list[str], output_dir: Path):
        self.class_names = class_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def analyze(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        labels = range(len(self.class_names))
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        report_dict = classification_report(
            y_true, y_pred, labels=labels, target_names=self.class_names,
            output_dict=True, zero_division=0,
        )
        report_text = classification_report(
            y_true, y_pred, labels=labels, target_names=self.class_names, zero_division=0,
        )
        (self.output_dir / "classification_report.txt").write_text(report_text)

        # "accuracy" here is per-class recall (TP / (TP+FN)) — the standard reading of
        # "per-class accuracy" for a multi-class confusion matrix.
        per_class_report = pd.DataFrame([
            {
                "class": name,
                "support": int(report_dict[name]["support"]),
                "accuracy": report_dict[name]["recall"],
            }
            for name in self.class_names
        ])
        per_class_report.to_csv(self.output_dir / "per_class_report.csv", index=False)

        heatmap_path = self.save_confusion_heatmap(cm)
        logger.info(f"Recognition error analysis written to {self.output_dir}")
        return {
            "confusion_matrix": cm,
            "per_class_report": per_class_report,
            "heatmap_path": heatmap_path,
        }

    def save_confusion_heatmap(self, cm: np.ndarray) -> Path:
        n = len(self.class_names)
        side = max(6, n * 0.3)
        fig, ax = plt.subplots(figsize=(side, side))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        if n <= MAX_LABELED_CLASSES:
            ax.set_xticks(range(n))
            ax.set_xticklabels(self.class_names, rotation=90)
            ax.set_yticks(range(n))
            ax.set_yticklabels(self.class_names)
        fig.colorbar(im, ax=ax)

        path = self.output_dir / "confusion_matrix.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path
