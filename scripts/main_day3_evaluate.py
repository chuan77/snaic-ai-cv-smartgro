# main_day3_evaluate.py
import logging
from pathlib import Path
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("Day3Audit")

class CheckoutModelAuditor:
    """Runs data audits over compiled validation sets to pinpoint classification error trends."""
    def __init__(self, check_weights: Path):
        self.model = YOLO(str(check_weights))

    def perform_validation_audit(self, data_yaml: Path):
        logger.info("Executing network validation verification loops...")
        metrics = self.model.val(data=str(data_yaml), device="mps")
        logger.info(f"Evaluation mAP50-95 Accuracy score: {metrics.box.map:.4f}")
        logger.info(f"Evaluation mAP50 Accuracy score: {metrics.box.map50:.4f}")

if __name__ == "__main__":
    # auditor = CheckoutModelAuditor(Path("./runs/detect/train/weights/best.pt"))
    # auditor.perform_validation_audit(Path("./synthetic_dataset/data.yaml"))
    pass
