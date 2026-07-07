"""Day 3: validates a trained YOLO checkpoint against a held-out split. Used by main_day3_fine_tune.py."""
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
        
        # Log localized mean Average Precision parameters
        logger.info(f"Evaluation mAP50-95 Accuracy score: {metrics.box.map:.4f}")
        logger.info(f"Evaluation mAP50 Accuracy score: {metrics.box.map50:.4f}")
        logger.info("Day 3 performance tracking metrics saved successfully.")