from pathlib import Path
from src.models.auditor import CheckoutModelAuditor

if __name__ == "__main__":
    print("Executing Day 3: Transfer Learning Layer Tuning Custom Classes")
    auditor = CheckoutModelAuditor(Path("./runs/detect/train/weights/best.pt"))
    auditor.perform_validation_audit(Path("./synthetic_dataset/data.yaml"))