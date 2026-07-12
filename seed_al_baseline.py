"""Seeds al_pipeline_state.json with the currently-live model's real mAP50 and DINOv2
variant accuracy, so the first autonomous promotion decision compares a retrain
candidate against reality instead of a phantom 0.0 baseline. Run this once, before
loading the com.smartcart.al-scheduler launchd service."""
import os
from pathlib import Path

from dotenv import load_dotenv

from src.pipeline.candidate_promotion import get_promoted_state_path, seed_baseline_state

if __name__ == "__main__":
    load_dotenv()

    weights_path = Path(os.environ.get("SMARTCART_WEIGHTS_PATH", "./runs/detect/train/weights/best.pt"))
    data_yaml = Path("./synthetic_dataset/data.yaml")
    gallery_index_path = Path("./artifacts/gallery_index.npy")
    gallery_meta_path = Path("./artifacts/gallery_meta.csv")
    state_path = get_promoted_state_path()

    state = seed_baseline_state(
        weights_path=weights_path,
        data_yaml=data_yaml,
        gallery_index_path=gallery_index_path,
        gallery_meta_path=gallery_meta_path,
        state_path=state_path,
    )

    print(f"Seeded baseline state at {state_path}:")
    print(f"  mAP50={state['map50']:.4f}, variant_accuracy={state['variant_accuracy']:.4f}")
