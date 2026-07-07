from pathlib import Path
from src.deploy.register import AutonomousPOSRegister

if __name__ == "__main__":
    print("Executing Day 5: ONNX Runtime Compiler & Gradio Boot Sequence")
    app = AutonomousPOSRegister(
         model_weights=Path("yolo11n.pt"), # Update to './runs/detect/train/weights/best.pt' post-training
        catalog_prices_csv=Path("./artifacts/catalog_prices.csv")
    )
    app.launch_interface()