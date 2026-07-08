# main_day2_train.py
import logging
import random
import cv2
import numpy as np
import yaml
from pathlib import Path
from ultralytics import YOLO
from typing import Dict
from src.data.gallery import GroceryDatasetIndexer
from src.data.synthesizer import ProgrammaticCheckoutSynthesizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("Day2Synthesis")
if __name__ == "__main__":
    print("Executing Day 2: Baseline Detector Optimization Train")
    root_path = Path("./dataset/GroceryStoreDataset/dataset/train")
    cmap = GroceryDatasetIndexer(root_path).build_class_map()
    synth = ProgrammaticCheckoutSynthesizer(root_path, Path("./synthetic_dataset"), cmap)
    synth.generate_split("train", total_scenes=3300)
    synth.generate_split("val", total_scenes=660)
    ypath = synth.write_yaml_config()

    # Initialize YOLO Network Training Optimization Model
    model = YOLO("yolo11n.pt")
    model.train(
        data=str(ypath),
        epochs=40,
        imgsz=640,
        device="mps",
        workers=2,
        patience=15,
        project=str(Path("./runs/detect").resolve()),
        name="train",
        exist_ok=True,
    )