# main_day2_train.py
import logging
import random
import cv2
import numpy as np
import yaml
from pathlib import Path
from ultralytics import YOLO
from typing import Dict
from src.data.synthesizer import ProgrammaticCheckoutSynthesizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("Day2Synthesis")
if __name__ == "__main__":
    print("Executing Day 2: Baseline Detector Optimization Train")
    cmap = {
        "Fruit/Apple/Royal-Gala": 0,
        "Vegetables/Carrots": 1,
        "Packages/Milk/Arla-Standard-Milk": 2,
        "Ready-To-Eat/Instant-Noodles": 3,
        "Snacks/Chocolate-Bar": 4,
    }
    synth = ProgrammaticCheckoutSynthesizer(Path("./dataset/GroceryStoreDataset/dataset/train"), Path("./synthetic_dataset"), cmap)
    synth.generate_split("train", total_scenes=200)
    synth.generate_split("val", total_scenes=40)
    ypath = synth.write_yaml_config()
    
    # Initialize YOLO Network Training Optimization Model
    model = YOLO("yolo11n.pt")
    model.train(data=str(ypath), epochs=20, imgsz=640, device="mps", workers=2, name="train", exist_ok=True)