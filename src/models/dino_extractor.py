"""Standalone DINOv2 wrapper. Not yet wired into any entrypoint — src/data/gallery.py currently
extracts embeddings with its own inline copy of this same backbone/transform setup."""
import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as T
from typing import Dict, List, Any

class DinoFeatureExtractor:
    """Leverages a frozen foundation model backbone to extract feature representations."""
    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(self.device).eval()
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract(self, pil_image: Image.Image) -> np.ndarray:
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.model(tensor).squeeze().cpu().numpy()