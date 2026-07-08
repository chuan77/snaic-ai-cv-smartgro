"""Standalone DINOv2 wrapper, plus free functions (load_backbone/extract_features) used by the
Day 3 recognition pipeline. src/data/gallery.py still extracts embeddings with its own inline
copy of this same backbone/transform setup for the Day 1 gallery."""
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


def default_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def build_transform() -> T.Compose:
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_backbone(
    device: torch.device | None = None, model_name: str = "dinov2_vits14"
) -> torch.nn.Module:
    device = device if device is not None else default_device()
    return torch.hub.load("facebookresearch/dinov2", model_name).to(device).eval()


def extract_features(model: torch.nn.Module, image_batches: list[torch.Tensor]) -> np.ndarray:
    """Runs model over pre-stacked (B,C,H,W) batches under no_grad, returns concatenated (N,...) array."""
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = default_device()

    outputs = []
    with torch.no_grad():
        for batch in image_batches:
            outputs.append(model(batch.to(device)).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def extract_features_from_paths(
    paths: list[str],
    model: torch.nn.Module,
    batch_size: int = 16,
    transform: T.Compose | None = None,
) -> np.ndarray:
    """Loads+transforms+stacks paths in batches, then extract_features per batch (= feats_of)."""
    transform = transform if transform is not None else build_transform()
    batches = []
    for i in range(0, len(paths), batch_size):
        chunk = paths[i:i + batch_size]
        batches.append(torch.stack([transform(Image.open(p).convert("RGB")) for p in chunk]))
    return extract_features(model, batches)


class DinoFeatureExtractor:
    """Leverages a frozen foundation model backbone to extract feature representations."""
    def __init__(self):
        self.device = default_device()
        self.model = load_backbone(self.device)
        self.transform = build_transform()

    def extract(self, pil_image: Image.Image) -> np.ndarray:
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.model(tensor).squeeze().cpu().numpy()
