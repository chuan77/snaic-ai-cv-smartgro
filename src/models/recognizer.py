"""Linear classification head trained on frozen DINOv2 features (Day 3 recognition)."""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.models.dino_extractor import default_device


class LinearHead:
    """Thin torch.nn.Linear wrapper trained with cross-entropy + Adam on frozen features."""

    def __init__(
        self, in_dim: int, n_classes: int, device: torch.device | None = None, seed: int = 0
    ):
        self.in_dim = in_dim
        self.n_classes = n_classes
        self.device = device if device is not None else default_device()
        self.seed = seed
        torch.manual_seed(seed)
        self.linear = nn.Linear(in_dim, n_classes).to(self.device)
        self.train_accuracy_: float | None = None

    # lr=1e-3 (a common default elsewhere) leaves this linear probe under-converged within
    # epochs=200 on raw (unnormalized) DINOv2 features — 1e-2 reliably converges in that budget.
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 200, lr: float = 1e-2) -> "LinearHead":
        torch.manual_seed(self.seed)
        X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.as_tensor(y, dtype=torch.long, device=self.device)

        optimizer = torch.optim.Adam(self.linear.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        self.linear.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            logits = self.linear(X_t)
            loss = loss_fn(logits, y_t)
            loss.backward()
            optimizer.step()

        self.linear.eval()
        self.train_accuracy_ = float((self.predict(X) == y).mean())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.linear(X_t)
        return logits.argmax(dim=1).cpu().numpy()

    def save(self, path: Path) -> None:
        torch.save({
            "state_dict": self.linear.state_dict(),
            "in_dim": self.in_dim,
            "n_classes": self.n_classes,
            "seed": self.seed,
        }, path)

    @classmethod
    def load(cls, path: Path, device: torch.device | None = None) -> "LinearHead":
        checkpoint = torch.load(path, map_location="cpu")
        head = cls(
            in_dim=checkpoint["in_dim"],
            n_classes=checkpoint["n_classes"],
            device=device,
            seed=checkpoint["seed"],
        )
        head.linear.load_state_dict(checkpoint["state_dict"])
        head.linear.eval()
        return head
