from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.models.dino_extractor import (
    build_transform,
    default_device,
    extract_features,
    extract_features_from_paths,
    load_backbone,
)


def test_default_device_selects_mps_when_available():
    with patch("src.models.dino_extractor.torch.backends.mps.is_available", return_value=True):
        assert default_device() == torch.device("mps")


def test_default_device_selects_cpu_when_mps_unavailable():
    with patch("src.models.dino_extractor.torch.backends.mps.is_available", return_value=False):
        assert default_device() == torch.device("cpu")


def test_build_transform_resizes_to_224_and_normalizes_channels():
    transform = build_transform()
    img = Image.new("RGB", (50, 80))

    tensor = transform(img)

    assert tensor.shape == (3, 224, 224)


def test_load_backbone_loads_dinov2_vits14_by_default():
    with patch("src.models.dino_extractor.torch.hub.load") as mock_load:
        load_backbone(device=torch.device("cpu"))

    mock_load.assert_called_once_with("facebookresearch/dinov2", "dinov2_vits14")


def test_load_backbone_uses_custom_model_name():
    with patch("src.models.dino_extractor.torch.hub.load") as mock_load:
        load_backbone(device=torch.device("cpu"), model_name="dinov2_vitb14")

    mock_load.assert_called_once_with("facebookresearch/dinov2", "dinov2_vitb14")


def test_load_backbone_moves_model_to_device_and_sets_eval():
    with patch("src.models.dino_extractor.torch.hub.load") as mock_load:
        model = load_backbone(device=torch.device("cpu"))

    mock_load.return_value.to.assert_called_once_with(torch.device("cpu"))
    mock_load.return_value.to.return_value.eval.assert_called_once()
    assert model is mock_load.return_value.to.return_value.eval.return_value


def test_load_backbone_defaults_device_to_default_device():
    with patch("src.models.dino_extractor.torch.hub.load") as mock_load, \
         patch("src.models.dino_extractor.default_device", return_value=torch.device("cpu")):
        load_backbone()

    mock_load.return_value.to.assert_called_once_with(torch.device("cpu"))


def test_extract_features_concatenates_batches_in_order():
    model = nn.Identity()
    batch_a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    batch_b = torch.tensor([[5.0, 6.0]])

    result = extract_features(model, [batch_a, batch_b])

    np.testing.assert_array_equal(result, np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))


def test_extract_features_from_paths_returns_one_row_per_image(tmp_path):
    paths = []
    for i in range(3):
        p = tmp_path / f"img{i}.png"
        Image.new("RGB", (10, 10), color=(i * 10, i * 10, i * 10)).save(p)
        paths.append(str(p))
    model = nn.Flatten(start_dim=1)

    result = extract_features_from_paths(paths, model, batch_size=2)

    assert result.shape[0] == 3


def test_extract_features_from_paths_matches_manual_batching(tmp_path):
    paths = []
    for i in range(5):
        p = tmp_path / f"img{i}.png"
        Image.new("RGB", (10, 10), color=(i * 10, i * 10, i * 10)).save(p)
        paths.append(str(p))
    model = nn.Flatten(start_dim=1)

    result = extract_features_from_paths(paths, model, batch_size=2)

    transform = build_transform()
    expected = torch.stack([transform(Image.open(p).convert("RGB")) for p in paths])
    expected = model(expected).numpy()
    np.testing.assert_array_almost_equal(result, expected)
