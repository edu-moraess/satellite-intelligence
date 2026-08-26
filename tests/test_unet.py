"""Unit tests for U-Net (forward pass only, no weights)."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from src.deep_learning.models.unet import UNet, build_unet, NUM_CLASSES


def test_unet_forward():
    model = build_unet(in_channels=3, num_classes=NUM_CLASSES, base_filters=16)
    model.eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, NUM_CLASSES, 64, 64)


def test_unet_channels():
    model = build_unet(in_channels=4, num_classes=6)
    x = torch.randn(2, 4, 32, 32)
    y = model(x)
    assert y.shape == (2, 6, 32, 32)
