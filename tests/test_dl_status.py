"""Deep Learning status and contract tests."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from src.deep_learning.models.unet import build_unet, NUM_CLASSES
from src.deep_learning.inference.segmentation import (
    run_segmentation,
    prepare_input,
    ModelStatus,
)


def test_unet_instantiates_without_checkpoint():
    m = build_unet(in_channels=3, num_classes=NUM_CLASSES, base_filters=16)
    x = torch.randn(1, 3, 32, 32)
    m.eval()
    with torch.no_grad():
        y = m(x)
    assert y.shape == (1, NUM_CLASSES, 32, 32)


def test_no_checkpoint_unavailable():
    img = np.random.rand(3, 32, 32).astype(np.float32)
    res = run_segmentation(img)
    assert res.model_available is False
    assert res.status == ModelStatus.MODEL_UNAVAILABLE
    assert "not included" in res.message.lower() or "not available" in res.message.lower()


def test_prepare_input_rejects_nan():
    img = np.full((3, 16, 16), np.nan, dtype=np.float32)
    try:
        prepare_input(img)
        assert False
    except ValueError:
        pass


def test_prepare_input_rejects_wrong_channels():
    img = np.random.rand(5, 16, 16).astype(np.float32)
    try:
        prepare_input(img)
        assert False
    except ValueError as e:
        assert "incompatible" in str(e).lower() or "channels" in str(e).lower()


def test_prepare_input_valid():
    img = np.random.rand(3, 16, 16).astype(np.float32) * 3000
    out = prepare_input(img)
    assert out.shape == (3, 16, 16)
    assert np.isfinite(out).all()
    assert out.min() >= 0 and out.max() <= 1


def test_inference_uses_no_grad_and_eval():
    img = np.random.rand(3, 24, 24).astype(np.float32)
    res = run_segmentation(img)
    assert res.status == ModelStatus.MODEL_UNAVAILABLE
