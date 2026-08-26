"""Unit tests for inference (no checkpoint → graceful)."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from src.deep_learning.inference.segmentation import run_segmentation, ModelStatus


def test_no_checkpoint():
    img = np.random.rand(3, 32, 32).astype(np.float32)
    result = run_segmentation(img)
    assert result.model_available is False
    assert result.status == ModelStatus.MODEL_UNAVAILABLE
    assert "not available" in result.message.lower() or "checkpoint" in result.message.lower()
