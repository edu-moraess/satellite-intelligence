"""Unit tests for change detection."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from src.change_detection.detector import detect_change


def test_identical():
    a = np.ones((10, 10), dtype=np.float32)
    r = detect_change(a, a)
    assert r.valid
    assert r.magnitude_mean < 1e-6


def test_shape_mismatch():
    a = np.ones((10, 10))
    b = np.ones((8, 8))
    r = detect_change(a, b)
    assert not r.valid


def test_change_detected():
    a = np.zeros((20, 20), dtype=np.float32)
    b = np.ones((20, 20), dtype=np.float32)
    r = detect_change(a, b, threshold=0.5)
    assert r.valid
    assert r.changed_pct > 90
