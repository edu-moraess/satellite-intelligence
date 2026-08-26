"""Change detection spatial alignment tests."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from src.change_detection.detector import detect_change


def test_shape_mismatch_rejected():
    a = np.ones((10, 10), dtype=np.float32)
    b = np.ones((8, 8), dtype=np.float32)
    r = detect_change(a, b)
    assert not r.valid
    assert "not spatially aligned" in r.message.lower() or "incompatible" in r.message.lower()


def test_crs_mismatch_rejected():
    a = np.ones((10, 10), dtype=np.float32)
    b = np.ones((10, 10), dtype=np.float32)
    r = detect_change(
        a, b,
        meta_t1={"crs": "EPSG:4326", "transform": (1, 0, 0, 0, -1, 0)},
        meta_t2={"crs": "EPSG:32723", "transform": (1, 0, 0, 0, -1, 0)},
    )
    assert not r.valid
    assert "aligned" in r.message.lower()


def test_aligned_succeeds():
    a = np.ones((20, 20), dtype=np.float32) * 0.2
    b = a.copy()
    b[5:10, 5:10] = 0.9
    r = detect_change(a, b, threshold=0.3)
    assert r.valid
    assert r.changed_pct > 0
