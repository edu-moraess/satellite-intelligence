"""Segmentation metrics (IoU, accuracy) for offline evaluation."""

from __future__ import annotations

from typing import Dict

import numpy as np


def pixel_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    mask = target >= 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(pred[mask] == target[mask]))


def mean_iou(pred: np.ndarray, target: np.ndarray, num_classes: int = 6) -> float:
    ious = []
    for c in range(num_classes):
        inter = np.logical_and(pred == c, target == c).sum()
        union = np.logical_or(pred == c, target == c).sum()
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0
