"""
Inference pipeline for U-Net semantic segmentation.

Loads checkpoint when available; otherwise reports model unavailable.
Never invents weights or random segmentations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..models.unet import CLASS_NAMES, NUM_CLASSES, UNet, build_unet

CHECKPOINT_DIR = Path(__file__).resolve().parents[3] / "models" / "checkpoints"
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "unet_sentinel2.pt"


@dataclass
class SegmentationResult:
    """Structured output of the segmentation pipeline."""

    segmentation: np.ndarray          # (H, W) int labels
    class_percentages: Dict[str, float]
    confidence: Optional[np.ndarray]  # (H, W) max softmax probability
    class_names: List[str]
    model_available: bool
    message: str

    def to_dict(self) -> dict:
        return {
            "class_percentages": self.class_percentages,
            "model_available": self.model_available,
            "message": self.message,
            "class_names": self.class_names,
        }


def _find_checkpoint() -> Optional[Path]:
    if DEFAULT_CHECKPOINT.exists():
        return DEFAULT_CHECKPOINT
    # any .pt / .pth in checkpoints
    if CHECKPOINT_DIR.exists():
        for p in CHECKPOINT_DIR.glob("*.pt"):
            return p
        for p in CHECKPOINT_DIR.glob("*.pth"):
            return p
    return None


@torch.no_grad()
def run_segmentation(
    image: np.ndarray,
    device: Optional[str] = None,
) -> SegmentationResult:
    """
    Run U-Net inference on a multi-band image array.

    Parameters
    ----------
    image : np.ndarray
        Shape (C, H, W) or (H, W, C). Expected C=3 (RGB) or C=4.
        Values should be roughly [0, 1] or [0, 10000] (will be normalized).
    """
    ckpt_path = _find_checkpoint()
    if ckpt_path is None:
        return SegmentationResult(
            segmentation=np.zeros((1, 1), dtype=np.uint8),
            class_percentages={name: 0.0 for name in CLASS_NAMES},
            confidence=None,
            class_names=CLASS_NAMES,
            model_available=False,
            message="Deep Learning model not available. No checkpoint found in models/checkpoints/.",
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare tensor
    arr = image.astype(np.float32)
    if arr.ndim == 3 and arr.shape[0] not in (3, 4) and arr.shape[-1] in (3, 4):
        arr = np.transpose(arr, (2, 0, 1))  # HWC → CHW
    if arr.ndim != 3:
        return SegmentationResult(
            segmentation=np.zeros((1, 1), dtype=np.uint8),
            class_percentages={name: 0.0 for name in CLASS_NAMES},
            confidence=None,
            class_names=CLASS_NAMES,
            model_available=False,
            message="Invalid image shape for inference.",
        )

    # Simple percentile normalization to [0, 1]
    for c in range(arr.shape[0]):
        band = arr[c]
        valid = band[np.isfinite(band)]
        if valid.size > 0:
            p2, p98 = np.percentile(valid, (2, 98))
            if p98 > p2:
                arr[c] = np.clip((band - p2) / (p98 - p2), 0, 1)
            else:
                arr[c] = 0.0
        arr[c] = np.nan_to_num(arr[c], nan=0.0)

    in_channels = arr.shape[0]
    model = build_unet(in_channels=in_channels, num_classes=NUM_CLASSES)
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
    except Exception as e:
        return SegmentationResult(
            segmentation=np.zeros((1, 1), dtype=np.uint8),
            class_percentages={name: 0.0 for name in CLASS_NAMES},
            confidence=None,
            class_names=CLASS_NAMES,
            model_available=False,
            message=f"Failed to load checkpoint: {e}",
        )

    model.to(device)
    model.eval()

    tensor = torch.from_numpy(arr).unsqueeze(0).to(device)  # (1, C, H, W)
    # ensure size divisible by 16
    _, _, h, w = tensor.shape
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h))

    logits = model(tensor)
    probs = F.softmax(logits, dim=1)
    conf, pred = torch.max(probs, dim=1)

    # unpad
    pred = pred[0, :h, :w].cpu().numpy().astype(np.uint8)
    conf = conf[0, :h, :w].cpu().numpy().astype(np.float32)

    # percentages
    total = pred.size
    percentages = {}
    for i, name in enumerate(CLASS_NAMES):
        percentages[name] = float(100.0 * np.sum(pred == i) / total) if total > 0 else 0.0

    return SegmentationResult(
        segmentation=pred,
        class_percentages=percentages,
        confidence=conf,
        class_names=CLASS_NAMES,
        model_available=True,
        message="Segmentation completed successfully.",
    )
