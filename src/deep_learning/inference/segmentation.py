"""
Inference pipeline for U-Net semantic segmentation.

Loads checkpoint when available; otherwise reports model unavailable.
Never invents weights or random segmentations.

Expected input channels: 3 (RGB: B04, B03, B02) or 4 (RGB+NIR).
Output classes: 6 (Background, Vegetation, Water, Urban, Bare Soil, Agriculture).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..models.unet import CLASS_NAMES, NUM_CLASSES, build_unet

CHECKPOINT_DIR = Path(__file__).resolve().parents[3] / "models" / "checkpoints"
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "unet_sentinel2.pt"

# Contract: U-Net expects 3 or 4 input channels (RGB or RGB+NIR)
EXPECTED_IN_CHANNELS = (3, 4)


class ModelStatus(str, Enum):
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    MODEL_LOAD_ERROR = "MODEL_LOAD_ERROR"
    MODEL_INFERENCE_ERROR = "MODEL_INFERENCE_ERROR"
    MODEL_READY = "MODEL_READY"
    SUCCESS = "SUCCESS"


@dataclass
class SegmentationResult:
    """Structured output of the segmentation pipeline."""

    segmentation: np.ndarray
    class_percentages: Dict[str, float]
    confidence: Optional[np.ndarray]
    class_names: List[str]
    model_available: bool
    message: str
    status: ModelStatus = ModelStatus.MODEL_UNAVAILABLE

    def to_dict(self) -> dict:
        return {
            "class_percentages": self.class_percentages,
            "model_available": self.model_available,
            "message": self.message,
            "class_names": self.class_names,
            "status": self.status.value,
        }


def _find_checkpoint() -> Optional[Path]:
    if DEFAULT_CHECKPOINT.exists() and DEFAULT_CHECKPOINT.stat().st_size > 0:
        return DEFAULT_CHECKPOINT
    if CHECKPOINT_DIR.exists():
        for pattern in ("*.pt", "*.pth"):
            for p in sorted(CHECKPOINT_DIR.glob(pattern)):
                if p.stat().st_size > 0:
                    return p
    return None


def _empty_result(status: ModelStatus, message: str) -> SegmentationResult:
    return SegmentationResult(
        segmentation=np.zeros((1, 1), dtype=np.uint8),
        class_percentages={name: 0.0 for name in CLASS_NAMES},
        confidence=None,
        class_names=CLASS_NAMES,
        model_available=False,
        message=message,
        status=status,
    )


def prepare_input(image: np.ndarray) -> np.ndarray:
    """
    Validate and normalize input to (C, H, W) float32 in [0, 1].
    Rejects NaN/Inf, empty arrays, wrong channel counts.
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Input image is empty or invalid.")

    arr = image.astype(np.float32)
    if arr.ndim == 3 and arr.shape[0] not in EXPECTED_IN_CHANNELS and arr.shape[-1] in EXPECTED_IN_CHANNELS:
        arr = np.transpose(arr, (2, 0, 1))
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array (C,H,W) or (H,W,C), got shape {arr.shape}")
    if arr.shape[0] not in EXPECTED_IN_CHANNELS:
        raise ValueError(
            f"U-Net input configuration is incompatible with available Sentinel-2 bands. "
            f"Expected {EXPECTED_IN_CHANNELS} channels, got {arr.shape[0]}."
        )
    if not np.isfinite(arr).any():
        raise ValueError("Input contains only NaN/Inf values.")

    for c in range(arr.shape[0]):
        band = arr[c]
        valid = band[np.isfinite(band)]
        if valid.size > 0:
            p2, p98 = np.percentile(valid, (2, 98))
            if p98 > p2:
                arr[c] = np.clip((band - p2) / (p98 - p2), 0, 1)
            else:
                arr[c] = 0.0
        arr[c] = np.nan_to_num(arr[c], nan=0.0, posinf=0.0, neginf=0.0)
    return arr


@torch.no_grad()
def run_segmentation(
    image: np.ndarray,
    device: Optional[str] = None,
) -> SegmentationResult:
    """
    Run U-Net inference on a multi-band image array.

    Without a real checkpoint returns MODEL_UNAVAILABLE — never fakes results.
    """
    ckpt_path = _find_checkpoint()
    if ckpt_path is None:
        return _empty_result(
            ModelStatus.MODEL_UNAVAILABLE,
            "Deep Learning model not available. "
            "U-Net checkpoint is not included in V1.",
        )

    try:
        arr = prepare_input(image)
    except ValueError as e:
        return _empty_result(ModelStatus.MODEL_INFERENCE_ERROR, str(e))

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    in_channels = arr.shape[0]
    model = build_unet(in_channels=in_channels, num_classes=NUM_CLASSES)

    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        elif isinstance(state, dict) and any(k.startswith("inc.") or k.startswith("outc.") for k in state):
            model.load_state_dict(state)
        else:
            model.load_state_dict(state)
    except Exception as e:
        return _empty_result(
            ModelStatus.MODEL_LOAD_ERROR,
            f"Failed to load checkpoint: {e}",
        )

    model.to(device)
    model.eval()

    try:
        tensor = torch.from_numpy(arr).unsqueeze(0).to(device)
        _, _, h, w = tensor.shape
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        if pad_h or pad_w:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h))

        logits = model(tensor)
        probs = F.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
        pred = pred[0, :h, :w].cpu().numpy().astype(np.uint8)
        conf = conf[0, :h, :w].cpu().numpy().astype(np.float32)

        if not np.isfinite(conf).all():
            return _empty_result(
                ModelStatus.MODEL_INFERENCE_ERROR,
                "Inference produced non-finite confidence values.",
            )

        total = pred.size
        percentages = {
            name: float(100.0 * np.sum(pred == i) / total) if total > 0 else 0.0
            for i, name in enumerate(CLASS_NAMES)
        }
        return SegmentationResult(
            segmentation=pred,
            class_percentages=percentages,
            confidence=conf,
            class_names=CLASS_NAMES,
            model_available=True,
            message="Segmentation completed successfully.",
            status=ModelStatus.SUCCESS,
        )
    except Exception as e:
        return _empty_result(
            ModelStatus.MODEL_INFERENCE_ERROR,
            f"Inference failed: {e}",
        )
