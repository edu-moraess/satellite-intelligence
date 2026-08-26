"""
Satellite Dataset stub for offline training.

Training is intentionally kept outside the Streamlit application.
This module provides a minimal torch Dataset interface for future use.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    Dataset = object  # type: ignore


class SatelliteSegmentationDataset(Dataset):
    """
    Minimal dataset for image / mask pairs.

    Parameters
    ----------
    image_paths : list of paths to multi-band arrays or files
    mask_paths  : list of paths to label masks
    transform   : optional callable applied to (image, mask)
    """

    def __init__(
        self,
        image_paths: list,
        mask_paths: list,
        transform: Optional[Callable] = None,
    ):
        assert len(image_paths) == len(mask_paths)
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple:
        # Placeholder – real implementation loads from disk / STAC
        raise NotImplementedError(
            "Dataset loading is reserved for offline training scripts. "
            "Do not call this from the Streamlit application."
        )
