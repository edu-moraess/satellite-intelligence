"""Training configuration for U-Net semantic segmentation (V2)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class TrainConfig:
    """Serializable training hyperparameters."""

    dataset_name: str = "loveda"
    dataset_root: str = "data/datasets/LoveDA"
    input_channels: int = 3
    image_size: int = 256
    num_classes: int = 6
    class_names: List[str] = field(
        default_factory=lambda: [
            "Background",
            "Vegetation",
            "Water",
            "Urban",
            "Bare Soil",
            "Agriculture",
        ]
    )

    base_filters: int = 32

    epochs: int = 30
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 2

    use_dice: bool = True
    dice_weight: float = 0.5
    ce_weight: float = 0.5

    checkpoint_dir: str = "models/checkpoints"
    checkpoint_name: str = "unet_v2_best.pt"
    early_stopping_patience: int = 8

    training_domain: str = "Aerial imagery (LoveDA)"
    target_domain: str = "Sentinel-2 L2A RGB"
    domain_gap: str = (
        "Model is trained on high-resolution aerial RGB (LoveDA). "
        "Inference on Sentinel-2 10 m RGB is experimental due to domain gap "
        "(resolution, spectral response, atmosphere)."
    )

    def to_dict(self) -> Dict:
        return asdict(self)

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.checkpoint_dir) / self.checkpoint_name
