"""
U-Net architecture for semantic segmentation of satellite imagery.

Classes:
    0 Background
    1 Vegetation
    2 Water
    3 Urban
    4 Bare Soil
    5 Agriculture
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


NUM_CLASSES = 6
CLASS_NAMES = [
    "Background",
    "Vegetation",
    "Water",
    "Urban",
    "Bare Soil",
    "Agriculture",
]


class DoubleConv(nn.Module):
    """(Conv → BN → ReLU) × 2"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # pad if needed
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = nn.functional.pad(
            x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2]
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Classic U-Net for multi-class semantic segmentation.

    Input:  (B, C, H, W) – typically C=3 or C=4 (RGB or RGB+NIR)
    Output: (B, num_classes, H, W) logits
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = NUM_CLASSES,
        base_filters: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.inc = DoubleConv(in_channels, base_filters)
        self.down1 = Down(base_filters, base_filters * 2)
        self.down2 = Down(base_filters * 2, base_filters * 4)
        self.down3 = Down(base_filters * 4, base_filters * 8)
        self.down4 = Down(base_filters * 8, base_filters * 16)

        self.up1 = Up(base_filters * 16, base_filters * 8)
        self.up2 = Up(base_filters * 8, base_filters * 4)
        self.up3 = Up(base_filters * 4, base_filters * 2)
        self.up4 = Up(base_filters * 2, base_filters)
        self.outc = nn.Conv2d(base_filters, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


def build_unet(
    in_channels: int = 3,
    num_classes: int = NUM_CLASSES,
    base_filters: int = 32,
) -> UNet:
    """Factory helper."""
    return UNet(in_channels=in_channels, num_classes=num_classes, base_filters=base_filters)
