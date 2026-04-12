"""
CNN encoder for RGB-D frames.

Input  : (B, 4, 64, 64)
Output : (B, 256)

Uses **LayerNorm**, NOT BatchNorm (SAC with off-policy replay
violates BatchNorm's i.i.d. mini-batch assumption).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VisualEncoder(nn.Module):
    """Encodes 4-channel RGB-D images into a 256-d feature vector."""

    def __init__(self, in_channels: int = 4, out_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=2),   # -> (16, 30, 30)
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),            # -> (32, 14, 14)
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),            # -> (64, 6, 6)
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),                          # -> (64, 4, 4)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 4, 64, 64) float32

        Returns
        -------
        (B, 256)
        """
        h = self.conv(x)
        h = h.reshape(h.size(0), -1)
        return self.fc(h)
