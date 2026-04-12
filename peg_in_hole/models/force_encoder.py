"""
1-D CNN encoder for the force/torque temporal history.

Input  : (B, 5, 6)   — time × [fx, fy, fz, tx, ty, tz] (world frame)
Output : (B, 128)

The fy channel (insertion axis) dominates during phase 1.
Temporal conv captures the contact ramp-up signature.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ForceEncoder(nn.Module):
    """1-D temporal CNN for F/T history."""

    def __init__(self, in_channels: int = 6, time_steps: int = 5, out_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            # Conv1d expects (B, C, T)
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * time_steps, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 5, 6)  — time-major

        Returns
        -------
        (B, 128)
        """
        # Reshape to (B, 6, 5) — channels first, time last for Conv1d
        h = x.permute(0, 2, 1)
        h = self.conv(h)          # (B, 64, 5)
        h = h.reshape(h.size(0), -1)
        return self.fc(h)
