"""
Learned soft-gating module — fuses visual and force features conditioned
on the current task phase.

Uses **Sigmoid** (NOT Softmax).
Sigmoid lets both gates suppress simultaneously (e.g., camera occluded AND
sensor noisy).  Softmax forces zero-sum competition which is undesirable.

Expected learned behaviour
--------------------------
Phase 0 (approach) : g_v ~ 0.7–0.9,  g_f ~ 0.1–0.3
Phase 1 (insertion): g_v ~ 0.2–0.4,  g_f ~ 0.6–0.9
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GatingModule(nn.Module):
    """Phase-conditioned soft gating of visual and force modalities."""

    def __init__(
        self,
        visual_dim: int = 256,
        force_dim: int = 128,
        fused_dim: int = 256,
    ):
        super().__init__()
        gate_in = visual_dim + force_dim + 1  # +1 for phase_bit → 385
        self.gate_net = nn.Sequential(
            nn.Linear(gate_in, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
            nn.Sigmoid(),   # ← NOT Softmax
        )

        self.v_proj = nn.Linear(visual_dim, fused_dim)
        self.f_proj = nn.Linear(force_dim, fused_dim)

    def forward(
        self,
        visual_feat: torch.Tensor,
        force_feat: torch.Tensor,
        phase_bit: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        visual_feat : (B, 256)
        force_feat  : (B, 128)
        phase_bit   : (B, 1)

        Returns
        -------
        fused  : (B, 256)
        g_v    : (B, 1) visual gate value  — for logging
        g_f    : (B, 1) force gate value   — for logging
        """
        g_in = torch.cat([visual_feat, force_feat, phase_bit], dim=-1)  # (B, 385)
        gates = self.gate_net(g_in)  # (B, 2)
        g_v = gates[:, 0:1]          # (B, 1)
        g_f = gates[:, 1:2]          # (B, 1)

        v_proj = self.v_proj(visual_feat)  # (B, 256)
        f_proj = self.f_proj(force_feat)   # (B, 256)

        fused = g_v * v_proj + g_f * f_proj  # (B, 256)
        return fused, g_v, g_f
