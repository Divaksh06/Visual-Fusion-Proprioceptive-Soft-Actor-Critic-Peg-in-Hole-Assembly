"""
Asymmetric Twin Critics for SAC.

During **training** the critic receives privileged information
(true_offset = peg_tip - hole_center from MuJoCo ground truth).
The **actor** never sees this.

At test time: pass true_offset as zeros, or retrain the critic head without it.

Architecture
------------
Input (train) : concat([fused, prop, action, true_offset])  →  (B, 285)
Input (eval)  : concat([fused, prop, action])                →  (B, 282)
Output        : scalar Q-value
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _build_mlp(in_dim: int, hidden: int = 512, mid: int = 256) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, mid),
        nn.ReLU(inplace=True),
        nn.Linear(mid, 1),
    )


class TwinCritic(nn.Module):
    """Clipped double-Q with optional privileged offset input."""

    def __init__(
        self,
        fused_dim: int = 256,
        prop_dim: int = 20,
        action_dim: int = 6,
        offset_dim: int = 3,
    ):
        super().__init__()
        self.offset_dim = offset_dim
        base_in = fused_dim + prop_dim + action_dim  # 282
        priv_in = base_in + offset_dim                # 285

        self.q1 = _build_mlp(priv_in)
        self.q2 = _build_mlp(priv_in)

    def forward(
        self,
        fused: torch.Tensor,
        proprioception: torch.Tensor,
        action: torch.Tensor,
        true_offset: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        fused : (B, 256)
        proprioception : (B, 20)
        action : (B, 6)
        true_offset : (B, 3) or None  — privileged info

        Returns
        -------
        q1, q2 : (B, 1) each
        """
        if true_offset is None:
            true_offset = torch.zeros(
                fused.size(0), self.offset_dim, device=fused.device
            )
        x = torch.cat([fused, proprioception, action, true_offset], dim=-1)
        return self.q1(x), self.q2(x)
