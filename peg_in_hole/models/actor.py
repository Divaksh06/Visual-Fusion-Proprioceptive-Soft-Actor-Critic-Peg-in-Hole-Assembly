"""
Phase-conditioned SAC actor.

Input  : fused (B, 256) + proprioception (B, 20)  → (B, 276)
Output : 6-D Cartesian delta action [dx, dy, dz, droll, dpitch, dyaw]

Translation clipped  : ±5 mm
Rotation clipped     : ±2 deg (0.035 rad)

Sampling: reparameterization trick, tanh squash, log-prob correction.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0

# Action limits (used for rescaling after tanh)
ACT_LIMIT_POS = 0.005    # metres
ACT_LIMIT_ROT = 0.035    # radians


class Actor(nn.Module):
    """Stochastic actor with tanh-squashed Gaussian."""

    def __init__(self, fused_dim: int = 256, prop_dim: int = 20, action_dim: int = 6):
        super().__init__()
        in_dim = fused_dim + prop_dim  # 276
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
        )
        self.mean_head = nn.Linear(256, action_dim)
        self.log_std_head = nn.Linear(256, action_dim)

        # Action scale: first 3 dims = position, last 3 = rotation
        self.register_buffer(
            "action_scale",
            torch.tensor(
                [ACT_LIMIT_POS] * 3 + [ACT_LIMIT_ROT] * 3, dtype=torch.float32
            ),
        )

    def forward(
        self,
        fused: torch.Tensor,
        proprioception: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        fused : (B, 256)
        proprioception : (B, 20)
        deterministic : if True, return mean action (no sampling)

        Returns
        -------
        action : (B, 6) scaled and clipped
        log_prob : (B, 1) log probability of the sampled action
        """
        x = torch.cat([fused, proprioception], dim=-1)
        h = self.net(x)

        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()

        dist = Normal(mean, std)

        if deterministic:
            z = mean
        else:
            z = dist.rsample()  # reparameterization trick

        # Tanh squashing
        action_tanh = torch.tanh(z)

        # Log probability with tanh correction
        log_prob = dist.log_prob(z) - torch.log(1 - action_tanh.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)  # (B, 1)

        # Scale to physical action limits
        action = action_tanh * self.action_scale

        return action, log_prob
