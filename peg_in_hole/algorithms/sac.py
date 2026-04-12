"""
Soft Actor-Critic (SAC) with automatic entropy tuning.

Key design choices:
* Gradient norm clipping = 1.0 on all parameters
* Soft target update τ = 0.005
* Updates every environment step (after warmup)
"""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from peg_in_hole.models import (
    VisualEncoder,
    ForceEncoder,
    GatingModule,
    Actor,
    TwinCritic,
)


class SAC:
    """SAC agent wrapping all model components."""

    def __init__(self, cfg: dict, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.gamma = cfg.get("gamma", 0.99)
        self.tau = cfg.get("tau", 0.005)
        self.target_entropy = cfg.get("target_entropy", -6)

        # ---- Networks ----
        self.visual_enc = VisualEncoder().to(device)
        self.force_enc = ForceEncoder().to(device)
        self.gating = GatingModule().to(device)
        self.actor = Actor().to(device)
        self.critic = TwinCritic().to(device)
        self.critic_target = deepcopy(self.critic).to(device)

        # Freeze target parameters
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # ---- Entropy coefficient (log α) ----
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp().item()

        # ---- Optimisers ----
        actor_lr = cfg.get("actor_lr", 3e-4)
        critic_lr = cfg.get("critic_lr", 3e-4)
        alpha_lr = cfg.get("alpha_lr", 3e-4)

        self.actor_opt = Adam(
            list(self.visual_enc.parameters())
            + list(self.force_enc.parameters())
            + list(self.gating.parameters())
            + list(self.actor.parameters()),
            lr=actor_lr,
        )
        self.critic_opt = Adam(self.critic.parameters(), lr=critic_lr)
        self.alpha_opt = Adam([self.log_alpha], lr=alpha_lr)

    # ------------------------------------------------------------------
    # Encode observations → features
    # ------------------------------------------------------------------
    def _encode(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (fused, proprioception, phase_bit)."""
        rgbd = obs["rgb_d"]                 # (B, 4, 64, 64)
        ft = obs["ft_history"]              # (B, 5, 6)
        prop = obs["proprioception"]        # (B, 20)

        vis_feat = self.visual_enc(rgbd)    # (B, 256)
        frc_feat = self.force_enc(ft)       # (B, 128)
        phase_bit = prop[:, -1:]            # (B, 1)

        fused, g_v, g_f = self.gating(vis_feat, frc_feat, phase_bit)
        return fused, prop, phase_bit

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    @torch.no_grad()
    def select_action(self, obs: dict, deterministic: bool = False) -> tuple[dict, dict]:
        """
        Select action for a single observation.

        Returns
        -------
        action_np : dict with 'action' key → (6,) numpy
        gate_info : dict with 'g_v', 'g_f' floats
        """
        obs_t = self._obs_to_tensor(obs, batch=True)
        fused, prop, _ = self._encode(obs_t)
        action, _ = self.actor(fused, prop, deterministic=deterministic)
        action_np = action.squeeze(0).cpu().numpy()

        # Gate values for logging
        rgbd = obs_t["rgb_d"]
        ft = obs_t["ft_history"]
        vis_feat = self.visual_enc(rgbd)
        frc_feat = self.force_enc(ft)
        phase_bit = prop[:, -1:]
        _, g_v, g_f = self.gating(vis_feat, frc_feat, phase_bit)

        return (
            action_np,
            {"g_v": g_v.item(), "g_f": g_f.item()},
        )

    # ------------------------------------------------------------------
    # Update step
    # ------------------------------------------------------------------
    def update(self, batch: dict) -> dict:
        """
        Run one SAC gradient step.

        Parameters
        ----------
        batch : dict of tensors on device, keys:
            obs, next_obs (each a dict of rgb_d, ft_history, proprioception),
            action (B, 6), reward (B, 1), done (B, 1), true_offset (B, 3)

        Returns
        -------
        log_dict : dict of scalar metrics
        """
        obs = batch["obs"]
        next_obs = batch["next_obs"]
        action = batch["action"]
        reward = batch["reward"]
        done = batch["done"]
        true_offset = batch.get("true_offset", None)

        # ---- Encode ----
        fused, prop, _ = self._encode(obs)
        with torch.no_grad():
            fused_next, prop_next, _ = self._encode(next_obs)

        # ---- Critic update ----
        with torch.no_grad():
            next_action, next_log_prob = self.actor(fused_next, prop_next)
            fused_next_tgt, prop_next_tgt, _ = self._encode(next_obs)
            q1_tgt, q2_tgt = self.critic_target(
                fused_next_tgt, prop_next_tgt, next_action, true_offset
            )
            q_tgt = torch.min(q1_tgt, q2_tgt) - self.alpha * next_log_prob
            target_q = reward + (1.0 - done) * self.gamma * q_tgt

        q1, q2 = self.critic(fused.detach(), prop.detach(), action, true_offset)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # ---- Actor update ----
        fused_a, prop_a, _ = self._encode(obs)
        new_action, log_prob = self.actor(fused_a, prop_a)
        q1_a, q2_a = self.critic(fused_a.detach(), prop_a.detach(), new_action, true_offset)
        q_a = torch.min(q1_a, q2_a)
        actor_loss = (self.alpha * log_prob - q_a).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.visual_enc.parameters())
            + list(self.force_enc.parameters())
            + list(self.gating.parameters())
            + list(self.actor.parameters()),
            1.0,
        )
        self.actor_opt.step()

        # ---- Alpha (entropy coeff) update ----
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        self.alpha = self.log_alpha.exp().item()

        # ---- Soft target update ----
        self._soft_update()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha,
            "q_mean": ((q1 + q2) / 2).mean().item(),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _soft_update(self):
        for p, p_tgt in zip(self.critic.parameters(), self.critic_target.parameters()):
            p_tgt.data.mul_(1 - self.tau)
            p_tgt.data.add_(self.tau * p.data)

    def _obs_to_tensor(self, obs: dict, batch: bool = False) -> dict:
        """Convert numpy obs dict to torch tensors on device, optionally add batch dim."""
        out = {}
        for k, v in obs.items():
            t = torch.as_tensor(v, dtype=torch.float32, device=self.device)
            if batch and t.dim() == len(v.shape):
                t = t.unsqueeze(0)
            out[k] = t
        return out

    def state_dict(self) -> dict:
        return {
            "visual_enc": self.visual_enc.state_dict(),
            "force_enc": self.force_enc.state_dict(),
            "gating": self.gating.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
        }

    def load_state_dict(self, sd: dict):
        self.visual_enc.load_state_dict(sd["visual_enc"])
        self.force_enc.load_state_dict(sd["force_enc"])
        self.gating.load_state_dict(sd["gating"])
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        self.critic_target.load_state_dict(sd["critic_target"])
        self.log_alpha = sd["log_alpha"].to(self.device).requires_grad_(True)
        self.alpha = self.log_alpha.exp().item()
