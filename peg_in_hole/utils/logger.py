"""
Logging utility — WandB with TensorBoard fallback.
"""

from __future__ import annotations

from pathlib import Path

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


class Logger:
    """Simple WandB / console logger."""

    def __init__(self, cfg: dict, enabled: bool = True):
        self.enabled = enabled and _WANDB_AVAILABLE
        if self.enabled:
            wandb.init(
                project=cfg.get("wandb_project", "peg_in_hole"),
                entity=cfg.get("wandb_entity", None),
                config=cfg,
            )

    def log(self, data: dict, step: int | None = None):
        if self.enabled:
            wandb.log(data, step=step)
        else:
            # Console fallback
            items = " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in data.items())
            print(f"[step {step}] {items}")

    def finish(self):
        if self.enabled:
            wandb.finish()
