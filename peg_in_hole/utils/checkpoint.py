"""
Save / load model checkpoints.
"""

from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(
    sac_agent,
    path: str | Path,
    step: int,
    stage_id: int,
    extra: dict | None = None,
):
    """Save all model weights + metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "stage_id": stage_id,
        "sac": sac_agent.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    print(f"[checkpoint] saved → {path}")


def load_checkpoint(path: str | Path, sac_agent, device: torch.device) -> dict:
    """Load checkpoint and restore model weights. Returns metadata dict."""
    payload = torch.load(path, map_location=device, weights_only=False)
    sac_agent.load_state_dict(payload["sac"])
    print(f"[checkpoint] loaded ← {path}  (step={payload.get('step', '?')})")
    return payload
