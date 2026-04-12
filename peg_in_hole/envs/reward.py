"""
Modular reward function for peg-in-hole insertion.

All distances and forces are in **world frame**.
Insertion axis is +Y: INSERTION_AXIS = [0, 1, 0].

CRITICAL: R_approach MUST be exactly 0.0 during phase 1.
Otherwise the agent learns to hover at the hole entrance.
"""

from __future__ import annotations

import numpy as np

INSERTION_AXIS = np.array([0.0, 1.0, 0.0])


def compute_reward(
    info: dict,
    prev_depth: float,
    phase: int,
    cfg: dict,
) -> float:
    """
    Compute the scalar reward for one environment step.

    Parameters
    ----------
    info : dict
        Must contain: peg_tip_world (3,), hole_center_world (3,),
        ft_force_world (3,), peg_y_axis_world (3,).
    prev_depth : float
        Insertion depth at the *previous* step (dot with +Y).
    phase : int
        0 = approach, 1 = insertion.
    cfg : dict
        Reward weight hyper-parameters (from default.yaml → reward_weights).

    Returns
    -------
    float
        Total reward.
    """
    # Unpack weights with defaults
    alpha = cfg.get("alpha", 5.0)
    beta = cfg.get("beta", 2.0)
    gamma = cfg.get("gamma", 10.0)
    delta = cfg.get("delta", 1.0)
    force_limit = cfg.get("force_limit", 5.0)
    success_bonus = cfg.get("success_bonus", 200.0)
    success_depth = cfg.get("success_depth", 0.015)
    epsilon = cfg.get("epsilon", 0.01)

    peg_tip = info["peg_tip_world"]
    hole_ctr = info["hole_center_world"]

    # Lateral error (XZ plane only — perpendicular to insertion axis +Y)
    lateral_err = np.linalg.norm((peg_tip - hole_ctr)[[0, 2]])

    # Insertion depth along +Y
    depth = np.dot(peg_tip - hole_ctr, INSERTION_AXIS)
    delta_depth = depth - prev_depth

    # Force perpendicular to insertion axis
    ft_world = info["ft_force_world"]
    lat_force = np.linalg.norm(ft_world[[0, 2]])

    # Peg body local Y axis projected onto world insertion axis
    peg_y_world = info["peg_y_axis_world"]
    axis_align = np.dot(peg_y_world, INSERTION_AXIS)

    # ---------- Component rewards ----------

    # R_approach: penalise lateral distance — ONLY in phase 0
    R_approach = -alpha * lateral_err if phase == 0 else 0.0

    # R_align: penalise misalignment — both phases
    R_align = -beta * (1.0 - axis_align)

    # R_insertion: reward forward progress — ONLY in phase 1
    R_insertion = gamma * max(0.0, delta_depth) if phase == 1 else 0.0

    # R_force: penalise excessive lateral force — ONLY in phase 1
    R_force = -delta * max(0.0, lat_force - force_limit) if phase == 1 else 0.0

    # R_success: large bonus for full insertion
    R_success = success_bonus if depth >= success_depth else 0.0

    # R_step: small constant penalty to encourage speed
    R_step = -epsilon

    return R_approach + R_align + R_insertion + R_force + R_success + R_step
