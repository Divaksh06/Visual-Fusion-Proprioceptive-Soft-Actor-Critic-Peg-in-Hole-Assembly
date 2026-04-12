#!/usr/bin/env python3
"""
Evaluation script for a trained peg-in-hole agent.

Usage
-----
    python test.py --checkpoint checkpoints/best.pt --render --episodes 50
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import yaml

from peg_in_hole.envs.peg_hole_env import PegInHoleEnv
from peg_in_hole.algorithms.sac import SAC
from peg_in_hole.utils.checkpoint import load_checkpoint
from peg_in_hole.utils.normalizer import RunningNormalizer


def parse_args():
    parser = argparse.ArgumentParser(description="Test peg-in-hole agent")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    render_mode = "human" if args.render else None
    env = PegInHoleEnv(cfg, render_mode=render_mode)

    sac = SAC(cfg, device)
    meta = load_checkpoint(args.checkpoint, sac, device)
    print(f"[test] Loaded checkpoint from step {meta.get('step', '?')}, "
          f"stage {meta.get('stage_id', '?')}")

    # Normalisers (in a real deployment, load saved stats; here re-init)
    prop_normalizer = RunningNormalizer(shape=(20,), exclude_last=True)
    ft_normalizer = RunningNormalizer(shape=(5, 6))

    # ---- Run episodes ----
    results = {
        "insertion_depth_mm": [],
        "success": [],
        "max_lateral_force_N": [],
        "gate_traces": [],
        "phase_switch_steps": [],
    }

    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        gate_trace = []
        max_lat_force = 0.0
        phase_switch_step = None
        prev_phase = 0

        while not done:
            # Normalise
            prop_normalizer.update(obs["proprioception"][np.newaxis])
            ft_normalizer.update(obs["ft_history"][np.newaxis])
            obs_norm = {
                "rgb_d": obs["rgb_d"],
                "ft_history": ft_normalizer.normalize(obs["ft_history"][np.newaxis])[0],
                "proprioception": prop_normalizer.normalize(obs["proprioception"][np.newaxis])[0],
            }

            action, gate_info = sac.select_action(obs_norm, deterministic=True)
            gate_trace.append((gate_info["g_v"], gate_info["g_f"]))

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            lat_f = info.get("lateral_force", 0.0)
            if lat_f > max_lat_force:
                max_lat_force = lat_f

            if info["phase"] == 1 and prev_phase == 0 and phase_switch_step is None:
                phase_switch_step = info["step"]
            prev_phase = info["phase"]

            if args.render:
                env.render()

        depth_mm = info.get("insertion_depth", 0.0) * 1000
        success = info.get("is_success", False)

        results["insertion_depth_mm"].append(depth_mm)
        results["success"].append(success)
        results["max_lateral_force_N"].append(max_lat_force)
        results["gate_traces"].append(gate_trace)
        results["phase_switch_steps"].append(phase_switch_step)

        print(f"  Episode {ep+1:3d} | depth={depth_mm:6.2f} mm | "
              f"success={success} | max_lat_force={max_lat_force:.2f} N | "
              f"phase_switch={phase_switch_step}")

    # ---- Aggregates ----
    depths = np.array(results["insertion_depth_mm"])
    successes = np.array(results["success"], dtype=float)
    forces = np.array(results["max_lateral_force_N"])

    print("\n" + "=" * 60)
    print(f"  Success rate       : {successes.mean()*100:.1f} %")
    print(f"  Insertion depth    : {depths.mean():.2f} ± {depths.std():.2f} mm")
    print(f"  Max lateral force  : {forces.mean():.2f} ± {forces.std():.2f} N")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
