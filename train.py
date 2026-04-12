#!/usr/bin/env python3
"""
Main training script for Peg-in-Hole RL.

Usage
-----
    python train.py --config config/default.yaml --cuda
    python train.py --config config/default.yaml --cuda --render
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from peg_in_hole.envs.peg_hole_env import PegInHoleEnv
from peg_in_hole.algorithms.sac import SAC
from peg_in_hole.algorithms.her_buffer import HERReplayBuffer
from peg_in_hole.algorithms.curriculum_sac import CurriculumSAC
from peg_in_hole.utils.logger import Logger
from peg_in_hole.utils.normalizer import RunningNormalizer
from peg_in_hole.utils.checkpoint import save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train peg-in-hole RL agent")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--render", action="store_true", help="Render during training")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def verify_fk(env: PegInHoleEnv):
    """
    Verification step — confirm PyRoKi FK matches MuJoCo peg_tip site
    at a known configuration.
    """
    import mujoco

    q_test = np.zeros(6)
    pyroki_pos, _ = env.heal_ik.fk(q_test)

    env.mj_data.qpos[:6] = q_test
    mujoco.mj_forward(env.mj_model, env.mj_data)
    mujoco_pos = env.mj_data.site("peg_tip").xpos.copy()

    diff = np.linalg.norm(pyroki_pos - mujoco_pos)
    print(f"[verify_fk] PyRoKi pos = {pyroki_pos}")
    print(f"[verify_fk] MuJoCo pos = {mujoco_pos}")
    print(f"[verify_fk] Difference = {diff*1000:.2f} mm")
    assert diff < 1e-3, (
        f"FK mismatch! pyroki={pyroki_pos}, mujoco={mujoco_pos}, diff={diff:.6f}m. "
        f"Check joint_5 sign convention between URDF and XML."
    )
    print("[verify_fk] PASSED — FK within 1 mm")


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"[train] device = {device}")

    # Environment
    render_mode = "human" if args.render else None
    env = PegInHoleEnv(cfg, render_mode=render_mode)

    # ----- Step 1: Verify FK -----
    print("\n=== Step 1: FK Verification ===")
    verify_fk(env)

    # ----- Normalisers -----
    prop_normalizer = RunningNormalizer(shape=(20,), exclude_last=True)
    ft_normalizer = RunningNormalizer(shape=(5, 6))

    # ----- SAC Agent -----
    sac = SAC(cfg, device)

    # ----- HER Buffer -----
    buffer = HERReplayBuffer(
        capacity=cfg.get("buffer_capacity", 500_000),
        her_k=cfg.get("her_k", 4),
        device=device,
    )

    # ----- Curriculum -----
    curriculum_sac = CurriculumSAC(sac, env.curriculum)

    # ----- Logger -----
    logger = Logger(cfg, enabled=True)

    # ----- Training loop -----
    total_steps = cfg.get("total_steps", 2_000_000)
    warmup_steps = cfg.get("warmup_steps", 5_000)
    eval_every = cfg.get("eval_every", 10_000)
    save_every = cfg.get("save_every", 50_000)
    batch_size = cfg.get("batch_size", 256)

    global_step = 0
    episode_count = 0
    best_success_rate = 0.0

    print(f"\n=== Starting Training — {total_steps} steps ===\n")

    while global_step < total_steps:
        obs, info = env.reset()
        buffer.start_episode()

        episode_reward = 0.0
        episode_gate_trace = []
        phase_switch_step = None
        prev_phase = 0
        done = False

        while not done:
            # Update normalisers
            prop_normalizer.update(obs["proprioception"][np.newaxis])
            ft_normalizer.update(obs["ft_history"][np.newaxis])

            # Normalise obs for policy
            obs_norm = {
                "rgb_d": obs["rgb_d"],  # already [0,1]
                "ft_history": ft_normalizer.normalize(obs["ft_history"][np.newaxis])[0],
                "proprioception": prop_normalizer.normalize(obs["proprioception"][np.newaxis])[0],
            }

            # Action selection
            if global_step < warmup_steps:
                action = env.action_space.sample()
                gate_info = {"g_v": 0.5, "g_f": 0.5}
            else:
                action, gate_info = sac.select_action(obs_norm, deterministic=False)

            episode_gate_trace.append((gate_info["g_v"], gate_info["g_f"]))

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Track phase switch
            if info["phase"] == 1 and prev_phase == 0 and phase_switch_step is None:
                phase_switch_step = info["step"]
            prev_phase = info["phase"]

            # Privileged info for critic
            true_offset = info["peg_tip_world"] - info["hole_center_world"]

            # Store transition
            ft_force_norm = np.linalg.norm(info.get("ft_force_world", np.zeros(3)))
            buffer.add_transition({
                "obs": obs,
                "next_obs": next_obs,
                "action": action,
                "reward": reward,
                "done": terminated,
                "true_offset": true_offset,
                "achieved_goal": info["achieved_goal"],
                "desired_goal": info["desired_goal"],
                "ft_force_norm": ft_force_norm,
            })

            episode_reward += reward
            obs = next_obs
            global_step += 1

            # ----- SAC update -----
            if global_step >= warmup_steps and len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                update_info = sac.update(batch)

                if global_step % 1000 == 0:
                    logger.log(update_info, step=global_step)

            # ----- Periodic eval logging -----
            if global_step % eval_every == 0 and global_step > 0:
                logger.log({
                    "global_step": global_step,
                    "stage_id": curriculum_sac.current_stage_id(),
                    "stage_name": curriculum_sac.current_stage_name(),
                }, step=global_step)

            # ----- Save checkpoint -----
            if global_step % save_every == 0 and global_step > 0:
                save_checkpoint(
                    sac,
                    f"checkpoints/step_{global_step}.pt",
                    step=global_step,
                    stage_id=curriculum_sac.current_stage_id(),
                )

        # ---- End of episode ----
        buffer.end_episode()
        episode_count += 1

        is_success = info.get("is_success", False)
        stage_info = curriculum_sac.report_episode(is_success)

        logger.log({
            "episode": episode_count,
            "episode_reward": episode_reward,
            "insertion_depth_mm": info.get("insertion_depth", 0.0) * 1000,
            "is_success": float(is_success),
            "max_lateral_force": info.get("lateral_force", 0.0),
            "stage_id": stage_info["stage_id"],
            "success_rate": stage_info["success_rate"],
            "phase_switch_step": phase_switch_step if phase_switch_step else -1,
        }, step=global_step)

        # Log gate values
        if len(episode_gate_trace) > 0:
            g_vs, g_fs = zip(*episode_gate_trace)
            logger.log({
                "gate_v_mean": np.mean(g_vs),
                "gate_f_mean": np.mean(g_fs),
            }, step=global_step)

        if stage_info["stage_changed"]:
            print(f"[curriculum] Stage changed → {stage_info['stage_name']} "
                  f"(id={stage_info['stage_id']})")

        # Save best
        if stage_info["success_rate"] > best_success_rate:
            best_success_rate = stage_info["success_rate"]
            save_checkpoint(
                sac,
                f"checkpoints/best.pt",
                step=global_step,
                stage_id=stage_info["stage_id"],
            )

        if args.render:
            env.render()

    # Final save
    save_checkpoint(sac, "checkpoints/final.pt", step=global_step,
                    stage_id=curriculum_sac.current_stage_id())
    logger.finish()
    env.close()
    print(f"\n[train] Done. {episode_count} episodes, {global_step} steps.")


if __name__ == "__main__":
    main()
