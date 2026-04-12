"""
Hindsight Experience Replay (HER) buffer — contact-aware variant.

Strategy : "future", k=4
Goal     : scalar insertion_depth (NOT xyz — xyz creates degenerate relabeling)

Contact-aware filter
--------------------
Only relabel transitions from episodes where at least one step had
norm(ft_force_world) > 0.3 N.  Episodes with zero contact produce
meaningless achieved goals.

Do NOT normalise rewards — this distorts the relabeled reward scale vs real
rewards.
"""

from __future__ import annotations

import numpy as np
import torch


class HERReplayBuffer:
    """Fixed-size replay buffer with contact-aware HER relabeling."""

    def __init__(self, capacity: int, her_k: int = 4, device: torch.device = torch.device("cpu")):
        self.capacity = capacity
        self.her_k = her_k
        self.device = device

        self._storage: list[dict] = []
        self._pos = 0

        # Episode accumulator
        self._current_episode: list[dict] = []

    # ------------------------------------------------------------------
    # Episode-level interface
    # ------------------------------------------------------------------
    def start_episode(self):
        self._current_episode = []

    def add_transition(self, transition: dict):
        """
        transition keys:
            obs, next_obs    — dicts of numpy arrays
            action           — (6,)
            reward           — float
            done             — bool
            true_offset      — (3,)
            achieved_goal    — float (insertion depth)
            desired_goal     — float
            ft_force_norm    — float (norm of ft_force_world)
        """
        self._current_episode.append(transition)

    def end_episode(self):
        """Finalise episode: store raw transitions and HER-relabeled ones."""
        ep = self._current_episode
        if len(ep) == 0:
            return

        # Store original transitions
        for t in ep:
            self._store(t)

        # Contact-aware filter: only relabel if episode had contact
        had_contact = any(t["ft_force_norm"] > 0.3 for t in ep)
        if not had_contact:
            self._current_episode = []
            return

        # HER "future" relabeling
        T = len(ep)
        for idx in range(T):
            future_indices = list(range(idx + 1, T))
            if len(future_indices) == 0:
                continue
            k_actual = min(self.her_k, len(future_indices))
            chosen = np.random.choice(future_indices, size=k_actual, replace=False)

            for f_idx in chosen:
                # Relabel goal with the achieved goal at future step
                new_goal = ep[f_idx]["achieved_goal"]
                # Recompute reward: success if achieved_goal >= desired_goal
                achieved = ep[idx]["achieved_goal"]
                success_depth = new_goal  # the new desired goal
                # Simple sparse-ish reward for relabeled transition
                relabel_reward = 200.0 if achieved >= success_depth else -0.01

                relabeled = {
                    "obs": ep[idx]["obs"],
                    "next_obs": ep[idx]["next_obs"],
                    "action": ep[idx]["action"],
                    "reward": relabel_reward,
                    "done": achieved >= success_depth,
                    "true_offset": ep[idx]["true_offset"],
                    "achieved_goal": achieved,
                    "desired_goal": new_goal,
                    "ft_force_norm": ep[idx]["ft_force_norm"],
                }
                self._store(relabeled)

        self._current_episode = []

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def sample(self, batch_size: int) -> dict:
        """Sample a batch and return tensors on device."""
        indices = np.random.randint(0, len(self._storage), size=batch_size)
        batch_list = [self._storage[i] for i in indices]
        return self._collate(batch_list)

    def __len__(self):
        return len(self._storage)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _store(self, transition: dict):
        if len(self._storage) < self.capacity:
            self._storage.append(transition)
        else:
            self._storage[self._pos] = transition
        self._pos = (self._pos + 1) % self.capacity

    def _collate(self, batch_list: list[dict]) -> dict:
        """Stack a list of transitions into batched tensors."""
        def _stack_obs(key):
            obs_dicts = [b[key] for b in batch_list]
            return {
                k: torch.as_tensor(
                    np.stack([o[k] for o in obs_dicts]),
                    dtype=torch.float32,
                    device=self.device,
                )
                for k in obs_dicts[0].keys()
            }

        return {
            "obs": _stack_obs("obs"),
            "next_obs": _stack_obs("next_obs"),
            "action": torch.as_tensor(
                np.stack([b["action"] for b in batch_list]),
                dtype=torch.float32, device=self.device,
            ),
            "reward": torch.as_tensor(
                np.array([[b["reward"]] for b in batch_list]),
                dtype=torch.float32, device=self.device,
            ),
            "done": torch.as_tensor(
                np.array([[float(b["done"])] for b in batch_list]),
                dtype=torch.float32, device=self.device,
            ),
            "true_offset": torch.as_tensor(
                np.stack([b["true_offset"] for b in batch_list]),
                dtype=torch.float32, device=self.device,
            ),
        }
