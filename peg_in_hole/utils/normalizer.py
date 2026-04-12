"""
Running mean / std normalizer — one per observation branch.

IMPORTANT: Do NOT normalise the phase_bit channel (last element of
proprioception).  If normalised it becomes ~0 and the gating module
cannot distinguish phases.
"""

from __future__ import annotations

import numpy as np


class RunningNormalizer:
    """Welford online mean/variance tracker with per-element normalisation."""

    def __init__(self, shape: tuple, clip: float = 5.0, exclude_last: bool = False):
        """
        Parameters
        ----------
        shape : shape of the observation (excluding batch dim)
        clip : clip normalised values to [-clip, clip]
        exclude_last : if True, do NOT normalise the last element
                       (for proprioception — preserves the phase_bit)
        """
        self.shape = shape
        self.clip = clip
        self.exclude_last = exclude_last

        flat = int(np.prod(shape))
        self._mean = np.zeros(flat, dtype=np.float64)
        self._var = np.ones(flat, dtype=np.float64)
        self._count = 1e-4  # avoid division by zero

    def update(self, x: np.ndarray):
        """Update running statistics with a batch of observations."""
        flat = x.reshape(-1, int(np.prod(self.shape)))
        batch_mean = flat.mean(axis=0)
        batch_var = flat.var(axis=0)
        batch_count = flat.shape[0]

        delta = batch_mean - self._mean
        total = self._count + batch_count
        new_mean = self._mean + delta * batch_count / total
        m_a = self._var * self._count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self._count * batch_count / total
        new_var = m2 / total

        self._mean = new_mean
        self._var = new_var
        self._count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalise observation."""
        orig_shape = x.shape
        flat = x.reshape(-1, int(np.prod(self.shape)))
        std = np.sqrt(self._var + 1e-8)
        normed = np.clip((flat - self._mean) / std, -self.clip, self.clip)

        if self.exclude_last:
            # Restore last element un-normalised
            normed[:, -1] = flat[:, -1]

        return normed.reshape(orig_shape).astype(np.float32)
