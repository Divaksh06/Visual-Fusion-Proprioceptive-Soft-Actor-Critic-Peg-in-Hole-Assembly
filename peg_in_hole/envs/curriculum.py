"""
Curriculum manager — controls training stage progression and regression.

Stages are loaded from curriculum_stages.yaml.  Progression happens when the
rolling success rate over the evaluation window exceeds the current stage's
``advance_threshold``.  If it drops more than ``regression_drop`` below the
threshold the stage steps *back* by one (regression protection).
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import yaml


class CurriculumManager:
    """Manages curriculum stage progression."""

    def __init__(self, config_path: str | None = None):
        if config_path is not None and Path(config_path).exists():
            with open(config_path, "r") as f:
                raw = yaml.safe_load(f)
            self.stages = raw["stages"]
            self.eval_window = raw.get("eval_window", 100)
            self.regression_drop = raw.get("regression_drop", 0.20)
        else:
            # Fallback — single stage (full task)
            self.stages = [{
                "id": 0,
                "name": "full_task",
                "pos_noise_mm": 5.0,
                "rot_noise_deg": 6.0,
                "success_depth_mm": 15.0,
                "advance_threshold": 0.65,
            }]
            self.eval_window = 100
            self.regression_drop = 0.20

        self._current_id = 0
        self._episode_results: deque = deque(maxlen=self.eval_window)

    # ---- Public API ----

    def current_stage(self) -> dict:
        return self.stages[self._current_id]

    def current_success_depth(self) -> float:
        """Return success_depth in meters."""
        return self.stages[self._current_id]["success_depth_mm"] / 1000.0

    def stage_id(self) -> int:
        return self._current_id

    def report_episode(self, success: bool) -> dict:
        """
        Report one episode result.  Returns dict with 'stage_changed', 'stage_id',
        'success_rate'.
        """
        self._episode_results.append(float(success))
        rate = self._rolling_success_rate()

        stage_changed = False

        # Advance?
        if len(self._episode_results) >= self.eval_window:
            thresh = self.stages[self._current_id]["advance_threshold"]
            if rate >= thresh and self._current_id < len(self.stages) - 1:
                self._current_id += 1
                self._episode_results.clear()
                stage_changed = True
            # Regress?
            elif rate < (thresh - self.regression_drop) and self._current_id > 0:
                self._current_id -= 1
                self._episode_results.clear()
                stage_changed = True

        return {
            "stage_changed": stage_changed,
            "stage_id": self._current_id,
            "success_rate": rate,
            "stage_name": self.stages[self._current_id]["name"],
        }

    # ---- Internal ----

    def _rolling_success_rate(self) -> float:
        if len(self._episode_results) == 0:
            return 0.0
        return sum(self._episode_results) / len(self._episode_results)
