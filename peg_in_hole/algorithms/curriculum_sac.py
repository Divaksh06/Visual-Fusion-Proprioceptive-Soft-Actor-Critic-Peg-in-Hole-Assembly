"""
SAC + Curriculum Stage Progression.

Wraps the base SAC agent and coordinates with the CurriculumManager to
advance / regress training stages based on rolling success rate.
"""

from __future__ import annotations

from peg_in_hole.algorithms.sac import SAC
from peg_in_hole.envs.curriculum import CurriculumManager


class CurriculumSAC:
    """Thin orchestrator around SAC + CurriculumManager."""

    def __init__(self, sac: SAC, curriculum: CurriculumManager):
        self.sac = sac
        self.curriculum = curriculum

    def report_episode(self, success: bool) -> dict:
        """
        Report one episode outcome.
        Returns a dict with 'stage_changed', 'stage_id', 'success_rate', 'stage_name'.
        """
        return self.curriculum.report_episode(success)

    def current_stage_name(self) -> str:
        return self.curriculum.current_stage()["name"]

    def current_stage_id(self) -> int:
        return self.curriculum.stage_id()

    def current_success_depth(self) -> float:
        return self.curriculum.current_success_depth()
