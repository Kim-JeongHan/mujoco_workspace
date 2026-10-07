"""Shared physical grasp checks for open and close recipe modes."""

from typing import Literal


class GraspMonitor:
    """Wait for a closing grasp and monitor contact loss during closed-mode motion."""

    def __init__(self, grace_s: float) -> None:
        self.grace_s = grace_s
        self._lost_since: float | None = None

    def reset(self) -> None:
        self._lost_since = None

    @staticmethod
    def ready(mode: Literal[0, 1], grasped: bool) -> bool:
        """Closed-mode stages finish only after physical contact is established."""
        return mode == 0 or grasped

    def failure(
        self,
        *,
        mode: Literal[0, 1],
        hold: bool,
        started: bool,
        grasped: bool,
        now: float,
        stage_name: str,
    ) -> str | None:
        """Require a grasp before arm motion and tolerate brief contact loss afterward."""
        if mode == 0 or hold or grasped:
            self.reset()
        elif not started:
            return f"No two-finger physical grasp for {stage_name}"
        elif self._lost_since is None:
            self._lost_since = now
        elif now - self._lost_since > self.grace_s:
            return f"Lost two-finger grasp during {stage_name}"
        return None
