"""Observed physical milestones for cube-stack policy evaluation."""

from __future__ import annotations

import numpy as np

from mujoco_lab.behaviors.task import BookTask, CubeStackTask

DEFAULT_MIN_LIFT_HEIGHT_M = 0.04


class CubeProgressTracker:
    """Track each cube's best physical stage during one reset episode."""

    min_lift_height_m = DEFAULT_MIN_LIFT_HEIGHT_M

    def __init__(self, task: CubeStackTask) -> None:
        self.task = task
        sample = task.status()
        self.initial_heights = sample.centers[:, 2].copy()
        self._grasped = np.zeros(task.cubes, dtype=bool)
        self._lifted = np.zeros(task.cubes, dtype=bool)
        self._placed = np.zeros(task.cubes, dtype=bool)
        self._final_position_errors = sample.position_errors.copy()

    def observe(self) -> None:
        """Read the latest physics state once after an environment step."""
        sample = self.task.status()
        self._final_position_errors = sample.position_errors.copy()
        self._placed |= self.task.completed_placements()
        robots = tuple(self.task.simulator.robots.values())
        for index in range(self.task.cubes):
            grasped_now = any(self.task.has_grasp(index, robot=robot) for robot in robots)
            self._lifted[index] |= (
                self._grasped[index]
                and grasped_now
                and sample.centers[index, 2] - self.initial_heights[index] >= self.min_lift_height_m
            )
            self._grasped[index] |= grasped_now

    def result(self) -> dict[str, object]:
        """Return sticky milestones and the actual final position errors in meters."""
        stages = np.where(self._placed, 3, np.where(self._lifted, 2, self._grasped.astype(int)))
        return {
            "grasped": self._grasped.tolist(),
            "lifted": self._lifted.tolist(),
            "placed": self._placed.tolist(),
            "best_stage": stages.tolist(),
            "final_position_errors": self._final_position_errors.tolist(),
            "grasp_fraction": float(np.mean(self._grasped)),
            "lift_fraction": float(np.mean(self._lifted)),
            "place_fraction": float(np.mean(self._placed)),
            "best_progress": float(np.mean(stages) / 3),
            "final_position_error": float(np.mean(self._final_position_errors)),
        }


class BookProgressTracker:
    """Record physical book milestones and final pose errors without shaping reward."""

    min_lift_height_m = DEFAULT_MIN_LIFT_HEIGHT_M

    def __init__(self, task: BookTask) -> None:
        self.task = task
        self.grasped = False
        self.lifted = False
        self.inserted = False
        self.released = False
        self.status = task.status()

    def observe(self) -> None:
        """Read grasp contacts and released shelf placement after an action."""
        grasped_now = self.task.has_grasp()
        self.grasped |= grasped_now
        self.status = self.task.status()
        height = self.task.simulator.data.xpos[self.task.body, 2]
        self.lifted |= grasped_now and height - self.task.start_center[2] >= self.min_lift_height_m
        self.inserted |= (
            self.status.position_error < self.task.position_tolerance_m
            and self.status.rotation_error < self.task.rotation_tolerance_rad
            and self.status.is_touch_shelf
        )
        self.released |= self.status.is_released_stable

    def result(self) -> dict[str, object]:
        """Return sticky milestones and final position/angular errors in SI units."""
        return {
            "grasped": bool(self.grasped),
            "lifted": bool(self.lifted),
            "inserted": bool(self.inserted),
            "released": bool(self.released),
            "position_error": self.status.position_error,
            "rotation_error": self.status.rotation_error,
        }
