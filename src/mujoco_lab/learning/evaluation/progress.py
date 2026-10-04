"""Observed physical milestones for cube-stack policy evaluation."""

from __future__ import annotations

import numpy as np

from mujoco_lab.behaviors.cube_stack import CubeStackTask, has_physical_grasp


class CubeProgressTracker:
    """Track each cube's best physical stage during one reset episode."""

    lift_meters = 0.04

    def __init__(self, task: CubeStackTask) -> None:
        self.task = task
        sample = task.measurements()
        self.initial_heights = sample.centers[:, 2].copy()
        self._grasped = np.zeros(task.cubes, dtype=bool)
        self._lifted = np.zeros(task.cubes, dtype=bool)
        self._placed = np.zeros(task.cubes, dtype=bool)
        self._final_distances = sample.goal_distances.copy()

    def observe(self) -> None:
        """Read the latest physics state once after an environment step."""
        sample = self.task.measurements()
        self._final_distances = sample.goal_distances.copy()
        self._placed |= self.task.completed_placements()
        robots = tuple(self.task.simulator.robots.values())
        for index in range(self.task.cubes):
            grasped_now = any(has_physical_grasp(robot, index) for robot in robots)
            self._lifted[index] |= (
                self._grasped[index]
                and grasped_now
                and sample.centers[index, 2] - self.initial_heights[index] >= self.lift_meters
            )
            self._grasped[index] |= grasped_now

    def result(self) -> dict[str, object]:
        """Return sticky milestones and the actual final goal distances in meters."""
        stages = np.where(self._placed, 3, np.where(self._lifted, 2, self._grasped.astype(int)))
        return {
            "cube_grasped": self._grasped.tolist(),
            "cube_lifted": self._lifted.tolist(),
            "cube_placed": self._placed.tolist(),
            "cube_best_stage": stages.tolist(),
            "cube_final_goal_distance": self._final_distances.tolist(),
            "grasp_fraction": float(np.mean(self._grasped)),
            "lift_fraction": float(np.mean(self._lifted)),
            "place_fraction": float(np.mean(self._placed)),
            "best_progress": float(np.mean(stages) / 3),
            "final_goal_distance": float(np.mean(self._final_distances)),
        }


class BookProgressTracker:
    """Record physical book milestones and final pose errors without shaping reward."""

    def __init__(self, task) -> None:
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
        self.lifted |= grasped_now and height - self.task.start_center[2] >= 0.04
        self.inserted |= (
            self.status.position_error < 0.015
            and self.status.rotation_error < 0.12
            and self.status.supported
        )
        self.released |= self.status.released_stable

    def result(self) -> dict[str, object]:
        """Return sticky milestones and final position/angular errors in SI units."""
        return {
            "book_grasped": bool(self.grasped),
            "book_lifted": bool(self.lifted),
            "book_inserted": bool(self.inserted),
            "book_released": bool(self.released),
            "book_position_error": self.status.position_error,
            "book_rotation_error": self.status.rotation_error,
        }
