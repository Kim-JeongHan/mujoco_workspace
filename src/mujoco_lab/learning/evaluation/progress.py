"""Observed physical milestones for cube-stack policy evaluation."""

from __future__ import annotations

import numpy as np

from mujoco_lab.tasks.cube_stack import CubeStackTask
from mujoco_lab.tasks.cube_stack_motion import has_physical_grasp


class CubeProgressTracker:
    """Track each cube's best physical stage during one reset episode."""

    lift_meters = 0.04
    place_hold_seconds = 0.5

    def __init__(self, task: CubeStackTask) -> None:
        self.task = task
        sample = task.measurements()
        self.initial_heights = sample.centers[:, 2].copy()
        self._grasped = np.zeros(task.cubes, dtype=bool)
        self._lifted = np.zeros(task.cubes, dtype=bool)
        self._placed = np.zeros(task.cubes, dtype=bool)
        self._place_since: list[float | None] = [None] * task.cubes
        self._final_distances = sample.goal_distances.copy()

    def observe(self) -> None:
        """Read the latest physics state once after an environment step."""
        sample = self.task.measurements()
        now = float(self.task.simulator.data.time)
        self._final_distances = sample.goal_distances.copy()
        robots = tuple(self.task.simulator.robots.values())
        for index in range(self.task.cubes):
            grasped_now = any(has_physical_grasp(robot, index) for robot in robots)
            self._lifted[index] |= (
                self._grasped[index]
                and grasped_now
                and sample.centers[index, 2] - self.initial_heights[index] >= self.lift_meters
            )
            self._grasped[index] |= grasped_now
            if sample.stable_placement[index]:
                if self._place_since[index] is None:
                    self._place_since[index] = now
                if now - self._place_since[index] >= self.place_hold_seconds:
                    self._placed[index] = True
            else:
                self._place_since[index] = None

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
