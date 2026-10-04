"""Book insertion Gym environment with robot-local physical actions."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
from gymnasium import spaces

from mujoco_lab.behaviors import BookTask
from mujoco_lab.learning.envs.joint_action_env import JointActionEnv


def book_observation_layout() -> tuple[int, list[int]]:
    """Return the frame width and rot6d indices for a seven-joint book robot."""
    return 48, [*range(18, 24), *range(27, 33), *range(36, 42)]


class BookEnv(JointActionEnv):
    """Observe the robot, free book, shelf goal, and their relative positions.

    Observations contain arm positions, arm velocities, gripper width, grasp
    position and rotation, book position and rotation, goal position and
    rotation, book-minus-grasp offset, and goal-minus-book offset. Rotations
    flatten the first two matrix columns row-major.
    """

    def __init__(
        self,
        task: BookTask,
        *,
        xy_range: float = 0.0,
        book_yaw_range_degrees: float = 0.0,
        max_steps: int = 30_000,
        physics_steps_per_action: int = 1,
    ) -> None:
        super().__init__(
            task, max_steps=max_steps, physics_steps_per_action=physics_steps_per_action
        )
        self.xy_range = float(xy_range)
        self.book_yaw_range_degrees = float(book_yaw_range_degrees)
        self.robot = task.robot
        self._book_qpos = int(self.model.jnt_qposadr[task.joint])
        self._home_xy = self.data.qpos[self._book_qpos : self._book_qpos + 2].copy()
        self._home_quat = self.data.qpos[self._book_qpos + 3 : self._book_qpos + 7].copy()
        robot_width = 2 * len(self._arm_slots[self.robot.name]) + int(
            self.robot.gripper is not None
        )
        rotation_start = robot_width + 3
        self._rotation_indices = [
            *range(rotation_start, rotation_start + 6),
            *range(rotation_start + 9, rotation_start + 15),
            *range(rotation_start + 18, rotation_start + 24),
        ]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(robot_width + 33,), dtype=np.float32
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.simulator.reset()
        adr = self._book_qpos
        self.data.qpos[adr : adr + 2] = self._home_xy + self.np_random.uniform(
            -self.xy_range, self.xy_range, size=2
        )
        yaw = float(
            self.np_random.uniform(-self.book_yaw_range_degrees, self.book_yaw_range_degrees)
        )
        half_angle = np.deg2rad(yaw) / 2
        yaw_quat = np.array([np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)])
        mujoco.mju_mulQuat(self.data.qpos[adr + 3 : adr + 7], yaw_quat, self._home_quat)
        mujoco.mj_forward(self.model, self.data)
        self.robot.update_state()
        self.task.reset()
        self._steps = 0
        self._done = False
        return self._get_obs(), {"seed": seed, "book_yaw_degrees": yaw}

    def _is_success(self) -> bool:
        return self.task.status().released_stable

    @property
    def camera_lookat(self) -> np.ndarray:
        """Center the evaluation view between the book start and shelf goal."""
        center = (self.task.start_center + self.data.site_xpos[self.task.target]) / 2
        center[2] += 0.12
        return center

    def _get_obs(self) -> np.ndarray:
        state = self.robot.state.snapshot()
        slots = self._arm_slots[self.robot.name]
        site = self._grasp_sites[self.robot.name]
        grasp = self.data.site_xpos[site]
        book = self.data.xpos[self.task.body]
        goal = self.data.site_xpos[self.task.target]
        parts = [state.qpos[slots], state.qvel[slots]]
        if self.robot.gripper is not None:
            parts.append([self.robot.gripper.get_width()])
        parts.extend(
            (
                grasp,
                self.data.site_xmat[site].reshape(3, 3)[:, :2].reshape(-1),
                book,
                self.data.xmat[self.task.body].reshape(3, 3)[:, :2].reshape(-1),
                goal,
                self.data.site_xmat[self.task.target].reshape(3, 3)[:, :2].reshape(-1),
                book - grasp,
                goal - book,
            )
        )
        return np.concatenate(parts).astype(np.float32)
