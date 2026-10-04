"""Cube stacking Gym environment with robot-local physical actions."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
from gymnasium import spaces

from mujoco_lab.assets.randomization import sample_cube_positions
from mujoco_lab.behaviors import CubeStackTask
from mujoco_lab.learning.envs.joint_action_env import JointActionEnv


def cube_stack_observation_layout(cubes: int) -> tuple[int, list[int]]:
    """Return the frame width and rot6d indices for one robot and ``cubes`` cubes."""
    return 24 + 15 * cubes, [
        *range(18, 24),
        *(index for cube in range(cubes) for index in range(27 + 15 * cube, 33 + 15 * cube)),
    ]


class CubeStackEnv(JointActionEnv):
    """Apply each robot's arm targets and optional gripper target per step.

    Robots follow simulator insertion order. Observations group all arm positions,
    arm velocities, gripper widths, grasp positions, and grasp rotations, then
    append each cube's position, rotation, offsets to every grasp, and goal offset.
    Rotations flatten the first two matrix columns row-major. Arm positions
    and gripper targets use their joints' native radians or meters. Configure
    joint-target arm controllers before stepping; this environment selects none.
    Arm commands replan a minimum-jerk segment lasting one action period,
    retaining the preceding target's position, velocity, and acceleration.
    """

    action_space: spaces.Box
    observation_space: spaces.Box

    def __init__(
        self,
        task: CubeStackTask,
        *,
        xy_range: float = 0.02,
        min_gap: float = 0.01,
        cube_yaw_range_degrees: float = 0.0,
        max_steps: int = 30_000,
        physics_steps_per_action: int = 1,
    ) -> None:
        super().__init__(
            task, max_steps=max_steps, physics_steps_per_action=physics_steps_per_action
        )
        self.xy_range = float(xy_range)
        self.min_gap = float(min_gap)
        self.cube_yaw_range_degrees = float(cube_yaw_range_degrees)
        if not np.isfinite(self.cube_yaw_range_degrees) or self.cube_yaw_range_degrees < 0:
            raise ValueError("cube_yaw_range_degrees must be finite and nonnegative")

        self._cube_bodies = []
        self._goal_bodies = []
        self._cube_qpos = []
        home_xy = []
        home_quat = []
        cube_half_sizes = []
        for index in range(task.cubes):
            name = f"cube{index}/object_0"
            self._cube_bodies.append(self.model.body(name).id)
            self._goal_bodies.append(self.model.body(f"cube{index}/object_target_0").id)
            qpos = int(self.model.joint(f"cube{index}/object_joint_0").qposadr[0])
            self._cube_qpos.append(qpos)
            home_xy.append(self.data.qpos[qpos : qpos + 2].copy())
            home_quat.append(self.data.qpos[qpos + 3 : qpos + 7].copy())
            cube_half_sizes.append(self.model.geom(name).size[:2].copy())
        self._home_xy = np.array(home_xy)
        self._home_quat = np.array(home_quat)
        self._cube_half_sizes = np.array(cube_half_sizes)

        arm_values = sum(len(slots) for slots in self._arm_slots.values())
        grippers = sum(robot.gripper is not None for robot in self.robots)
        size = 2 * arm_values + grippers + 9 * len(self.robots)
        size += task.cubes * (12 + 3 * len(self.robots))
        if len(self.robots) == 1 and arm_values == 7 and grippers == 1:
            size, _ = cube_stack_observation_layout(task.cubes)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(size,), dtype=np.float32
        )

        robot_rotation_start = 2 * arm_values + grippers + 3 * len(self.robots)
        cube_start = robot_rotation_start + 6 * len(self.robots)
        cube_width = 12 + 3 * len(self.robots)
        self._rotation_indices = [
            *range(robot_rotation_start, cube_start),
            *(
                index
                for cube in range(task.cubes)
                for index in range(
                    cube_start + cube * cube_width + 3, cube_start + cube * cube_width + 9
                )
            ),
        ]

    def _randomize_cubes(self) -> list[float]:
        if self.cube_yaw_range_degrees:
            yaws = self.np_random.uniform(
                -self.cube_yaw_range_degrees,
                self.cube_yaw_range_degrees,
                size=len(self._cube_qpos),
            )
            angles = np.deg2rad(yaws)
            cosines = np.abs(np.cos(angles))
            sines = np.abs(np.sin(angles))
            half_sizes = np.column_stack(
                (
                    cosines * self._cube_half_sizes[:, 0] + sines * self._cube_half_sizes[:, 1],
                    sines * self._cube_half_sizes[:, 0] + cosines * self._cube_half_sizes[:, 1],
                )
            )
        else:
            yaws = np.zeros(len(self._cube_qpos))
            angles = np.zeros(len(self._cube_qpos))
            half_sizes = self._cube_half_sizes
        positions = sample_cube_positions(
            self._home_xy,
            half_sizes,
            self.np_random,
            self.xy_range,
            self.min_gap,
            100,
        )
        for adr, xy, angle, home in zip(
            self._cube_qpos, positions, angles, self._home_quat, strict=True
        ):
            self.data.qpos[adr : adr + 2] = xy
            if self.cube_yaw_range_degrees:
                half_angle = angle / 2
                yaw_quat = np.array([np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)])
                mujoco.mju_mulQuat(self.data.qpos[adr + 3 : adr + 7], yaw_quat, home)
        return yaws.tolist()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.task.reset()
        yaws = self._randomize_cubes()
        mujoco.mj_forward(self.model, self.data)
        for robot in self.robots:
            robot.update_state()
        self._steps = 0
        self._done = False
        return self._get_obs(), {"seed": seed, "cube_yaws_degrees": yaws}

    def _is_success(self) -> bool:
        return self.task.status().released_stable_stack

    @property
    def camera_lookat(self) -> np.ndarray:
        """Center the evaluation view between the starting cubes and stack goals."""
        center = (self.task.starts.mean(axis=0) + self.task.goals.mean(axis=0)) / 2
        center[2] += 0.12
        return center

    def _get_obs(self) -> np.ndarray:
        """Read grouped robot states, then each cube pose and relative positions."""
        arm_qpos, arm_qvel, widths, ee_positions, ee_rotations = [], [], [], [], []

        # robot
        for robot in self.robots:
            state = robot.state.snapshot()
            slots = self._arm_slots[robot.name]
            arm_qpos.extend(state.qpos[slots])
            arm_qvel.extend(state.qvel[slots])
            if robot.gripper is not None:
                widths.append(robot.gripper.get_width())
            site = self._grasp_sites[robot.name]
            ee_positions.append(self.data.site_xpos[site].copy())
            ee_rotations.extend(self.data.site_xmat[site].reshape(3, 3)[:, :2].reshape(-1))
        parts = [arm_qpos, arm_qvel, widths, *ee_positions, ee_rotations]

        # cube
        for body_id, goal_id in zip(self._cube_bodies, self._goal_bodies, strict=True):
            position = self.data.body(body_id).xpos
            rotation = self.data.body(body_id).xmat.reshape(3, 3)[:, :2].reshape(-1)
            goal = self.data.body(goal_id).xpos
            parts.extend((position, rotation))
            parts.extend(position - ee_pos for ee_pos in ee_positions)
            parts.append(goal - position)
        observation = np.concatenate(parts).astype(np.float32)
        if observation.shape != self.observation_space.shape or not np.isfinite(observation).all():
            raise RuntimeError("CubeStackEnv produced an invalid observation")
        return observation
