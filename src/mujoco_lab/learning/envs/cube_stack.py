"""Cube stacking Gym environment with robot-local physical actions."""

from __future__ import annotations

from numbers import Integral
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from mujoco_lab.assets.randomization import sample_cube_positions
from mujoco_lab.control import ControlTarget
from mujoco_lab.tasks import CubeStackTask


class CubeStackEnv(gym.Env):
    """Apply each robot's arm targets and optional gripper target per step.

    Robots follow simulator insertion order. Observations group all arm positions,
    arm velocities, gripper widths, grasp positions, and grasp rotations, then
    append each cube's position, rotation, offsets to every grasp, and goal offset.
    Rotations flatten the first two matrix columns row-major. Arm positions
    and gripper targets use their joints' native radians or meters. Configure
    joint-target arm controllers before stepping; this environment selects none.
    """

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
        super().__init__()
        self.task = task
        self.simulator = task.simulator
        self.simulator.target_updater = None
        self.robots = tuple(self.simulator.robots.values())
        if not self.robots:
            raise ValueError("CubeStackEnv requires at least one robot")
        if not np.isfinite(xy_range) or xy_range < 0:
            raise ValueError("xy_range must be finite and nonnegative")
        if not np.isfinite(min_gap) or min_gap < 0:
            raise ValueError("min_gap must be finite and nonnegative")
        if (
            isinstance(cube_yaw_range_degrees, bool)
            or not np.isfinite(cube_yaw_range_degrees)
            or cube_yaw_range_degrees < 0
        ):
            raise ValueError("cube_yaw_range_degrees must be finite and nonnegative")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if (
            isinstance(physics_steps_per_action, bool)
            or not isinstance(physics_steps_per_action, Integral)
            or physics_steps_per_action <= 0
        ):
            raise ValueError("physics_steps_per_action must be a positive integer")
        self.xy_range = float(xy_range)
        self.min_gap = float(min_gap)
        self.cube_yaw_range_degrees = float(cube_yaw_range_degrees)
        self.max_steps = max_steps
        self.physics_steps_per_action = int(physics_steps_per_action)
        self._steps = 0
        self._done = False
        self.model = self.simulator.model
        self.data = self.simulator.data

        self._grasp_sites = {robot.name: robot.state.site_id("grasp") for robot in self.robots}
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

        self._arm_names = {}
        self._arm_slots = {}
        self._action_slices = {}
        lows, highs = [], []
        for robot in self.robots:
            names, slots = robot.get_arm_joint_mapping()
            limits = robot.state.get_joint_limits(slots)
            self._arm_names[robot.name] = names
            self._arm_slots[robot.name] = slots
            start = len(lows)
            lows.extend(limits[:, 0])
            highs.extend(limits[:, 1])
            if robot.gripper is not None:
                gripper_limits = robot.gripper.get_control_limits()
                lows.append(gripper_limits[0])
                highs.append(gripper_limits[1])
            self._action_slices[robot.name] = slice(start, len(lows))

        self._action_low = np.asarray(lows, dtype=float)
        self._action_high = np.asarray(highs, dtype=float)
        self.action_space = spaces.Box(
            low=self._action_low.astype(np.float32),
            high=self._action_high.astype(np.float32),
        )
        arm_values = sum(len(slots) for slots in self._arm_slots.values())
        grippers = sum(robot.gripper is not None for robot in self.robots)
        size = 2 * arm_values + grippers + 9 * len(self.robots)
        size += task.cubes * (12 + 3 * len(self.robots))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(size,), dtype=np.float32
        )

    @property
    def action_dt(self) -> float:
        """Nominal simulated seconds for one action."""
        return self.simulator.dt * self.physics_steps_per_action

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
                    cosines * self._cube_half_sizes[:, 0]
                    + sines * self._cube_half_sizes[:, 1],
                    sines * self._cube_half_sizes[:, 0]
                    + cosines * self._cube_half_sizes[:, 1],
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

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._done:
            raise RuntimeError("Reset CubeStackEnv before stepping after an episode ends")
        if action.shape != self.action_space.shape or not np.isfinite(action).all():
            raise ValueError(f"Action must contain {self.action_space.shape[0]} finite targets")
        low, high = self._action_low, self._action_high
        if np.any(action < low - 1e-6) or np.any(action > high + 1e-6):
            raise ValueError("Action is outside the robot joint or gripper limits")
        action = np.clip(action, low, high)
        orders = {
            robot.name: robot.get_control_target_order(self._arm_names[robot.name])
            for robot in self.robots
        }
        for robot in self.robots:
            block = action[self._action_slices[robot.name]]
            names = self._arm_names[robot.name]
            if names:
                arm = block[: len(names)]
                robot.target = ControlTarget(arm[orders[robot.name]])
            if robot.gripper is not None:
                robot.gripper.set_target(float(block[len(names)]))
        start_time = float(self.data.time)
        terminated = False
        physics_steps = 0
        for _ in range(self.physics_steps_per_action):
            self.simulator.run_steps(1)
            physics_steps += 1
            mujoco.mj_forward(self.model, self.data)
            for robot in self.robots:
                robot.update_state()
            if self.task.status().released_stable_stack:
                terminated = True
                break
        self._steps += 1
        truncated = self._steps >= self.max_steps and not terminated
        self._done = terminated or truncated
        reason = "success" if terminated else "time_limit" if truncated else None
        return (
            self._get_obs(),
            float(terminated),
            terminated,
            truncated,
            {
                "success": terminated,
                "termination_reason": reason,
                "physics_steps": physics_steps,
                "elapsed_dt": float(self.data.time) - start_time,
            },
        )

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
