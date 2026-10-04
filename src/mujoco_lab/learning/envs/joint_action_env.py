"""Shared robot joint actions and physics cadence for learning tasks."""

from __future__ import annotations

from numbers import Integral
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from mujoco_lab.control import ControlTarget, min_jerk_target


class JointActionEnv(gym.Env):
    """Apply joint and gripper targets in simulator robot insertion order.

    Each action replans a minimum-jerk segment from the preceding target,
    retaining its position, velocity, and acceleration. Success is checked
    after every physics tick, allowing an action to finish early.
    """

    action_space: spaces.Box
    observation_space: spaces.Box
    _rotation_indices: list[int]

    def __init__(self, task, *, max_steps=30_000, physics_steps_per_action=1):
        super().__init__()
        self.task = task
        self.simulator = task.simulator
        self.simulator.target_updater = None
        self.robots = tuple(self.simulator.robots.values())
        if not self.robots:
            raise ValueError("JointActionEnv requires at least one robot")
        self.model = self.simulator.model
        self.data = self.simulator.data
        self.max_steps = max_steps
        if (
            isinstance(physics_steps_per_action, bool)
            or not isinstance(physics_steps_per_action, Integral)
            or physics_steps_per_action <= 0
        ):
            raise ValueError("physics_steps_per_action must be a positive integer")
        self.physics_steps_per_action = int(physics_steps_per_action)
        self._steps = 0
        self._done = False
        self._grasp_sites = {robot.name: robot.state.site_id("grasp") for robot in self.robots}
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

    @property
    def observation_metadata(self) -> dict[str, int | list[int]]:
        """Return the frame width and rotation columns for dataset metadata."""
        return {
            "frame_dim": self.observation_space.shape[0],
            "rotation_indices": self._rotation_indices,
        }

    def _is_success(self) -> bool:
        raise NotImplementedError

    def _get_obs(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def action_dt(self) -> float:
        """Nominal simulated seconds for one action."""
        return self.simulator.dt * self.physics_steps_per_action

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._done:
            raise RuntimeError("Reset the environment before stepping after an episode ends")
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
        start_time = float(self.data.time)
        transit = self.action_dt
        arm_segments = {}
        for robot in self.robots:
            block = action[self._action_slices[robot.name]]
            names = self._arm_names[robot.name]
            if names:
                arm_pos = block[: len(names)]
                if robot.target is None:
                    raise ValueError("Configure a joint-target controller before stepping")
                arm_segments[robot.name] = (
                    robot.target,
                    ControlTarget(arm_pos[orders[robot.name]]),
                )
            if robot.gripper is not None:
                gripper_pos = block[len(names)]
                robot.gripper.set_target(float(gripper_pos))

        def update_arm_targets():
            elapsed = float(self.data.time) - start_time
            for robot in self.robots:
                if robot.name in arm_segments:
                    start, end = arm_segments[robot.name]
                    robot.target = min_jerk_target(start, end, elapsed, transit)

        terminated = False
        physics_steps = 0
        for _ in range(self.physics_steps_per_action):
            update_arm_targets()
            self.simulator.run_steps(1)
            physics_steps += 1
            mujoco.mj_forward(self.model, self.data)
            for robot in self.robots:
                robot.update_state()
            if self._is_success():
                terminated = True
                break
        # Carry the target at the exact action endpoint into the next segment.
        update_arm_targets()
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
