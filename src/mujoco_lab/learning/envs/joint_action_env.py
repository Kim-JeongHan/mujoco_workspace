"""Shared robot joint actions and physics cadence for learning tasks."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class JointActionEnv(gym.Env):
    """Apply arm joint and gripper opening-width targets in robot insertion order.

    Policy actions replace position goals without resetting the reference.
    Each Robot owns its reference, advanced by Simulator before every control tick.
    Gripper width commands are applied immediately.
    Success is checked after every physics tick, allowing an action to finish early.
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

        self.physics_steps_per_action = int(physics_steps_per_action)
        if self.physics_steps_per_action <= 0:
            raise ValueError("physics_steps_per_action must be positive")
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

        self.action_space = spaces.Box(
            low=np.asarray(lows, dtype=np.float32),
            high=np.asarray(highs, dtype=np.float32),
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
        """Advance an arm reference with bounded derivatives for action_dt seconds."""
        if self._done:
            raise RuntimeError("Reset the environment before stepping after an episode ends")
        if action.shape != self.action_space.shape:
            raise ValueError("Action must match the environment action shape")
        start_time = float(self.data.time)
        for robot in self.robots:
            robot.set_goal(
                action[self._action_slices[robot.name]], joint_names=self._arm_names[robot.name]
            )

        terminated = False
        physics_steps = 0
        for _ in range(self.physics_steps_per_action):
            self.simulator.run_steps(1)
            physics_steps += 1
            if self._is_success():
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
