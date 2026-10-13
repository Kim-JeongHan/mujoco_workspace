"""Goal relabeling contracts for flat robot observations."""

from typing import Any, Protocol

import numpy as np
from gymnasium import spaces


class GoalRelabeling(Protocol):
    """Describe goals and rewards using only stored observation arrays."""

    def achieved_goal(self, observation: np.ndarray, /) -> np.ndarray: ...

    def desired_goal(self, observation: np.ndarray, /) -> np.ndarray: ...

    def relabel_observation(self, observation: np.ndarray, goal: np.ndarray, /) -> np.ndarray: ...

    def compute_goal_reward(self, achieved: np.ndarray, desired: np.ndarray, /) -> np.ndarray: ...


class GoalEnv(GoalRelabeling, Protocol):
    """Expose physical transitions along with pure goal calculations."""

    action_space: spaces.Box

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]: ...

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]: ...


def rot6d_angle(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Compute rotation distance from row-major first-two-column representations."""
    first_columns = first.reshape(*first.shape[:-1], 3, 2)
    second_columns = second.reshape(*second.shape[:-1], 3, 2)
    first_last = np.cross(first_columns[..., 0], first_columns[..., 1])
    second_last = np.cross(second_columns[..., 0], second_columns[..., 1])
    trace = np.sum(first_columns * second_columns, axis=(-2, -1))
    trace += np.sum(first_last * second_last, axis=-1)
    return np.arccos(np.clip((trace - 1) / 2, -1, 1))
