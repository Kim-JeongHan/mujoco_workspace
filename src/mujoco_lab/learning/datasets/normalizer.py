from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Normalizer:
    """Feature-wise normalizer for states and actions."""

    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    @staticmethod
    def _safe_std(std: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        return np.maximum(std, eps)

    @classmethod
    def from_data(
        cls,
        states: np.ndarray,
        actions: np.ndarray,
        *,
        state_passthrough_indices: Sequence[int] = (),
    ) -> Normalizer:
        state_mean = states.mean(axis=0)
        state_std = cls._safe_std(states.std(axis=0))
        state_mean[list(state_passthrough_indices)] = 0
        state_std[list(state_passthrough_indices)] = 1
        action_mean = actions.mean(axis=0)
        action_std = cls._safe_std(actions.std(axis=0))
        return cls(state_mean, state_std, action_mean, action_std)

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        return (state - self.state_mean) / self.state_std

    def denormalize_state(self, state: np.ndarray) -> np.ndarray:
        return state * self.state_std + self.state_mean

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        return (action - self.action_mean) / self.action_std

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        return action * self.action_std + self.action_mean
