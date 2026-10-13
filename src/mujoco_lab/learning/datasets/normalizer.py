from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Normalizer:
    """Feature-wise normalizer for states and actions."""

    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray
    state_clip: float | None = field(default=None, kw_only=True)

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
        """Fit statistics with float64 accumulation and store them as float32."""
        state_mean = states.mean(axis=0, dtype=np.float64).astype(np.float32)
        state_std = cls._safe_std(states.std(axis=0, dtype=np.float64)).astype(np.float32)
        state_mean[list(state_passthrough_indices)] = 0
        state_std[list(state_passthrough_indices)] = 1
        action_mean = actions.mean(axis=0, dtype=np.float64).astype(np.float32)
        action_std = cls._safe_std(actions.std(axis=0, dtype=np.float64)).astype(np.float32)
        return cls(state_mean, state_std, action_mean, action_std)

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        normalized = (state - self.state_mean) / self.state_std
        return (
            normalized
            if self.state_clip is None
            else np.clip(normalized, -self.state_clip, self.state_clip)
        )

    def denormalize_state(self, state: np.ndarray) -> np.ndarray:
        return state * self.state_std + self.state_mean

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        return (action - self.action_mean) / self.action_std

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        return action * self.action_std + self.action_mean


class RunningMoments:
    """Merge online frame statistics while preserving rotation passthrough columns."""

    def __init__(self, frame_dim: int, passthrough_indices: Sequence[int] = ()) -> None:
        self.count = 0
        self.mean = np.zeros(frame_dim, dtype=np.float64)
        self.m2 = np.zeros(frame_dim, dtype=np.float64)
        self.passthrough_indices = list(passthrough_indices)

    def update(self, frames: np.ndarray, normalizer: Normalizer) -> None:
        """Update the shared normalizer in place using parallel variance accumulation."""
        frames = frames.reshape(-1, len(self.mean))
        size = len(frames)
        mean = frames.mean(axis=0, dtype=np.float64)
        m2 = frames.var(axis=0, dtype=np.float64) * size
        delta = mean - self.mean
        count = self.count + size
        self.m2 += m2 + delta**2 * self.count * size / count
        self.mean += delta * size / count
        self.count = count
        normalizer.state_mean[:] = self.mean
        # A centimeter-scale floor avoids amplifying nearly constant goal offsets.
        normalizer.state_std[:] = np.maximum(np.sqrt(self.m2 / count), 0.01)
        normalizer.state_mean[self.passthrough_indices] = 0
        normalizer.state_std[self.passthrough_indices] = 1
