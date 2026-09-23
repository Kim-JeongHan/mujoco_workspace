"""Collision checking utilities."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np


class CollisionChecker(ABC):
    """Abstract base class for collision checkers."""

    @abstractmethod
    def is_collision_free(self, state: np.ndarray) -> bool:
        """Check if a state is collision-free.

        Args:
            state: The state to check

        Returns:
            True if collision-free, False otherwise
        """
        pass

    @abstractmethod
    def is_path_collision_free(
        self, from_state: np.ndarray, to_state: np.ndarray, resolution: float = 0.1
    ) -> bool:
        """Check if the straight-line path between two states is collision-free.

        Args:
            from_state: Starting state
            to_state: Ending state
            resolution: Step size for checking intermediate points

        Returns:
            True if the entire path is collision-free, False otherwise
        """
        pass


class EmptyCollisionChecker(CollisionChecker):
    """Collision checker for obstacle-free environments."""

    def is_collision_free(self, state: np.ndarray) -> bool:
        """Always returns True (no obstacles).

        Args:
            state: The state to check

        Returns:
            Always True
        """
        return True

    def is_path_collision_free(
        self, from_state: np.ndarray, to_state: np.ndarray, resolution: float = 0.1
    ) -> bool:
        """Always returns True (no obstacles).

        Args:
            from_state: Starting state
            to_state: Ending state
            resolution: Step size (unused)

        Returns:
            Always True
        """
        return True


class BoundedCollisionChecker(CollisionChecker):
    """Collision checker that enforces state bounds and delegates collision checks."""

    def __init__(
        self,
        base_checker: CollisionChecker,
        bounds: Sequence[Sequence[float]],
        *,
        eps: float = 1e-9,
    ) -> None:
        """Initialize checker.

        Args:
            base_checker: Underlying collision checker to delegate obstacle checks to.
            bounds: Sequence of (min, max) tuples per dimension.
            eps: Tolerance used for inclusive bounds check.
        """
        bounds_array = np.array(bounds, dtype=float)
        if bounds_array.ndim != 2 or bounds_array.shape[1] != 2:
            raise ValueError("bounds must be shape (N, 2)")
        if np.any(bounds_array[:, 0] > bounds_array[:, 1]):
            raise ValueError("Each bound must satisfy min <= max")

        self.base_checker = base_checker
        self.bounds = bounds_array
        self.eps = eps

    def _within_bounds(self, state: np.ndarray) -> bool:
        """Return True when position is inside bounds."""
        position = np.array(state, dtype=float)
        if position.ndim != 1:
            raise ValueError("state must be a 1-D array")
        if position.size < self.bounds.shape[0]:
            raise ValueError(
                f"state must have at least {self.bounds.shape[0]} dimensions, got {position.size}"
            )

        lower = self.bounds[:, 0]
        upper = self.bounds[:, 1]
        truncated = position[: self.bounds.shape[0]]
        return bool(np.all(truncated >= lower - self.eps) and np.all(truncated <= upper + self.eps))

    def is_collision_free(self, state: np.ndarray) -> bool:
        """Check state bounds and delegate obstacle checks."""
        if not self._within_bounds(state):
            return False
        return self.base_checker.is_collision_free(state)

    def is_path_collision_free(
        self,
        from_state: np.ndarray,
        to_state: np.ndarray,
        resolution: float | None = None,
    ) -> bool:
        """Check straight-line bounds and delegate collision sampling to the base."""
        if not self._within_bounds(from_state) or not self._within_bounds(to_state):
            return False
        if resolution is None:
            return self.base_checker.is_path_collision_free(from_state, to_state)
        return self.base_checker.is_path_collision_free(from_state, to_state, resolution)
