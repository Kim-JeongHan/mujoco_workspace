"""Planning-space strategies for sampling-based planners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EdgePath:
    """Concrete edge produced by a planning space."""

    states: np.ndarray
    cost: float


class PlanningSpace(ABC):
    """State-space operations used by sampling-based planners."""

    @abstractmethod
    def distance(self, start: np.ndarray, goal: np.ndarray) -> float:
        """Return the distance used for nearest, near, and goal checks."""

    def steer(self, start: np.ndarray, goal: np.ndarray, step_size: float) -> np.ndarray:
        """Steer along the straight chart segment by at most step_size."""
        if step_size <= 0:
            raise ValueError("step_size must be positive")

        direction = goal - start
        distance = float(np.linalg.norm(direction))
        if distance <= step_size:
            return goal.copy()
        return start + direction / distance * step_size

    @abstractmethod
    def edge_states(self, start: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """Return ordered states that make up the edge from start to goal."""

    @abstractmethod
    def edge_cost(self, start: np.ndarray, goal: np.ndarray) -> float:
        """Return the cost of the edge from start to goal."""

    def connect(self, start: np.ndarray, goal: np.ndarray) -> EdgePath:
        """Return the edge states and edge cost."""
        return EdgePath(
            states=self.edge_states(start, goal),
            cost=self.edge_cost(start, goal),
        )


class EuclideanSpace(PlanningSpace):
    """Default Euclidean space matching the historical RRT* behavior."""

    def distance(self, start: np.ndarray, goal: np.ndarray) -> float:
        """Return Euclidean distance between states."""
        return float(np.linalg.norm(goal - start))

    def edge_states(self, start: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """Return the straight edge endpoints."""
        return np.vstack([start, goal])

    def edge_cost(self, start: np.ndarray, goal: np.ndarray) -> float:
        """Return the Euclidean edge cost."""
        return self.distance(start, goal)
