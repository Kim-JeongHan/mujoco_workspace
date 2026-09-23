"""Reusable path planner adapters for one-shot sampling searches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .collision import CollisionChecker
from .graph import Node
from .sampling import PRM, RRT, PRMConfig, RRTConfig, RRTConnect, RRTConnectConfig


class PathPlanner(Protocol):
    """Plan independent joint-space queries against a supplied scene."""

    @property
    def name(self) -> str:
        """Return the planner name for run metadata and failures."""

    @property
    def seed(self) -> int | None:
        """Return the base seed for consecutive stage queries."""

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        bounds: list[tuple[float, float]],
        collision_checker: CollisionChecker,
        *,
        seed: int | None,
    ) -> np.ndarray | None:
        """Return a path from start to goal, or None when no route exists."""


def _states(nodes: list[Node] | None) -> np.ndarray | None:
    if nodes is None:
        return None
    return np.asarray([node.state for node in nodes], dtype=float)


@dataclass(frozen=True)
class RRTConnectPlanner:
    """Run a fresh RRT-Connect search for each query."""

    config: RRTConnectConfig
    name = "rrt_connect"

    @property
    def seed(self) -> int | None:
        return self.config.seed

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        bounds: list[tuple[float, float]],
        collision_checker: CollisionChecker,
        *,
        seed: int | None,
    ) -> np.ndarray | None:
        config = self.config.model_copy(update={"seed": seed})
        return _states(RRTConnect(start, goal, bounds, collision_checker, config).plan())


@dataclass(frozen=True)
class RRTPlanner:
    """Run a fresh RRT search for each query."""

    config: RRTConfig
    name = "rrt"

    @property
    def seed(self) -> int | None:
        return self.config.seed

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        bounds: list[tuple[float, float]],
        collision_checker: CollisionChecker,
        *,
        seed: int | None,
    ) -> np.ndarray | None:
        config = self.config.model_copy(update={"seed": seed})
        return _states(RRT(start, goal, bounds, collision_checker, config).plan())


@dataclass(frozen=True)
class PRMPlanner:
    """Run a fresh PRM search for each query."""

    config: PRMConfig
    name = "prm"

    @property
    def seed(self) -> int | None:
        return self.config.seed

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        bounds: list[tuple[float, float]],
        collision_checker: CollisionChecker,
        *,
        seed: int | None,
    ) -> np.ndarray | None:
        config = self.config.model_copy(update={"seed": seed})
        return _states(PRM(start, goal, bounds, collision_checker, config).plan())


def planner_from_config(config: RRTConnectConfig | RRTConfig | PRMConfig) -> PathPlanner:
    """Translate a serializable planner configuration at an application boundary."""
    if isinstance(config, RRTConnectConfig):
        return RRTConnectPlanner(config)
    if isinstance(config, RRTConfig):
        return RRTPlanner(config)
    if isinstance(config, PRMConfig):
        return PRMPlanner(config)
    raise TypeError("planning must be an RRTConnectConfig, RRTConfig, or PRMConfig")
