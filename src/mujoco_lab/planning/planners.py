"""Execute independent sampling queries from planner configurations."""

from __future__ import annotations

import numpy as np

from .collision import CollisionChecker
from .sampling import (
    PRM,
    RRG,
    RRT,
    PRMConfig,
    PRMStar,
    PRMStarConfig,
    RRGConfig,
    RRTConfig,
    RRTConnect,
    RRTConnectConfig,
    RRTStar,
    RRTStarConfig,
)

type PlannerConfig = (
    RRTConnectConfig | RRTConfig | PRMConfig | RRTStarConfig | PRMStarConfig | RRGConfig
)

_PLANNER_NAMES = (
    (RRTConnectConfig, "rrt_connect"),
    (RRTConfig, "rrt"),
    (RRTStarConfig, "rrt_star"),
    (PRMStarConfig, "prm_star"),
    (RRGConfig, "rrg"),
    (PRMConfig, "prm"),
)


def default_planning() -> RRTConnectConfig:
    """Return the default manipulation RRT-Connect configuration."""
    return RRTConnectConfig(max_iterations=500, step_size=0.2, goal_tolerance=0.04, seed=7)


def planner_name(config: PlannerConfig) -> str:
    """Return the configured algorithm's name for run metadata and failures."""
    for config_type, name in _PLANNER_NAMES:
        if isinstance(config, config_type):
            return name
    raise TypeError("planning must be a supported planner configuration")


def plan_path(
    config: PlannerConfig,
    start: np.ndarray,
    goal: np.ndarray,
    bounds: list[tuple[float, float]],
    collision_checker: CollisionChecker,
) -> np.ndarray | None:
    """Run a fresh search with the configured seed without modifying its settings.

    Return joint-space vertices, or None for an empty or failed search.
    Extend RRG's near-goal result only over a collision-free edge.
    """
    args = (start, goal, bounds, collision_checker)
    match config:
        case RRTConnectConfig():
            search = RRTConnect(*args, config=config.model_copy())
        case RRTConfig():
            search = RRT(*args, config=config.model_copy())
        case RRTStarConfig():
            search = RRTStar(*args, config=config.model_copy())
        case PRMStarConfig():
            search = PRMStar(*args, config=config.model_copy())
        case RRGConfig():
            search = RRG(*args, config=config.model_copy())
        case PRMConfig():
            search = PRM(*args, config=config.model_copy())
        case _:
            raise TypeError("planning must be a supported planner configuration")
    nodes = search.plan()
    if not nodes:
        return None
    path = np.asarray([node.state for node in nodes], dtype=float)
    if isinstance(config, RRGConfig) and not np.array_equal(path[-1], goal):
        if not collision_checker.is_path_collision_free(path[-1], goal):
            return None
        path = np.vstack((path, goal))
    return path
