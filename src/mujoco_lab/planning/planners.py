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
    *,
    seed: int | None,
) -> np.ndarray | None:
    """Run a fresh search without modifying the supplied configuration.

    Return finite joint-space vertices from the supplied start to the exact
    goal, or None when no route is found. RRG's near-goal result is extended
    only over a collision-free edge. Malformed search output raises RuntimeError.
    """
    args = (start, goal, bounds, collision_checker)
    match config:
        case RRTConnectConfig():
            search = RRTConnect(*args, config=config.model_copy(update={"seed": seed}))
        case RRTConfig():
            search = RRT(*args, config=config.model_copy(update={"seed": seed}))
        case RRTStarConfig():
            search = RRTStar(*args, config=config.model_copy(update={"seed": seed}))
        case PRMStarConfig():
            search = PRMStar(*args, config=config.model_copy(update={"seed": seed}))
        case RRGConfig():
            search = RRG(*args, config=config.model_copy(update={"seed": seed}))
        case PRMConfig():
            search = PRM(*args, config=config.model_copy(update={"seed": seed}))
        case _:
            raise TypeError("planning must be a supported planner configuration")
    nodes = search.plan()
    if nodes is None:
        return None
    try:
        path = np.asarray([node.state for node in nodes], dtype=float)
    except (TypeError, ValueError) as error:
        raise RuntimeError("Planner returned an invalid arm path") from error
    if (
        path.ndim != 2
        or path.shape[1] != len(start)
        or not len(path)
        or not np.isfinite(path).all()
        or not np.allclose(path[0], start, atol=1e-9, rtol=0)
    ):
        raise RuntimeError("Planner returned an invalid arm path")
    if isinstance(config, RRGConfig) and not np.array_equal(path[-1], goal):
        if not collision_checker.is_path_collision_free(path[-1], goal):
            return None
        path = np.vstack((path, goal))
    if not np.allclose(path[-1], goal, atol=1e-9, rtol=0):
        raise RuntimeError("Planner path endpoints do not match the query")
    return path
