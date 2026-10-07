"""Sampling queries preserve configuration and return usable joint-space paths."""

from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pytest

from mujoco_lab.planning import (
    EmptyCollisionChecker,
    PRMConfig,
    PRMStarConfig,
    RRGConfig,
    RRTConfig,
    RRTConnectConfig,
    RRTStarConfig,
    plan_path,
    planner_name,
)


@pytest.mark.parametrize("collision_free", [False, True])
def test_rrg_only_extends_the_exact_goal_over_a_collision_free_edge(monkeypatch, collision_free):
    start, near_goal, goal = np.zeros(2), np.full(2, 0.9), np.ones(2)
    checked = []

    class Search:
        def __init__(self, *args, **kwargs):
            pass

        def plan(self):
            return [SimpleNamespace(state=start), SimpleNamespace(state=near_goal)]

    def check_edge(a, b):
        checked.append((a.copy(), b.copy()))
        return collision_free

    monkeypatch.setattr("mujoco_lab.planning.planners.RRG", Search)
    checker = SimpleNamespace(is_path_collision_free=check_edge)
    path = plan_path(RRGConfig(seed=7), start, goal, [(-2.0, 2.0)] * 2, checker)
    np.testing.assert_array_equal(checked, [(near_goal, goal)])
    if collision_free:
        np.testing.assert_array_equal(path, [start, near_goal, goal])
    else:
        assert path is None


@pytest.mark.parametrize("nodes", [None, []])
@pytest.mark.parametrize(
    "config,search_name", [(RRTConnectConfig(), "RRTConnect"), (RRGConfig(), "RRG")]
)
def test_query_returns_none_for_failed_or_empty_search(monkeypatch, nodes, config, search_name):
    class Search:
        def __init__(self, *args, **kwargs):
            pass

        def plan(self):
            return nodes

    monkeypatch.setattr("mujoco_lab.planning.planners." + search_name, Search)
    assert (
        plan_path(
            config,
            np.zeros(2),
            np.ones(2),
            [(-2.0, 2.0)] * 2,
            EmptyCollisionChecker(),
        )
        is None
    )


@pytest.mark.parametrize(
    "config",
    [
        RRTConnectConfig(max_iterations=20, goal_tolerance=2.0, seed=7),
        RRTConfig(max_iterations=20, goal_bias=1.0, seed=7),
        RRTStarConfig(max_iterations=20, goal_bias=1.0, seed=7),
        PRMConfig(sample_number=8, max_retries=1, radius=10.0, seed=7),
        PRMStarConfig(sample_number=8, max_retries=1, radius_gain=10.0, seed=7),
        RRGConfig(max_iterations=20, goal_bias=1.0, seed=7),
    ],
    ids=planner_name,
)
def test_real_search_returns_a_repeatable_collision_free_path(config):
    start, goal = np.zeros(2), np.ones(2)
    bounds = [(-2.0, 2.0)] * 2
    checker = EmptyCollisionChecker()
    before = config.model_dump()
    path = plan_path(config, start, goal, bounds, checker)
    assert path is not None
    np.testing.assert_allclose(path[0], start)
    np.testing.assert_allclose(path[-1], goal)
    assert np.isfinite(path).all()
    assert all(checker.is_path_collision_free(a, b) for a, b in pairwise(path))
    np.testing.assert_array_equal(plan_path(config, start, goal, bounds, checker), path)
    assert config.model_dump() == before


@pytest.mark.parametrize("config_type", [RRTConfig, RRTStarConfig])
def test_real_search_accepts_near_goal_endpoints(config_type):
    config = config_type(step_size=1.0, goal_tolerance=0.1, goal_bias=1.0, max_iterations=2, seed=7)
    start, goal = np.array([0.0]), np.array([1.000005])
    path = plan_path(config, start, goal, [(-2.0, 2.0)], EmptyCollisionChecker())
    np.testing.assert_array_equal(path, [[0.0], [1.0]])
