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

PLANNERS = [
    (RRTConnectConfig(seed=7), "RRTConnect", "rrt_connect"),
    (RRTConfig(seed=7), "RRT", "rrt"),
    (PRMConfig(seed=7), "PRM", "prm"),
    (RRTStarConfig(seed=7), "RRTStar", "rrt_star"),
    (PRMStarConfig(seed=7), "PRMStar", "prm_star"),
    (RRGConfig(seed=7), "RRG", "rrg"),
]


@pytest.mark.parametrize(("config", "search_name", "name"), PLANNERS)
def test_query_uses_fresh_search_and_preserves_config(monkeypatch, config, search_name, name):
    searches = []

    class FakeSearch:
        def __init__(self, start, goal, bounds, checker, *, config):
            self.start = start
            self.goal = goal
            self.checker = checker
            self.config = config
            searches.append(self)

        def plan(self):
            return [SimpleNamespace(state=self.start), SimpleNamespace(state=self.goal)]

    monkeypatch.setattr("mujoco_lab.planning.planners." + search_name, FakeSearch)
    assert planner_name(config) == name
    start, goal = np.zeros(2), np.ones(2)
    bounds = [(-2.0, 2.0)] * 2
    checker = EmptyCollisionChecker()
    before = config.model_dump()
    for seed in (7, 8, None):
        np.testing.assert_array_equal(
            plan_path(config, start, goal, bounds, checker, seed=seed),
            np.vstack((start, goal)),
        )
    assert searches[0] is not searches[1]
    assert searches[0].config is not searches[1].config
    assert [search.config.seed for search in searches] == [7, 8, None]
    assert config.model_dump() == before
    assert all(search.checker is checker for search in searches)


@pytest.mark.parametrize(("config", "search_name", "name"), PLANNERS)
def test_query_passes_through_no_route(monkeypatch, config, search_name, name):
    class NoRoute:
        def __init__(self, *args, **kwargs):
            pass

        def plan(self):
            return None

    monkeypatch.setattr("mujoco_lab.planning.planners." + search_name, NoRoute)
    assert (
        plan_path(
            config, np.zeros(2), np.ones(2), [(-2.0, 2.0)] * 2, EmptyCollisionChecker(), seed=None
        )
        is None
    )
    assert config.seed == 7


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
    path = plan_path(RRGConfig(seed=7), start, goal, [(-2.0, 2.0)] * 2, checker, seed=8)
    np.testing.assert_array_equal(checked, [(near_goal, goal)])
    if collision_free:
        np.testing.assert_array_equal(path, [start, near_goal, goal])
    else:
        assert path is None


@pytest.mark.parametrize(
    "states",
    [
        [],
        [[0.0]],
        [[0.0, 0.0], [1.0]],
        [[float("nan"), 0.0], [1.0, 1.0]],
        [[0.5, 0.0], [1.0, 1.0]],
        [[0.0, 0.0], [0.9, 0.9]],
    ],
)
def test_query_rejects_malformed_search_results(monkeypatch, states):
    class Search:
        def __init__(self, *args, **kwargs):
            pass

        def plan(self):
            return [SimpleNamespace(state=np.asarray(state)) for state in states]

    monkeypatch.setattr("mujoco_lab.planning.planners.RRTConnect", Search)
    with pytest.raises(RuntimeError, match="Planner"):
        plan_path(
            RRTConnectConfig(),
            np.zeros(2),
            np.ones(2),
            [(-2.0, 2.0)] * 2,
            EmptyCollisionChecker(),
            seed=7,
        )


@pytest.mark.parametrize(
    "config",
    [
        RRTConnectConfig(max_iterations=20, goal_tolerance=2.0),
        RRTConfig(max_iterations=20, goal_bias=1.0),
        RRTStarConfig(max_iterations=20, goal_bias=1.0),
        PRMConfig(sample_number=8, max_retries=1, radius=10.0),
        PRMStarConfig(sample_number=8, max_retries=1, radius_gain=10.0),
        RRGConfig(max_iterations=20, goal_bias=1.0),
    ],
    ids=planner_name,
)
def test_real_search_returns_a_repeatable_collision_free_path(config):
    start, goal = np.zeros(2), np.ones(2)
    bounds = [(-2.0, 2.0)] * 2
    checker = EmptyCollisionChecker()
    path = plan_path(config, start, goal, bounds, checker, seed=7)
    assert path is not None
    np.testing.assert_allclose(path[0], start)
    np.testing.assert_allclose(path[-1], goal)
    assert np.isfinite(path).all()
    assert all(checker.is_path_collision_free(a, b) for a, b in pairwise(path))
    np.testing.assert_array_equal(plan_path(config, start, goal, bounds, checker, seed=7), path)
