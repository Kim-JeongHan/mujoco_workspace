"""Planner adapters keep search state local to each request."""

from types import SimpleNamespace

import numpy as np
import pytest

from mujoco_lab.planning import (
    PRMConfig,
    PRMPlanner,
    RRTConfig,
    RRTConnectConfig,
    RRTConnectPlanner,
    RRTPlanner,
    planner_from_config,
)


@pytest.mark.parametrize(
    ("config", "adapter", "search_name", "name"),
    [
        (RRTConnectConfig(seed=7), RRTConnectPlanner, "RRTConnect", "rrt_connect"),
        (RRTConfig(seed=7), RRTPlanner, "RRT", "rrt"),
        (PRMConfig(seed=7), PRMPlanner, "PRM", "prm"),
    ],
)
def test_adapter_uses_fresh_search_and_preserves_config(
    monkeypatch, config, adapter, search_name, name
):
    searches = []

    class FakeSearch:
        def __init__(self, start, goal, bounds, checker, query_config):
            self.start = start
            self.goal = goal
            self.bounds = bounds
            self.checker = checker
            self.config = query_config
            searches.append(self)

        def plan(self):
            return [SimpleNamespace(state=self.start), SimpleNamespace(state=self.goal)]

    monkeypatch.setattr("mujoco_lab.planning.planners." + search_name, FakeSearch)
    planner = adapter(config)
    assert planner_from_config(config).name == name
    start, goal = np.zeros(2), np.ones(2)
    bounds = [(-2.0, 2.0)] * 2
    checker = object()
    for seed in (7, 8):
        np.testing.assert_array_equal(
            planner.plan(start, goal, bounds, checker, seed=seed),
            np.vstack((start, goal)),
        )
    assert searches[0] is not searches[1]
    assert searches[0].config is not searches[1].config
    assert [search.config.seed for search in searches] == [7, 8]
    assert config.seed == 7
    assert all(search.checker is checker for search in searches)


def test_adapter_passes_through_no_route(monkeypatch):
    class NoRoute:
        def __init__(self, *args):
            pass

        def plan(self):
            return None

    monkeypatch.setattr("mujoco_lab.planning.planners.RRT", NoRoute)
    planner = RRTPlanner(RRTConfig(seed=None))
    assert planner.plan(np.zeros(2), np.ones(2), [(-2.0, 2.0)] * 2, object(), seed=None) is None
    assert planner.seed is None
