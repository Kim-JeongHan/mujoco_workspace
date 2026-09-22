"""Check that the imported planners retain upstream seeded behavior."""

import importlib.util
import sys
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from mujoco_lab import planning


@pytest.fixture(scope="module")
def upstream_planning() -> ModuleType:
    """Load the retained upstream package under a separate package name."""
    source = Path(__file__).resolve().parents[1] / "third_party/planning/planning"
    spec = importlib.util.spec_from_file_location(
        "planning_upstream_reference",
        source / "__init__.py",
        submodule_search_locations=[str(source)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    viser = ModuleType("viser")
    viser.ViserServer = object
    sys.modules[spec.name] = module
    prior_viser = sys.modules.get("viser")
    sys.modules["viser"] = viser
    try:
        spec.loader.exec_module(module)
        importlib.import_module(f"{spec.name}.sampling")
        importlib.import_module(f"{spec.name}.search")
        importlib.import_module(f"{spec.name}.graph")
    finally:
        if prior_viser is None:
            sys.modules.pop("viser", None)
        else:
            sys.modules["viser"] = prior_viser
    return module


@pytest.mark.parametrize(
    ("planner_name", "config_name", "config_kwargs"),
    [
        ("RRT", "RRTConfig", {"max_iterations": 120, "step_size": 0.3, "goal_tolerance": 0.2}),
        (
            "RRTConnect",
            "RRTConnectConfig",
            {"max_iterations": 120, "step_size": 0.3, "goal_tolerance": 0.2},
        ),
        (
            "RRTStar",
            "RRTStarConfig",
            {"max_iterations": 120, "step_size": 0.3, "goal_tolerance": 0.2},
        ),
        (
            "InformedRRTStar",
            "RRTStarConfig",
            {"max_iterations": 120, "step_size": 0.3, "goal_tolerance": 0.2},
        ),
        (
            "RRG",
            "RRGConfig",
            {"max_iterations": 120, "step_size": 0.3, "goal_tolerance": 0.2},
        ),
        ("PRM", "PRMConfig", {"sample_number": 30, "max_retries": 1, "radius": 0.7}),
        (
            "PRMStar",
            "PRMStarConfig",
            {"sample_number": 30, "max_retries": 1, "radius_gain": 5},
        ),
    ],
)
def test_seeded_planner_matches_upstream(
    upstream_planning: ModuleType,
    planner_name: str,
    config_name: str,
    config_kwargs: dict[str, float | int],
) -> None:
    """The copied planners produce the same routes, costs, and graph sizes."""
    upstream_sampling = importlib.import_module(f"{upstream_planning.__name__}.sampling")
    results = []
    for package in (planning, upstream_sampling):
        config = getattr(package, config_name)(seed=11, **config_kwargs)
        planner = getattr(package, planner_name)(
            [0.0, 0.0], [1.0, 1.0], [(0.0, 1.0), (0.0, 1.0)], config=config
        )
        with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
            path = planner.plan()
        assert path is not None
        results.append(
            (
                np.stack([node.state for node in path]),
                planner.get_path_length(),
                planner.get_stats(),
                len(planner.graph.nodes),
                len(planner.graph.edges),
            )
        )

    actual, expected = results
    np.testing.assert_array_equal(actual[0], expected[0])
    assert actual[1:] == expected[1:]


def test_weighted_search_matches_upstream(upstream_planning: ModuleType) -> None:
    """AStar retains upstream's edge-cost-only shortest path behavior."""
    upstream_graph = importlib.import_module(f"{upstream_planning.__name__}.graph")
    upstream_search = importlib.import_module(f"{upstream_planning.__name__}.search")
    for graph_module, search_module in (
        (planning, planning),
        (upstream_graph, upstream_search),
    ):
        start, midpoint, goal = [graph_module.Node([value, 0.0]) for value in (0.0, 1.0, 2.0)]
        graph = graph_module.Graph()
        for node in (start, midpoint, goal):
            graph.add_node(node)
        graph.add_edge(start, goal, 10.0)
        graph.add_edge(start, midpoint, 2.0)
        graph.add_edge(midpoint, goal, 3.0)
        route = search_module.AStar(graph).search(start, goal)
        assert [node.state[0] for node in route] == [0.0, 1.0, 2.0]
