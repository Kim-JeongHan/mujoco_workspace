"""Cube spawn randomization preserves a usable MuJoCo scene."""

import numpy as np
import pytest

from mujoco_lab import RobotSpec, Simulator, create_cube_stack, create_environment
from mujoco_lab.assets.loader import randomize_cube_positions
from mujoco_lab.tasks import CubeStackTask


def _cube_positions(scene):
    return np.array([scene.body(f"cube{i}/object_0").pos for i in range(2)])


def test_randomization_is_seeded_and_compiles_with_forte_task():
    original = create_cube_stack(2)
    scene = create_cube_stack(2)
    same_seed = create_cube_stack(2)
    starts = _cube_positions(original)
    targets = [np.array(scene.body(f"cube{i}/object_target_0").pos) for i in range(2)]
    orientations = [np.array(scene.body(f"cube{i}/object_0").quat) for i in range(2)]

    randomize_cube_positions(scene, seed=12)
    randomize_cube_positions(same_seed, seed=12)
    positions = _cube_positions(scene)
    np.testing.assert_array_equal(positions, _cube_positions(same_seed))
    assert np.any(positions[:, :2] != starts[:, :2])
    assert np.all(np.abs(positions[:, :2] - starts[:, :2]) <= 0.02)
    np.testing.assert_array_equal(positions[:, 2], starts[:, 2])
    for i in range(2):
        np.testing.assert_array_equal(scene.body(f"cube{i}/object_0").quat, orientations[i])
        np.testing.assert_array_equal(scene.body(f"cube{i}/object_target_0").pos, targets[i])

    simulator = Simulator(scene, robots=[RobotSpec("forte", "forte")])
    task = CubeStackTask(simulator, 2)
    for i in range(2):
        world_start = simulator.data.body(f"cube{i}/object_0").xpos
        np.testing.assert_allclose(world_start, positions[i] + [0, 0, 0.8])
        np.testing.assert_allclose(task.starts[i], world_start)
        np.testing.assert_allclose(task.goals[i], targets[i] + [0, 0, 0.8])
    simulator.reset()
    for i in range(2):
        np.testing.assert_allclose(simulator.data.body(f"cube{i}/object_0").xpos, task.starts[i])


def test_crowded_cubes_keep_physical_geom_gap():
    scene = create_cube_stack(2)
    first = np.array(scene.body("cube0/object_0").pos)
    scene.body("cube1/object_0").pos = first + [0.015, 0, 0]

    randomize_cube_positions(scene, xy_range=0.08, min_gap=0.01, seed=5)
    simulator = Simulator(scene)
    model, data = simulator.model, simulator.data
    centers = np.array([data.geom(f"cube{i}/object_0").xpos[:2] for i in range(2)])
    half_sizes = np.array([model.geom(f"cube{i}/object_0").size[:2] for i in range(2)])
    assert np.any(np.abs(centers[0] - centers[1]) >= half_sizes.sum(axis=0) + 0.01)


def test_impossible_layout_does_not_move_any_cube_or_target():
    scene = create_cube_stack(2)
    first = np.array(scene.body("cube0/object_0").pos)
    # Centers are 3 cm apart: separated points, but overlapping 4 cm cubes.
    scene.body("cube1/object_0").pos = first + [0.03, 0, 0]
    before = _cube_positions(scene)
    targets = [np.array(scene.body(f"cube{i}/object_target_0").pos) for i in range(2)]

    with pytest.raises(ValueError, match="Could not place cube"):
        randomize_cube_positions(scene, xy_range=0.005, min_gap=0.01, seed=2)
    np.testing.assert_array_equal(_cube_positions(scene), before)
    for i in range(2):
        np.testing.assert_array_equal(scene.body(f"cube{i}/object_target_0").pos, targets[i])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"xy_range": -0.01},
        {"xy_range": np.inf},
        {"min_gap": -0.01},
        {"min_gap": np.nan},
        {"max_attempts": 0},
    ],
)
def test_invalid_arguments_are_rejected(kwargs):
    with pytest.raises(ValueError):
        randomize_cube_positions(create_cube_stack(2), **kwargs)


def test_scene_without_cube_objects_is_rejected():
    with pytest.raises(ValueError, match="no cube object bodies"):
        randomize_cube_positions(create_environment("empty"))
